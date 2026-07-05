# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from .configuration_hfp import HFPConfig

from ..core.hfp_bulk_state import HFPBulkState
from ..core.bulk_trigger_decoder import BulkTriggerDecoderLayer

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000, pe_scale=0.3):
        super().__init__()
        self.max_len = max_len
        # [FIX K7] PE olcegi. Onceden ham (norm=sqrt(d/2)=8) eklenirken embedding
        # normu 0.02*sqrt(d)=0.23 idi -> PE, token icerigini ~35x BOGUYORDU;
        # anahtar/deger'ler ~%97 pozisyon oluyor, icerik-tabanli recall imkansiz
        # (MQAR loss ln(val_space)'te sabitlenip binding hic ogrenilmiyordu).
        # embed *sqrt(d) (bkz. HFPModel) + PE *0.3 ile normlar dengelenir.
        self.pe_scale = pe_scale
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, offset: int = 0):
        # [FIX A2] Streaming/chunked prefill'de her chunk'a 0..L degil, GLOBAL pozisyon eklenir.
        seq_len = x.size(1)
        if offset + seq_len > self.max_len:
            offset = max(0, self.max_len - seq_len)
        return x + self.pe_scale * self.pe[:, offset:offset + seq_len, :].to(x.device)

class HFPPreTrainedModel(PreTrainedModel):
    config_class = HFPConfig
    base_model_prefix = "hfp"
    _supports_cache_class = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                # importance_gate bias'i bilerek -2.0 (gate-collapse onlemi); ezme.
                if torch.all(module.bias.data == -2.0):
                    pass
                else:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class HFPModel(HFPPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # [FIX K7] Vaswani sqrt(d) olcegi: emb (init std 0.02) tek basina norm~0.23;
        # sqrt(d) ile ~2.56'ya cikar ve sonuchennmus PE (*0.3, norm~2.4) ile dengelenir.
        self.embed_scale = math.sqrt(config.hidden_size)
        self.pos_encoder = SinusoidalPositionalEncoding(
            config.hidden_size, max_len=config.max_position_embeddings,
            pe_scale=getattr(config, "pe_scale", 0.3))

        self.layers = nn.ModuleList([
            BulkTriggerDecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                feedforward_dim=config.intermediate_size,
                bulk_dim=config.bulk_dim,
                vocab_size=None,
                local_window=getattr(config, "local_window", None),   # [FIX K5]
                dropout_p=getattr(config, "dropout_p", 0.1),          # [FIX K3]
                ffn_type=getattr(config, "ffn_type", "entangled")     # [HFP-SCALE]
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Katman-basi recurrent bellek (physics-inspired 'Bulk' analojisi;
        # teknik olarak: decay'li lineer-attention state'i M, z)
        # [TEMIZLIK B1] medium_freq/long_freq/medium_momentum kaldirildi.
        self.bulk_states = nn.ModuleList([
            HFPBulkState(
                hidden_size=config.hidden_size,
                short_len=config.short_len,
                max_short_len=getattr(config, "max_short_len", None), # [FIX K4]
                rec_block=getattr(config, "rec_block", 64),           # [K2]
                decay_mode=getattr(config, "decay_mode", "exp"),      # [HFP-CORE]
                conv_kernel=getattr(config, "conv_kernel", 3),        # [FIX K8] binding
                key_feature_map=getattr(config, "key_feature_map", "elu"),  # [HFP-CAP]
                dpfp_nu=getattr(config, "dpfp_nu", 2),                # [HFP-CAP]
                write_rule=getattr(config, "write_rule", "additive")  # [HFP-DELTA]
            )
            for _ in range(config.num_hidden_layers)
        ])
        # [K2] TBPTT: True ise chunk'lar arasi state detach edilmez
        self._detach_state = not getattr(config, "bptt_across_chunks", False)

        self.norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    @staticmethod
    def _offset_from_state(past_key_values_list):
        # state tuple: (short_memory, M, z, token_count, short_len_dynamic, write_idx, conv_state)
        first = past_key_values_list[0]
        if first is not None and len(first) >= 4 and isinstance(first[3], int):
            return int(first[3])
        return 0

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        x = self.embed_tokens(input_ids) * self.embed_scale  # [FIX K7]

        if past_key_values is None or not isinstance(past_key_values, (tuple, list)):
            past_key_values_list = [None] * len(self.layers)
        else:
            past_key_values_list = past_key_values

        # [FIX A2] Global pozisyon offseti (chunked prefill icin)
        offset = self._offset_from_state(past_key_values_list)
        x = self.pos_encoder(x, offset=offset)

        new_past_key_values = []
        gate_entropies = []
        for i, (layer, bulk_state) in enumerate(zip(self.layers, self.bulk_states)):
            x, _, new_past_state = layer(x, bulk_state, past_state=past_key_values_list[i],
                                         return_past_state=True, detach_state=self._detach_state)

            # [C1] Gradyanli gate-entropy topla (yalnizca agirlik > 0 iken loss'a katilir)
            if self.training and getattr(bulk_state, "_gate_entropy_live", None) is not None:
                gate_entropies.append(bulk_state._gate_entropy_live)

            if use_cache:
                new_past_key_values.append(new_past_state)

        x = self.norm(x)

        out = BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=new_past_key_values if use_cache else None
        )
        out.gate_entropy = (torch.stack(gate_entropies).mean() if gate_entropies else None)
        return out

from transformers.generation import GenerationMixin

class HFPForCausalLM(HFPPreTrainedModel, GenerationMixin):
    # [D1] lm_head, embed_tokens'a bagli (weight tying) — safetensors kaydinin
    # paylasilan tensoru dogru ele almasi icin bildirilmeli.
    # transformers v5: dict (hedef -> kaynak); v4'te de sorunsuz calisir.
    _tied_weights_keys = {"lm_head.weight": "hfp.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.hfp = HFPModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # [D1] Weight tying (GPT standardi): embedding ve lm_head paylasimi
        self.lm_head.weight = self.hfp.embed_tokens.weight
        self.post_init()

    # [D1] tie_weights()'in from_pretrained SONRASI baglantiyi yeniden kurabilmesi
    # icin standart erisimciler (bunlarsiz yukleme tying'i sessizce koparir).
    def get_input_embeddings(self):
        return self.hfp.embed_tokens

    def set_input_embeddings(self, value):
        self.hfp.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, **kwargs):
        outputs = self.hfp(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            # [C1] Opsiyonel gate-entropy duzenleyici - default kapali (weight=0.0).
            w = float(getattr(self.config, "aux_gate_entropy_weight", 0.0))
            if w > 0.0 and getattr(outputs, "gate_entropy", None) is not None:
                loss = loss + w * outputs.gate_entropy

            # [ORTHO] EntangledFFN ortogonallik duzenleyicisi. Paylasilan W_bulk'tan
            # turetilen P_A/P_B'nin farkli ("tek bulk'in iki ayri golgesi") kalmasi
            # icin baski. Onceden implement edilmis ama loss'a HIC bagli degildi
            # (olu kod); artik opsiyonel + default kapali (weight=0.0 => baseline degismez).
            w_o = float(getattr(self.config, "aux_ortho_weight", 0.0))
            if w_o > 0.0:
                ortho = sum(layer.ffn.get_orthogonality_loss() for layer in self.hfp.layers)
                loss = loss + w_o * ortho

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True)
        }
