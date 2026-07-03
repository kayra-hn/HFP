import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from .configuration_hfp import HFPConfig

from ..core.hfp_bulk_state import HFPBulkState
from ..core.bulk_trigger_decoder import BulkTriggerDecoderLayer

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class HFPPreTrainedModel(PreTrainedModel):
    config_class = HFPConfig
    base_model_prefix = "hfp"
    _supports_cache_class = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
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
        self.pos_encoder = SinusoidalPositionalEncoding(config.hidden_size, max_len=config.max_position_embeddings)
        
        self.layers = nn.ModuleList([
            BulkTriggerDecoderLayer(
                hidden_size=config.hidden_size, 
                num_heads=config.num_attention_heads, 
                feedforward_dim=config.intermediate_size, 
                bulk_dim=config.bulk_dim,
                vocab_size=None
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Her katman için kendi 5D Bulk hafızası
        self.bulk_states = nn.ModuleList([
            HFPBulkState(
                hidden_size=config.hidden_size,
                short_len=config.short_len,
                medium_freq=config.medium_freq,
                long_freq=config.long_freq,
                medium_momentum=config.medium_momentum
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        x = self.embed_tokens(input_ids)
        x = self.pos_encoder(x)
        
        if past_key_values is None or not isinstance(past_key_values, (tuple, list)):
            past_key_values_list = [None] * len(self.layers)
        else:
            past_key_values_list = past_key_values
            
        new_past_key_values = []
        for i, (layer, bulk_state) in enumerate(zip(self.layers, self.bulk_states)):
            x, _, new_past_state = layer(x, bulk_state, past_state=past_key_values_list[i], return_past_state=True)
            
            if use_cache:
                new_past_key_values.append(new_past_state)
            
        x = self.norm(x)
        
        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=new_past_key_values if use_cache else None
        )

from transformers.generation import GenerationMixin

class HFPForCausalLM(HFPPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.hfp = HFPModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None, use_cache=False, **kwargs):
        outputs = self.hfp(input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, **kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift yapıldı (Önceki token, sonrakini tahmin etmeye çalışıyor)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values
        )
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values:
            # Eğer cache varsa, sadece en son eklenen token'ı ileri (forward) geçir
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True)
        }
