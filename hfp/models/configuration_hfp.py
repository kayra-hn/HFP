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

from transformers import PretrainedConfig

class HFPConfig(PretrainedConfig):
    model_type = "hfp"

    def __init__(
        self,
        vocab_size=5000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        bulk_dim=128,
        tunnel_depth=3,
        tunnel_decay=0.8,
        dropout_p=0.1,
        short_len=8,
        max_short_len=None,             # [FIX K4] ring buffer kapasitesi; None -> max(short_len, 32)
        local_window=None,              # [FIX K5] None=tam causal attention; w=yalnizca son w token
                                        #   (bellek testleri icin sart: aksi halde attention tum baglami gorur)
        pe_scale=0.3,                   # [FIX K7] pozisyonel kodlama olcegi; ham PE (1.0) token
                                        #   icerigini ~35x boguyordu -> recall imkansizdi. 0.3 = dengeli.
        rec_block=64,                   # [K2] chunk-ici recurrence blok boyutu (hiz/bellek; sonucu degistirmez)
        decay_mode="exp",               # [HFP-CORE] "exp"=geometrik decay baseline; "cubic_flux"=
                                        #   makalenin dth/dtau=-eta*th^3 kubik-plato retention'i (ayirt edici);
                                        #   "cubic_flux_chunked"=[HFP-SCALE] iki-gecisli TAM paralel form
                                        #   (z-taramasi + GLA-tarzi chunkwise M; her rec_block'ta birebir)
        conv_kernel=3,                  # [FIX K8] binding conv kernel'i (Q/K token-karisimi). 1=kapali (ablasyon)
        key_feature_map="elu",          # [HFP-CAP] bellek anahtar ozellik-haritasi. "elu"=elu+1 (baseline,
                                        #   D=H). "dpfp"=Deterministic Parameter-Free Projection (D=2H*nu):
                                        #   efektif boyutu buyutur -> rank-collapse'i geciktirir, KAPASITE artar.
        dpfp_nu=2,                      # [HFP-CAP] dpfp genisleme faktoru (key_dim = 2*hidden_size*nu)
        bptt_across_chunks=False,       # [K2] True -> chunk'lar arasi state detach edilmez (TBPTT)
        max_position_embeddings=8192,   # [FIX A1] onceden tanimsizdi -> from_1b_profile cokuyordu
        aux_gate_entropy_weight=0.0,    # [C1] opsiyonel gate-entropy duzenleyici; 0.0 = kapali (durust baseline)
        write_rule="additive",          # [HFP-DELTA] "additive"=k v^T toplama (baseline);
                                        #   "delta"=olcum-guncelleme yazimi (girisim-dirençli, sirali)
        ffn_type="entangled",         # [HFP-SCALE] "entangled"=paylasilan-bulk FFN (rank<=bulk_dim);
                                        #   "standard"=kisitsiz FFN (olcekleme icin onerilir)
        aux_ortho_weight=0.0,           # [ORTHO] EntangledFFN P_A/P_B ortogonallik cezasi; 0.0 = kapali
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.bulk_dim = bulk_dim
        self.tunnel_depth = tunnel_depth
        self.tunnel_decay = tunnel_decay
        self.dropout_p = dropout_p
        self.short_len = short_len
        self.max_short_len = max_short_len
        self.local_window = local_window
        self.pe_scale = pe_scale
        self.rec_block = rec_block
        self.decay_mode = decay_mode
        self.conv_kernel = conv_kernel
        self.key_feature_map = key_feature_map
        self.dpfp_nu = dpfp_nu
        self.bptt_across_chunks = bptt_across_chunks
        self.max_position_embeddings = max_position_embeddings
        self.aux_gate_entropy_weight = aux_gate_entropy_weight
        self.write_rule = write_rule
        self.ffn_type = ffn_type
        self.aux_ortho_weight = aux_ortho_weight
        # [TEMIZLIK B1] medium_freq / long_freq / medium_momentum kaldirildi:
        # mimari lineer-attention (M, z + decay) oldugundan bunlarin hicbir entegrasyonu yok.
        self.ENABLE_COHERENCE = kwargs.pop("ENABLE_COHERENCE", False)
        # [D1] Weight tying resmen bildirilir: transformers v5 bunu gormezse
        # from_pretrained sonrasi lm_head/embedding baglantisini KURMAZ.
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @classmethod
    def from_1b_profile(cls, vocab_size=50257):
        """Creates a 1 Billion Parameter configuration for Cloud Training."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=8192,
            bulk_dim=512,
            short_len=64,
            max_short_len=64,                # [FIX K4] eskiden sessizce 32'ye kirpiliyordu
            max_position_embeddings=32768,   # [FIX A1] 1B profili icin pozisyon tavani
            ENABLE_COHERENCE=False
        )
