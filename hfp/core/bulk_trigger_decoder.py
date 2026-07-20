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
import torch.nn.functional as F
import math
from .hfp_config import config as hfp_config
from .hfp_utils import compute_curvature, magnitude_defect_flag, coherence_score, conservation_check, holographic_information_bound
from .hfp_bulk_state import HFPBulkState

class HFPLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(HFPLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# [FIX K3] TunnelingDropout KALDIRILDI: 3 forward onceki, FARKLI batch'e ait
# detached aktivasyonlari simdiki ciktiya ekliyordu -> batch'ler arasi sizinti +
# train/eval davranis farki. Ne dropout ne fizik; standart Dropout kullanilir.
# (Eski kod referans icin _legacy_reference/ altinda.)

class EntangledLinear(nn.Module):
    """Tek Bulk agirligindan (W_bulk) iki projeksiyon (P_A, P_B) - physics-inspired
    parametre baglama. Analoji: Paper II'nin 'tek Bulk vektorunun iki golgesi';
    izomorfizm/simulasyon iddiasi degildir."""
    def __init__(self, in_features_A, out_features_A, in_features_B, out_features_B, bulk_dim=128):
        super(EntangledLinear, self).__init__()
        self.max_in = max(in_features_A, in_features_B)
        self.W_bulk = nn.Parameter(torch.randn(bulk_dim, self.max_in) / math.sqrt(self.max_in))

        self.P_A = nn.Parameter(torch.randn(out_features_A, bulk_dim) / math.sqrt(bulk_dim))
        self.P_B = nn.Parameter(torch.randn(out_features_B, bulk_dim) / math.sqrt(bulk_dim))

        self.bias_A = nn.Parameter(torch.zeros(out_features_A))
        self.bias_B = nn.Parameter(torch.zeros(out_features_B))

    def get_orthogonality_loss(self):
        dot = self.P_A @ self.P_B.t()
        return torch.norm(dot, p='fro')

    def forward_A(self, x):
        if not self.training:
            if not hasattr(self, 'W_A_cache'):
                self.W_A_cache = self.P_A @ self.W_bulk[:, :x.size(-1)]
            W_A = self.W_A_cache
        else:
            if hasattr(self, 'W_A_cache'):
                del self.W_A_cache
            W_A = self.P_A @ self.W_bulk[:, :x.size(-1)]
        return F.linear(x, W_A, self.bias_A)

    def forward_B(self, x):
        if not self.training:
            if not hasattr(self, 'W_B_cache'):
                self.W_B_cache = self.P_B @ self.W_bulk[:, :x.size(-1)]
            W_B = self.W_B_cache
        else:
            if hasattr(self, 'W_B_cache'):
                del self.W_B_cache
            W_B = self.P_B @ self.W_bulk[:, :x.size(-1)]
        return F.linear(x, W_B, self.bias_B)

class EntangledFFN(nn.Module):
    def __init__(self, hidden_size, feedforward_dim, bulk_dim=128, dropout_p=0.1):
        super(EntangledFFN, self).__init__()
        self.entangled = EntangledLinear(hidden_size, feedforward_dim, feedforward_dim, hidden_size, bulk_dim)
        self.gelu = nn.GELU()
        # [FIX K3] Standart dropout (TunnelingDropout'un yerine)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        mid = self.entangled.forward_A(x)
        mid = self.gelu(mid)
        mid = self.dropout(mid)
        out = self.entangled.forward_B(mid)
        return out

    def get_orthogonality_loss(self):
        return self.entangled.get_orthogonality_loss()

class StandardFFN(nn.Module):
    """[HFP-SCALE] Rank kisiti olmayan standart Transformer FFN'i.
    EntangledFFN paylasilan W_bulk yuzunden rank<=bulk_dim darbogazi tasir
    (or. bulk_dim=128, H=768'de FFN rank-128'e sikisir). Olcekleme kosulari
    icin ffn_type="standard" bu darbogazi kaldirir. Parametre sayisi
    EntangledFFN'den fazladir; A/B kiyaslarinda parametre esitligine dikkat."""
    def __init__(self, hidden_size, feedforward_dim, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))

    def get_orthogonality_loss(self):
        return torch.zeros((), device=self.fc1.weight.device)

class BulkTriggerDecoderLayer(nn.Module):
    """
    BulkTriggerDecoderLayer V3: Lokal (pencereli) attention + recurrent Bulk hafiza.

    Mimari niyet (eski V2 yorumundaki 'Local Attention over Brane ONLY') artik
    gercekten uygulanir: [FIX K5]
    - local_window=None -> tam causal attention (eski davranis, geriye uyumlu).
    - local_window=w    -> her sorgu yalnizca son w tokeni gorur; uzun menzil
      bilgi YALNIZCA Bulk hafizadan (M, z) akabilir. Bellek iddialarini test
      etmek icin bu mod sarttir (aksi halde attention tum baglami gorur ve
      bellek olculmez).
    - Ring buffer'in yazilmamis (sifir) slotlari artik MASKELENIR (eski D2 sorunu).
    """
    def __init__(self, hidden_size, num_heads, feedforward_dim, bulk_dim=128,
                 vocab_size=None, return_aux=False, local_window=None, dropout_p=0.1,
                 ffn_type="entangled"):
        super(BulkTriggerDecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.local_window = local_window

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.return_aux = return_aux
        self.norm1 = nn.LayerNorm(hidden_size)

        # [HFP-SCALE] ffn_type: "entangled" (parametre-bagli, rank<=bulk_dim) |
        # "standard" (kisitsiz, olcekleme onerilen)
        if ffn_type == "standard":
            self.ffn = StandardFFN(hidden_size, feedforward_dim, dropout_p=dropout_p)
        else:
            self.ffn = EntangledFFN(hidden_size, feedforward_dim, bulk_dim=bulk_dim, dropout_p=dropout_p)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.lm_head = HFPLinear(hidden_size, vocab_size)
        else:
            self.lm_head = None

    def _build_mask(self, seq_len, n_past, valid_past, device):
        """True = maskeli. Sutunlar: [simdiki chunk (seq_len) | ring buffer (n_past)]."""
        ii = torch.arange(seq_len, device=device).view(-1, 1)
        jj = torch.arange(seq_len, device=device).view(1, -1)
        causal = jj > ii
        if self.local_window is not None:
            # [K5] Sliding window: yalnizca son w token gorulur
            causal = causal | (jj <= ii - self.local_window)
        if n_past > 0:
            # [K5/D2] Yazilmamis (sifir) slotlar maskelenir. Buffer dolana kadar
            # yazim sirasi 0,1,2,... oldugundan gecerli slotlar ilk valid_past tanedir.
            past_cols = (torch.arange(n_past, device=device) >= valid_past).view(1, -1)
            past_mask = past_cols.expand(seq_len, n_past)
            return torch.cat([causal, past_mask], dim=1)
        return causal

    def forward(self, x, bulk_state, past_state=None, return_past_state=False,
                return_aux=None, detach_state=True):
        if return_aux is None:
            return_aux = getattr(self, 'return_aux', False)

        # 1. Recurrent Bulk hafiza guncelle + oku ([K2] artik gradyanli yol)
        short_mem, retrieved_memory, new_past_state = bulk_state.update(
            x, past_state=past_state, detach_state=detach_state)

        aux_losses = []

        # Opsiyonel physics-inspired aux teshisleri (default kapali)
        if hfp_config.ENABLE_RYU_TAKAYANAGI:
            gate_entropy_tensor = bulk_state.gate_entropy_loss() / hfp_config.REG_WEIGHT if hfp_config.ENABLE_ENTROPY_MAP else torch.tensor(0.0, device=x.device)
            M_matrix = new_past_state[1]
            rt_loss = holographic_information_bound(gate_entropy_tensor, M_matrix)
            aux_losses.append(rt_loss.mean().unsqueeze(0))

        if hfp_config.ENABLE_ENTROPY_MAP:
            aux_losses.append(bulk_state.gate_entropy_loss())

        if hfp_config.ENABLE_5D_CURVATURE or hfp_config.ENABLE_CURVATURE:
            aux_losses.append(compute_curvature(short_mem).unsqueeze(0))

        if hfp_config.ENABLE_DEFECT_FLAG:
            aux_losses.append(magnitude_defect_flag(short_mem).mean().unsqueeze(0))
        if hfp_config.ENABLE_COHERENCE:
            aux_losses.append(coherence_score(short_mem).unsqueeze(0))
        if hfp_config.ENABLE_CONSERVATION:
            aux_losses.append(torch.tensor(1.0 if conservation_check(short_mem) else 0.0, device=short_mem.device))

        # 2. Lokal attention: simdiki chunk + (varsa) onceki chunk'larin ring buffer'i
        seq_len = x.size(1)
        if past_state is not None and past_state[0] is not None:
            past_short_mem = past_state[0]
            # [K5] state'teki token_count (index 3) gecerli slot sayisini verir
            valid_past = min(int(past_state[3]), past_short_mem.size(1))
        else:
            past_short_mem = None
            valid_past = 0

        if past_short_mem is not None and valid_past > 0:
            memory_bank = torch.cat([x, past_short_mem], dim=1)
            n_past = past_short_mem.size(1)
        else:
            memory_bank = x
            n_past = 0

        dual_mask = self._build_mask(seq_len, n_past, valid_past, x.device)
        attn_out, _ = self.cross_attention(query=x, key=memory_bank, value=memory_bank, attn_mask=dual_mask)

        # 3. Bulk hafizadan okunan icerik eklenir
        attn_out = attn_out + retrieved_memory

        x = self.norm1(x + attn_out)

        # 4. FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_aux:
            aux_losses.append(self.ffn.get_orthogonality_loss().unsqueeze(0))

        # 5. Logits
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x

        if return_aux:
            return logits, bulk_state, new_past_state, aux_losses
        if return_past_state:
            return logits, bulk_state, new_past_state
        return logits, bulk_state

if __name__ == "__main__":
    batch_size = 2
    hidden_size = 256
    num_heads = 8
    feedforward_dim = 1024
    vocab_size = 50000

    layer = BulkTriggerDecoderLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        vocab_size=vocab_size
    )
    memory_system = HFPBulkState(hidden_size=hidden_size)
    current_token = torch.randn(batch_size, 1, hidden_size)
    logits, updated_memory = layer(current_token, memory_system)

    print(f"Girdi Boyutu: {current_token.shape}")
    print(f"Logits Çıktı Boyutu: {logits.shape}")
