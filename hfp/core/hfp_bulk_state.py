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
import logging
from typing import List

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
from .hfp_utils import LandmarkBuffer, compute_gate_entropy, coherence_score
from .hfp_config import config as hfp_config


@torch.jit.script
def _cubic_zscan(K: torch.Tensor, eta: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """[HFP-SCALE HIZ] cubic_flux z-taramasi, TorchScript-derlenmis (birebir eager ile ayni).
        lam_t = 1/sqrt(1 + 2*eta*z_{t-1}^2) ;  z_t = lam_t*z_{t-1} + k_t
    Dogrusal-OLMAYAN recurrence (lam z'ye bagli) -> associative/paralel-scan MUMKUN DEGIL;
    ama loop'u fuze edip Python-yorumlayici overhead'ini kaldirir (uzun L'de ~2-4x hiz,
    matematik degismez). K:(B,L,D) eta:(1,D) z0:(B,D) -> lam_seq:(B,L,D)."""
    z = z0
    lam_list: List[torch.Tensor] = []
    L = int(K.shape[1])
    for t in range(L):
        lam = 1.0 / torch.sqrt(1.0 + 2.0 * eta * z * z)
        lam_list.append(lam)
        z = z * lam + K[:, t]
    return torch.stack(lam_list, dim=1)


class HFPBulkState(nn.Module):
    """
    HFPBulkState V4 (Recurrent Edition): Causal chunkwise linear attention.

    [FIX K2 - GRADYAN AKISI] Onceki surumde retrieval, M guncellemesinden ONCE
    yapiliyordu; tek-parca egitimde M=0 oldugundan retrieval hep sifirdi ve
    W_k/W_v/decay/importance_gate LM loss'tan HIC gradyan alamiyordu (bellek
    egitimde olu agirlikti). Bu surum gercek causal lineer attention'dir:
    her token, o ana KADARKI kumulatif M/z'den okur (kendi KV'si dahil),
    per-token decay ile. Boylece bellek yolu ayni forward icinde ciktiya
    baglanir ve TUM bellek parametreleri gradyan alir.

    Matematik (RetNet/GLA tarzi chunkwise form, tum usler >= 0 -> stabil):
        lam = sigmoid(decay)                          (K-kanali basina, 0..1)
        M_t = lam (.) M_{t-1} + k_t v_t^T ,  z_t = lam (.) z_{t-1} + k_t
        out_t = (q_t M_t) / (q_t . z_t)
    Blok ici (m token):
        cross:  (q_i * lam^i) M_0            intra:  S_ij = q_i . (lam^{i-j} (.) k_j), j<=i
    Uretim yolu (1 token/cagri) ayni formulun m=1 halidir -> egitim/uretim
    decay semantigi artik TUTARLI (eski surum decay'i cagri basina 1 kez
    uyguluyordu; 256-token chunk ile 1-token generate farkli davraniyordu).

    Onceki yapisal duzeltmeler korunur:
    - Matrix blowup: decay init sigmoid(2.19)~0.9 + retrieval LayerNorm.
    - Gate collapse: importance_gate bias -2.0.
    - Ring buffer: sabit boyutlu, vektorize yazim (Python token-dongusu kaldirildi).
    [FIX K4] max_short_len artik parametre: config.short_len > 32 sessizce
    kirpilmiyor (1B profili short_len=64 gercekten 64 slot alir).
    [FIX D3] batch>16'da sessiz half() donusumu kaldirildi.
    """

    def __init__(self, hidden_size, short_len=8, max_short_len=None,
                 rec_block=64, use_mixed_precision=False, clip_value=1.0,
                 decay_mode="exp", conv_kernel=3, key_feature_map="elu", dpfp_nu=2,
                 write_rule="additive"):
        super(HFPBulkState, self).__init__()
        self.hidden_size = hidden_size
        self.base_short_len = short_len
        # [HFP-CAP] Anahtar ozellik-haritasi ve efektif anahtar boyutu (key_dim).
        # "elu": elu(x)+1, key_dim=H (baseline). "dpfp": Deterministic Parameter-Free
        # Projection, key_dim=2*H*nu -> daha yuksek efektif boyut, rank-collapse
        # geciktirilir, bellek KAPASITESI (kac ayri olgu) artar. M artik (key_dim, H),
        # z (key_dim); deger (V) boyutu H olarak kalir. Retention (exp/cubic) ve
        # binding conv'dan BAGIMSIZ eksen.
        self.key_feature_map = key_feature_map
        self.dpfp_nu = max(1, dpfp_nu)
        self.key_dim = hidden_size if key_feature_map != "dpfp" else 2 * hidden_size * self.dpfp_nu
        # [HFP-CORE] Retention yasasi. "exp" = standart geometrik decay (RetNet/GLA
        # ailesi, baseline). "cubic_flux" = makalenin dth/dtau=-eta*th^3 kubik
        # akisinin birebir ayriklastirmasi: state-buyuklugune bagli, plato+power-law
        # unutma. Ailedeki hicbir modelde olmayan ayirt edici mekanizma.
        self.decay_mode = decay_mode
        # [HFP-DELTA] Yazim kurali. "additive" = M += k(x)v^T (baseline; ayni anahtara
        # tekrarli yazimlar GIRISIM yapar). "delta" = DeltaNet-tarzi olcum-guncelleme:
        #     M~ = lam (.) M ;  v_old = k^T M~ ;  M = M~ + beta * k (v - v_old)^T
        # Eski iliskiyi okuyup FARKI yazar -> ayni anahtarin eski degeri silinir,
        # girisim birikmez. k L2-normalize edilir, beta=sigmoid(gate) in (0,1) ->
        # (I - beta k k^T) kontraksiyon, state patlamaz. Payda (q.z) kullanilmaz
        # (delta'da kutle birikimi anlamsiz); cikti q.M -> retrieval_norm olcekler.
        # Sirali O(L) (WY/chunkwise formu ileriki is; GPU olcek icin gerekli).
        self.write_rule = write_rule

        # [FIX K4] Kapasite en az short_len; eskisi gibi sessizce 32'ye kirpma yok.
        if max_short_len is None:
            max_short_len = max(short_len, getattr(hfp_config, 'MAX_SHORT_LEN', 32))
        assert max_short_len >= short_len, \
            f"max_short_len ({max_short_len}) < short_len ({short_len})"
        self.max_short_len = max_short_len

        # [K2] Chunk-ici recurrence blok boyutu (dogruluk degil hiz/bellek dengesi;
        # sonuc blok boyutundan BAGIMSIZDIR - bkz. smoke_test.py tutarlilik testi).
        self.rec_block = max(1, rec_block)

        self.landmark_max = hfp_config.LANDMARK_MAX
        self.gate_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.use_mixed_precision = use_mixed_precision  # [D3] no-op; geriye uyumluluk icin duruyor
        self.clip_value = clip_value
        self.dynamic_short_thresh = hfp_config.ENTROPY_THRESH

        # Selective Scan Gating (Information Bottleneck)
        self.importance_gate = nn.Linear(hidden_size, hidden_size)
        # [GATE COLLAPSE FIX] sigmoid(-2.0) ~ 0.12 baslangici
        nn.init.constant_(self.importance_gate.bias, -2.0)
        self.gate_dropout = nn.Dropout(0.1)

        # [FIX K8 - KISA CAUSAL CONV / BINDING] Lineer-attention BELLEGININ
        # associative-recall yapabilmesi icin sart olan token-karisimi. Onceki
        # surumde her token bellege KENDI key(x_t)⊗value(x_t)'sini yaziyordu;
        # v1'in anahtari onu ONCELEYEN k1'i kodlamadigindan sorgu=k1 ile v1
        # GETIRILEMIYORDU (MQAR loss ln(val_space)'te sabit, full-attention %100).
        # Depthwise causal conv (kernel=3) Q/K yoluna uygulanir -> K[v1-pozisyonu]
        # artik onceki token k1'i kodlar, Q[k1] ile eslesir. V ORIJINAL x'ten
        # (temiz deger). Mamba/H3/Based hepsi bu kisa conv'u icerir. Retention
        # yasasindan (exp/cubic_flux) BAGIMSIZ - kimlige dokunmaz. Chunk-tutarlilik
        # icin conv state chunk'lar arasi tasinir (T4 korunur).
        self.conv_kernel = max(1, conv_kernel)
        self.short_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=self.conv_kernel,
                                    groups=hidden_size, bias=True, padding=0)

        # Linear Attention Projections
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # [FIX K2b - COK-OLCEKLI DECAY] Eskiden tum kanallar sigmoid(2.19)~0.9
        # ile TEK olcekte baslardi -> bellek ufku ~1/(1-0.9)=10 token; 100 token
        # geriden recall matematiksel olarak imkansizdi (lam^100~2e-5). Simdi
        # kanal basina lam 0.90..0.999 arasi lineer dagilir (RetNet/GLA-tarzi
        # multi-timescale): bazi kanallar ~10 token, bazilari ~1000 token tutar.
        # Sigmoid ciktisi (0,1) oldugundan matrix-blowup korumasi korunur; tum
        # usler >= 0 stabilite degismez. decay hala LM loss'tan gradyan alir.
        # [HFP-CAP] decay/eta artik anahtar-kanali basina -> key_dim boyutunda.
        lam_min = getattr(hfp_config, 'DECAY_LAM_MIN', 0.90)
        lam_max = getattr(hfp_config, 'DECAY_LAM_MAX', 0.999)
        lam_init = torch.linspace(lam_min, lam_max, self.key_dim)
        decay_logit = torch.log(lam_init / (1.0 - lam_init))  # sigmoid^{-1}
        self.decay = nn.Parameter(decay_logit)

        # [HFP-CORE] Kubik-flux esnekligi eta (per-kanal, >0). Tek-adim kararli
        # cozumden lam_t = 1/sqrt(1 + 2*eta*s_t^2), s_t = anlik state buyuklugu.
        # Gecis olcegi t* ~ 1/sqrt(2*eta): eta buyuk -> kisa plato, kucuk -> uzun.
        # Kanallar arasi 1e-4..1e-2 log-dagilir -> plato ~7..70 token, ogrenilebilir.
        eta_init = torch.logspace(-4.0, -2.0, self.key_dim)
        self.log_eta = nn.Parameter(torch.log(eta_init))

        # [HFP-DELTA] per-token yazim siddeti beta (0,1); bias +1 -> ~0.73 baslangic
        self.beta_gate = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.beta_gate.bias, 1.0)

        self.retrieval_norm = nn.LayerNorm(hidden_size)
        self.landmark_buffer = LandmarkBuffer(max_size=hfp_config.LANDMARK_MAX)

    def _feat(self, u):
        """[HFP-CAP] Anahtar/sorgu ozellik-haritasi -> (..., key_dim), hep >= 0."""
        if self.key_feature_map == "dpfp":
            u = torch.cat([F.relu(u), F.relu(-u)], dim=-1)                   # (..., 2H)
            parts = [u * torch.roll(u, shifts=i + 1, dims=-1) for i in range(self.dpfp_nu)]
            return torch.cat(parts, dim=-1)                                  # (..., 2H*nu) >= 0
        return F.elu(u) + 1.0                                               # (..., H) > 0

    def get_initial_state(self, batch_size, device, dtype):
        M = torch.zeros(batch_size, self.key_dim, self.hidden_size, device=device, dtype=dtype)
        z = torch.zeros(batch_size, self.key_dim, device=device, dtype=dtype)
        short_memory = torch.zeros(batch_size, self.max_short_len, self.hidden_size, device=device, dtype=dtype)
        # [FIX K8] conv_state: onceki chunk'in son (kernel-1) girdisi (causal conv icin)
        conv_state = torch.zeros(batch_size, self.conv_kernel - 1, self.hidden_size, device=device, dtype=dtype)
        # state: (short_memory, M, z, token_count, short_len_dynamic, write_idx, conv_state)
        return (short_memory, M, z, 0, self.base_short_len, 0, conv_state)

    def reset_state(self):
        self.landmark_buffer.clear()
        if hasattr(self, "_last_gate"):
            del self._last_gate
        if hasattr(self, "_gate_entropy_live"):
            del self._gate_entropy_live

    def gate_entropy_loss(self):
        if not hasattr(self, "_last_gate"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if hfp_config.ENABLE_ENTROPY_MAP:
            return compute_gate_entropy(self._last_gate) * hfp_config.REG_WEIGHT
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def _write_ring_buffer(self, short_memory, x, write_idx):
        """[K6] Vektorize ring-buffer yazimi (eski per-token Python dongusu yerine).
        clone(): detach edilmemis state ile in-place autograd hatasini onler."""
        B, L, H = x.shape
        cap = self.max_short_len
        short_memory = short_memory.clone()
        if L >= cap:
            # yalnizca son 'cap' token buffer'da kalir
            tail = x[:, L - cap:, :]
            idx = (write_idx + (L - cap) + torch.arange(cap, device=x.device)) % cap
            short_memory[:, idx, :] = tail
        else:
            idx = (write_idx + torch.arange(L, device=x.device)) % cap
            short_memory[:, idx, :] = x
        new_write_idx = (write_idx + L) % cap
        return short_memory, new_write_idx

    def update(self, x, past_state=None, detach_state=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        if past_state is not None:
            (short_memory, M, z, token_count, short_len_dynamic, write_idx, conv_state) = past_state
            if short_memory is not None and short_memory.size(0) != batch_size:
                (short_memory, M, z, token_count, short_len_dynamic, write_idx, conv_state) = self.get_initial_state(batch_size, device, dtype)
        else:
            (short_memory, M, z, token_count, short_len_dynamic, write_idx, conv_state) = self.get_initial_state(batch_size, device, dtype)

        # [K2] detach_state artik cagiran tarafindan kontrol edilir (TBPTT icin False).
        if detach_state:
            if short_memory is not None: short_memory = short_memory.detach()
            if M is not None: M = M.detach()
            if z is not None: z = z.detach()
            if conv_state is not None: conv_state = conv_state.detach()

        # 1. Ring buffer (vektorize)
        short_memory, write_idx = self._write_ring_buffer(short_memory, x, write_idx)
        token_count += seq_len
        active_len = min(token_count, short_len_dynamic)

        # 2. [FIX K8] Binding conv: Q/K'yi conv'lanmis girdiden hesapla (komsu token
        #    karisimi -> anahtar onceki token'i kodlar), V'yi ORIJINAL x'ten (temiz deger).
        kk = self.conv_kernel
        if kk > 1:
            if conv_state is None:
                conv_state = torch.zeros(batch_size, kk - 1, self.hidden_size, device=device, dtype=dtype)
            x_pad = torch.cat([conv_state, x], dim=1)                       # (B, kk-1+L, H)
            x_qk = self.short_conv(x_pad.transpose(1, 2)).transpose(1, 2)   # (B, L, H) causal
            new_conv_state = x_pad[:, x_pad.size(1) - (kk - 1):, :]         # son kk-1 girdi
        else:
            x_qk = x
            new_conv_state = conv_state

        Q = self._feat(self.W_q(x_qk))   # (B,L,key_dim) >= 0  [HFP-CAP]
        K = self._feat(self.W_k(x_qk))   # (B,L,key_dim) >= 0
        V_raw = self.W_v(x)              # (B,L,H) temiz deger

        # 3. Gating (retrieval'dan ONCE: gate'li V hem intra-chunk okumaya
        #    hem M guncellemesine girer -> gate gradyan alir)
        gate_logits = self.importance_gate(x) / self.gate_temperature
        gate = torch.sigmoid(self.gate_dropout(gate_logits))
        gate = gate.to(dtype)
        self._last_gate = gate.clone().detach()
        # [C1] Gradyanli gate-entropy - modeling opsiyonel olarak loss'a ekler.
        self._gate_entropy_live = compute_gate_entropy(gate)

        gate_entropy = None
        if hfp_config.ENABLE_ENTROPY_MAP or hfp_config.ENABLE_DEFECT_FLAG or hfp_config.ENABLE_RYU_TAKAYANAGI:
            gate_entropy = compute_gate_entropy(gate)

        V = V_raw * gate

        # 4. Retention recurrence — mod secilir (exp baseline / cubic_flux HFP-core).
        outputs = []

        if self.write_rule == "delta" and self.decay_mode == "exp":
            # [HFP-DELTA-PAR] Delta-yazimin TAM chunkwise paralel formu (exp decay).
            # Turetim: u_t = k_t^T (Lam M_{t-1}) okumasiyla M_t = Lam M_{t-1} + k_t w_t^T,
            # w_t = beta_t (v_t - u_t). Acilim:
            #   u_t = (Lam^t k_t)^T M_0 + sum_{j<t} S_tj w_j ,  S_tj = k_t^T Lam^{t-j} k_j
            # => (I + diag(beta) S_strict) W = diag(beta) (V - A)  [unit-alt-ucgen COZUM]
            # Sonra cikti/durum, exp-chunkwise cebiriyle ayni (V yerine W, INKLUSIF maske).
            # Sirali delta ile birebir esdeger (bkz. verify: chunked==sequential).
            lam = torch.sigmoid(self.decay).to(dtype)                         # (D,)
            beta_all = torch.sigmoid(self.beta_gate(x)).to(dtype)             # (B,L,1)
            Kn_all = K / (K.norm(dim=-1, keepdim=True) + 1e-6)                # ||k||=1
            for s0 in range(0, seq_len, self.rec_block):
                Qb = Q[:, s0:s0 + self.rec_block]
                Kb = Kn_all[:, s0:s0 + self.rec_block]
                Vb = V[:, s0:s0 + self.rec_block]
                bb = beta_all[:, s0:s0 + self.rec_block]                      # (B,m,1)
                m = Qb.size(1)
                p = torch.arange(1, m + 1, device=device, dtype=dtype)
                lam_i = lam.unsqueeze(0).pow(p.unsqueeze(1))                  # (m,D): lam^t
                lam_rev = lam.unsqueeze(0).pow((m - p).unsqueeze(1))          # (m,D): lam^{m-t}

                # A_t = (Lam^t k_t)^T M_0  (B,m,H)
                K_dec = Kb * lam_i.unsqueeze(0)
                A = torch.bmm(K_dec, M)

                # S_tj = k_t^T Lam^{t-j} k_j (kesin alt ucgen) — exp-D tensoruyla ayni yapi
                ii = torch.arange(m, device=device).view(m, 1)
                jj = torch.arange(m, device=device).view(1, m)
                e = (ii - jj).clamp_min(0).to(dtype)
                strict = (ii > jj).to(dtype)
                Dm = lam.view(1, 1, -1).pow(e.unsqueeze(-1))                  # (m,m,D)
                S = torch.einsum('bih,ijh,bjh->bij', Kb, Dm, Kb) * strict     # (B,m,m)

                # (I + diag(beta) S) W = diag(beta)(V - A)  -> unit alt-ucgen cozum
                T = torch.eye(m, device=device, dtype=dtype).unsqueeze(0) + bb * S
                W = torch.linalg.solve_triangular(T, bb * (Vb - A), upper=False)  # (B,m,H)

                # ciktilar: out_t = (Lam^t q_t)^T M_0 + sum_{j<=t} (q_t^T Lam^{t-j} k_j) w_j
                Q_dec = Qb * lam_i.unsqueeze(0)
                out_cross = torch.bmm(Q_dec, M)                               # (B,m,H)
                inc = (ii >= jj).to(dtype)
                Sq = torch.einsum('bih,ijh,bjh->bij', Qb, Dm, Kb) * inc       # (B,m,m)
                outputs.append(out_cross + torch.bmm(Sq, W))

                # durum guncelle
                lam_m = lam.pow(float(m))
                K_rev = Kb * lam_rev.unsqueeze(0)
                M = M * lam_m.view(1, -1, 1) + torch.bmm(K_rev.transpose(1, 2), W)
                z = z * lam_m.view(1, -1) + K_rev.sum(dim=1)
            retrieved = torch.cat(outputs, dim=1)                             # (B,L,H)

        elif self.write_rule == "delta" and self.decay_mode != "cubic_flux_chunked":
            # [HFP-DELTA] Sirali delta-yazim; decay_mode lam'i belirler (exp/cubic).
            beta = torch.sigmoid(self.beta_gate(x)).to(dtype)                # (B,L,1)
            if self.decay_mode == "exp":
                lam_exp = torch.sigmoid(self.decay).to(dtype).unsqueeze(0)   # (1,D)
            else:
                eta = torch.exp(self.log_eta).to(dtype).unsqueeze(0)         # (1,D)
            for t in range(seq_len):
                kt = K[:, t]; vt = V[:, t]; qt = Q[:, t]                     # (B,D)/(B,H)
                kn = kt / (kt.norm(dim=-1, keepdim=True) + 1e-6)             # ||k||=1
                if self.decay_mode == "exp":
                    lam_t = lam_exp
                else:
                    lam_t = 1.0 / torch.sqrt(1.0 + 2.0 * eta * z * z)        # (B,D)
                Mt = M * lam_t.unsqueeze(-1)
                v_old = torch.einsum('bd,bdh->bh', kn, Mt)                   # mevcut iliski
                M = Mt + beta[:, t].unsqueeze(-1) * torch.einsum('bd,bh->bdh', kn, vt - v_old)
                z = z * lam_t + kn
                outputs.append(torch.einsum('bd,bdh->bh', qt, M).unsqueeze(1))
            retrieved = torch.cat(outputs, dim=1)                           # (B,L,H)

        elif self.decay_mode == "cubic_flux":
            # [HFP-CORE] Makalenin dth/dtau = -eta*th^3 kubik akisinin birebir
            # ayriklastirmasi. Tek-adim kararli cozum -> per-kanal decay faktoru:
            #     lam_t = 1/sqrt(1 + 2*eta*z_{t-1}^2)   (z = anahtar-akumulatoru, per-kanal)
            #     M_t = lam_t (.) M_{t-1} + k_t v_t^T ;  z_t = lam_t (.) z_{t-1} + k_t
            #     out_t = (q_t M_t)/(q_t . z_t)        (causal-inclusive, kendi KV dahil)
            # NOT: decay M'in degil Z'nin (anahtar kutlesi) buyuklugune baglidir.
            # z bos iken lam~1 (PLATO, unutma yok); z buyudukce lam<1 (aktif, buyukluge
            # bagli unutma) -> plato + power-law kuyruk. Kendini-sinirlayan: decay
            # buyuklukle arttigindan state patlamaz. Sirali (O(L)); mod default degil.
            # Saf recurrence oldugundan chunk-tutarli (full == state-tasiyan chunked).
            eta = torch.exp(self.log_eta).to(dtype).unsqueeze(0)             # (1,H) > 0
            for t in range(seq_len):
                kt = K[:, t]; vt = V[:, t]; qt = Q[:, t]                     # (B,H)
                lam_t = 1.0 / torch.sqrt(1.0 + 2.0 * eta * z * z)           # (B,H)
                M = M * lam_t.unsqueeze(-1) + torch.einsum('bh,bg->bhg', kt, vt)
                z = z * lam_t + kt
                num = torch.einsum('bh,bhg->bg', qt, M)                      # (B,H)
                den = (qt * z).sum(-1, keepdim=True) + 1e-6                  # (B,1)
                outputs.append((num / den).unsqueeze(1))                     # (B,1,H)
            retrieved = torch.cat(outputs, dim=1)                           # (B,L,H)

        elif self.decay_mode == "cubic_flux_chunked":
            # [HFP-SCALE] cubic_flux'in IKI-GECISLI TAM paralel formu (yaklasim DEGIL).
            # Gozlem: lam_t yalnizca z_{t-1}'e baglidir ve z'nin recurrence'i M'siz,
            # elementwise-ucuzdur. O halde:
            #   GECIS 1: z-taramasi (sirali ama per-adim O(B*D) elementwise) ->
            #            per-token lam_t TAM olarak bilinir.
            #   GECIS 2: lam_t bilindiginde M-recurrence, GLA/Mamba2-tarzi
            #            chunkwise-paralel cozulur (log-uzayda kumulatif carpim;
            #            tum katsayilar <= 1 -> stabil).
            # Sonuc her rec_block icin sirali cubic_flux ile birebir aynidir
            # (bkz. review_scripts/scaling_checks.py); rec_block yalnizca hiz/bellek
            # dengesidir. Bellek: intra-blok tensoru (B,m,m,key_dim).
            eta = torch.exp(self.log_eta).to(dtype).unsqueeze(0)                 # (1,D)
            
            if self.write_rule == "delta":
                beta_all = torch.sigmoid(self.beta_gate(x)).to(dtype)             # (B,L,1)
                Kn_all = K / (K.norm(dim=-1, keepdim=True) + 1e-6)                # ||k||=1
                K_scan = Kn_all
            else:
                K_scan = K
                
            # [HFP-SCALE HIZ] GECIS 1 (z-taramasi) TorchScript-derlenmis _cubic_zscan ile;
            # birebir ayni lam_seq, ama Python-loop overhead'i kalkar (uzun L'de ~2-4x).
            # Sirali dogasi korunur (dogrusal-olmayan recurrence -> paralel-scan yok).
            lam_seq = _cubic_zscan(K_scan, eta, z)                                    # (B,L,D)
            loglam = torch.log(lam_seq.clamp_min(1e-12))

            for s0 in range(0, seq_len, self.rec_block):                         # GECIS 2
                Qb = Q[:, s0:s0 + self.rec_block]
                Kb = K[:, s0:s0 + self.rec_block]
                Vb = V[:, s0:s0 + self.rec_block]
                m = Qb.size(1)
                cs = torch.cumsum(loglam[:, s0:s0 + m], dim=1)                   # (B,m,D): log A_i, A_i = prod_{j<=i} lam_j
                A = torch.exp(cs)                                                # (B,m,D) <= 1

                ii = torch.arange(m, device=device).view(m, 1)
                jj = torch.arange(m, device=device).view(1, m)
                causal = (ii >= jj).to(dtype)                                    # (m,m)
                Dm = torch.exp(cs.unsqueeze(2) - cs.unsqueeze(1))                # (B,m,m,D): exp(cs_i - cs_j)
                
                if self.write_rule == "delta":
                    Knb = Kn_all[:, s0:s0 + m]
                    bb = beta_all[:, s0:s0 + m]
                    
                    # cross-block A_t
                    K_dec = Knb * A
                    A_cross = torch.bmm(K_dec, M)
                    
                    # intra-block S_tj (strict)
                    strict = (ii > jj).to(dtype)
                    Dm_strict = Dm * strict.view(1, m, m, 1)
                    S = torch.einsum('bih,bijh,bjh->bij', Knb, Dm_strict, Knb)
                    
                    # unit alt-ucgen cozum
                    T = torch.eye(m, device=device, dtype=dtype).unsqueeze(0) + bb * S
                    W = torch.linalg.solve_triangular(T, bb * (Vb - A_cross), upper=False)
                    
                    # ciktilar
                    Q_dec = Qb * A
                    out_cross = torch.bmm(Q_dec, M)
                    Dm_causal = Dm * causal.view(1, m, m, 1)
                    Sq = torch.einsum('bih,bijh,bjh->bij', Qb, Dm_causal, Knb)
                    outputs.append(out_cross + torch.bmm(Sq, W))
                    
                    # durum guncelle (blok sonu)
                    A_m = A[:, -1]
                    K_rev = Knb * torch.exp(cs[:, -1:] - cs)
                    M = M * A_m.unsqueeze(-1) + torch.bmm(K_rev.transpose(1, 2), W)
                    z = z * A_m + K_rev.sum(dim=1)
                else:
                    # cross-block: M_0/z_0 katkisi A_i ile soner
                    Q_dec = Qb * A                                                   # (B,m,D)
                    num_cross = torch.bmm(Q_dec, M)                                  # (B,m,H)
                    den_cross = (Q_dec * z.unsqueeze(1)).sum(-1)                     # (B,m)
    
                    # intra-blok: pair (i,j<=i) katsayisi prod_{s=j+1..i} lam_s = exp(cs_i - cs_j) <= 1
                    Dm_causal = Dm * causal.view(1, m, m, 1)
    
                    S = torch.einsum('bih,bijh,bjh->bij', Qb, Dm_causal, Kb)         # (B,m,m)
                    num_intra = torch.bmm(S, Vb)                                     # (B,m,H)
                    den_intra = S.sum(dim=2)                                         # (B,m)
    
                    den = (den_cross + den_intra + 1e-6).unsqueeze(-1)
                    outputs.append((num_cross + num_intra) / den)
    
                    # state guncelle (blok sonu): A_m = tum blok carpimi
                    A_m = A[:, -1]                                                   # (B,D)
                    K_dec = Kb * torch.exp(cs[:, -1:] - cs)                          # (B,m,D): prod_{s=j+1..m}
                    M = M * A_m.unsqueeze(-1) + torch.bmm(K_dec.transpose(1, 2), Vb)
                    z = z * A_m + K_dec.sum(dim=1)
            retrieved = torch.cat(outputs, dim=1)                              # (B,L,H)

        else:
            # [K2] exp mod: paralel chunkwise (per-token geometrik decay, causal-inclusive)
            lam = torch.sigmoid(self.decay).to(dtype)  # (H,), 0..1
            for s in range(0, seq_len, self.rec_block):
                Qb = Q[:, s:s + self.rec_block]
                Kb = K[:, s:s + self.rec_block]
                Vb = V[:, s:s + self.rec_block]
                m = Qb.size(1)

                p = torch.arange(1, m + 1, device=device, dtype=dtype)          # 1..m
                lam_i = lam.unsqueeze(0).pow(p.unsqueeze(1))                     # (m,H): lam^i
                lam_rev = lam.unsqueeze(0).pow((m - p).unsqueeze(1))             # (m,H): lam^{m-i}

                # cross-block: eski state'ten oku
                Q_dec = Qb * lam_i.unsqueeze(0)                                  # (B,m,H)
                num_cross = torch.bmm(Q_dec, M)                                  # (B,m,H)
                den_cross = (Q_dec * z.unsqueeze(1)).sum(-1)                     # (B,m)

                # intra-block: D_ij = lam^{i-j} (i>=j), tum usler >= 0 -> stabil
                ii = torch.arange(m, device=device).view(m, 1)
                jj = torch.arange(m, device=device).view(1, m)
                e = (ii - jj).clamp_min(0).to(dtype)                             # (m,m)
                causal = (ii >= jj).to(dtype)
                D = lam.view(1, 1, -1).pow(e.unsqueeze(-1)) * causal.unsqueeze(-1)  # (m,m,H)

                S = torch.einsum('bih,ijh,bjh->bij', Qb, D, Kb)                  # (B,m,m)
                num_intra = torch.bmm(S, Vb)                                     # (B,m,H)
                den_intra = S.sum(dim=2)                                         # (B,m) > 0 (Q,K>0)

                den = (den_cross + den_intra + 1e-6).unsqueeze(-1)
                outputs.append((num_cross + num_intra) / den)

                # state guncelle (blok sonu)
                lam_m = lam.pow(float(m))
                K_dec = Kb * lam_rev.unsqueeze(0)
                M = M * lam_m.view(1, -1, 1) + torch.bmm(K_dec.transpose(1, 2), Vb)
                z = z * lam_m.view(1, -1) + K_dec.sum(dim=1)
            retrieved = torch.cat(outputs, dim=1)                              # (B,L,H)

        retrieved_memory = self.retrieval_norm(retrieved)                      # (B,L,H)

        # 5. Dynamic Context Windowing & Landmarks (opsiyonel teshis yollari)
        if gate_entropy is not None:
            if gate_entropy < self.dynamic_short_thresh and short_len_dynamic < self.max_short_len:
                short_len_dynamic = min(short_len_dynamic + 4, self.max_short_len)

        if hfp_config.ENABLE_DEFECT_FLAG:
            coherence = None
            if hfp_config.ENABLE_COHERENCE:
                coherence = coherence_score(short_memory)
            if gate_entropy is not None and coherence is not None:
                priority = coherence.item() * gate_entropy.item()
            else:
                priority = gate.mean().item()
            self.landmark_buffer.push(priority, x.mean(dim=1))

        new_past_state = (short_memory, M, z, token_count, short_len_dynamic, write_idx, new_conv_state)

        active_short_view = short_memory[:, :active_len, :]
        return active_short_view, retrieved_memory, new_past_state
