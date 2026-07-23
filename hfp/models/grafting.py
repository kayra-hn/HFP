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

"""[FAZ 3 — GRAFTING] HFP bellegini pretrained bir Llama-ailesi LLM'e asilama.

LLM_GRAFTING_STRATEGY.md'nin implementasyonu. Acik teknik kararlarin cozumu:

1. GQA -> HFP esleme (strateji dok. "Dimension Uyumsuzlugu"):
   Bellek TAM-hidden yerine KAFA BASINA tutulur. Llama-3.2-1B: hidden 2048,
   32 Q kafasi, 8 KV kafasi, head_dim 64. K/V, GQA grubu basina
   repeat_interleave(n_rep) ile 32 kafaya genisletilir (softmax attention'in
   yaptigi esleme ile birebir ayni). DPFP kafa basina uygulanir:
   key_dim = 2*head_dim*nu (nu=2 -> 256). Katman state'i:
   M (32, 256, 64) + z (32, 256) ~ 0.5M float -> gercek O(1), VRAM sabit.

2. RoPE catismasi (strateji dok. "Kritik Karar"):
   HFP yolu RoPE'siz (NoPE) calisir — oncelikli test secenegi. Binding conv
   (kernel 3, depthwise) yerel sirayi zaten kodlar; uzun-menzil konum bilgisi
   recurrent decay'in kendisinden gelir. Base modelde kalan attention
   katmanlari RoPE'lerini korur (hibrit).

3. Warm-start: Llama'nin q/k/v/o_proj modulleri AYNEN paylasilir (kopya degil,
   referans; frozen). Egitilen SADECE HFP parametreleri: decay, log_eta,
   binding conv, retrieval norm, beta/alpha gate, out_gain.

4. Distilasyon (LoLCATs/T2R tarzi, OGRETMEN-KOPYASIZ):
   mode="teacher_forcing" -> katman icinde hem orijinal softmax attention
   (hedef, no-grad) hem HFP (ogrenci) ayni girdiyle kosulur; MSE modul
   attribute'unda biriktirilir; ileriye OGRETMEN ciktisi gecirilir (akis
   distribution'da kalir, katmanlar bagimsiz ogrenir). Ikinci model yok ->
   1B model tek T4'e sigar.
   mode="student" -> HFP ciktisi ileri gider (Stage 2: logit-KL / LM loss,
   ve inference).

5. Yazim kurali — ALPHA-GATE MELEZI (additive<->delta surekli interpolasyon):
       u_t   = k^_t^T (Lam M_{t-1})            (mevcut iliski okumasi)
       w_t   = beta_t (v_t - alpha * u_t)
       M_t   = Lam M_{t-1} + k^_t w_t^T
   alpha = sigmoid(per-kafa parametre): 0 -> saf additive (arsiv),
   1 -> saf delta (guncelleyen calisma-bellegi). Init -2 (~0.12, additive
   agirlikli; WikiText-2 ablasyonu additive'i destekliyor) — model kafa
   basina kendi karar verir. write_rule="additive" alpha=0/beta=1 sabitler,
   "delta" alpha=1 sabitler, "hybrid" alpha'yi ogrenir.
   Cozum chunkwise-paraleldir: (I + diag(alpha*beta) S_strict) W =
   diag(beta)(V - alpha*A_cross), unit-alt-ucgen triangular solve
   (hfp_bulk_state.py delta-paralel turetiminin alpha'li genellemesi).

6. Retention: decay_mode="exp" (cok-olcekli lam, paralel) veya
   "cubic_flux_chunked" (iki-gecisli: sirali z-taramasi -> paralel M;
   hfp_bulk_state.py ile ayni cebir, kafa-katlanmis batch'te).

Kullanim:
    from transformers import AutoModelForCausalLM
    from hfp.models.grafting import GraftConfig, graft_llama, set_graft_mode, \
        distill_loss, trainable_parameters, enable_streaming, reset_streaming

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",
                                                 torch_dtype=torch.float32)
    cfg = GraftConfig(decay_mode="cubic_flux_chunked", write_rule="hybrid")
    grafted = graft_llama(model, cfg)          # default: katmanlarin yarisi
    set_graft_mode(model, "teacher_forcing")   # Stage 1 distilasyon
    out = model(input_ids); aux = distill_loss(model)   # aux.backward()
    set_graft_mode(model, "student")           # Stage 2 / inference

Not: transformers >= 4.46 hedeflenir (attention forward'i (out, weights)
dondurur). Eski surumlerde GraftConfig.ret_len=3 ayarlayin.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraftConfig:
    decay_mode: str = "cubic_flux_chunked"   # "exp" | "cubic_flux_chunked"
    write_rule: str = "hybrid"               # "additive" | "delta" | "hybrid"
    key_feature_map: str = "dpfp"            # "dpfp" | "elu"
    dpfp_nu: int = 2
    conv_kernel: int = 3
    rec_block: int = 64
    lam_min: float = 0.90                    # exp cok-olcekli decay init
    lam_max: float = 0.999
    eta_log_min: float = -4.0                # cubic eta logspace init
    eta_log_max: float = -2.0
    alpha_init: float = -2.0                 # hybrid: sigmoid(-2)~0.12 additive-agirlikli
    ret_len: int = 2                         # attention forward tuple uzunlugu (HF surumu)


class HFPGraftAttention(nn.Module):
    """Llama self_attn'in yerine gecen HFP bellek modulu (hibrit katmanlarda).

    Orijinal attention modulu `teacher` olarak icerde tutulur (frozen):
    - teacher_forcing modunda hedef uretir ve ileri gecer (Stage 1),
    - student modunda hic kosulmaz (inference'ta O(1)).
    """

    def __init__(self, teacher_attn: nn.Module, layer_idx: int, cfg: GraftConfig,
                 hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.head_dim = head_dim

        # --- Ogretmen (frozen softmax attention; warm-start projeksiyon kaynagi) ---
        self.teacher = teacher_attn
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Paylasilan (frozen) projeksiyonlar — warm-start: kopya degil referans
        self.q_proj = teacher_attn.q_proj
        self.k_proj = teacher_attn.k_proj
        self.v_proj = teacher_attn.v_proj
        self.o_proj = teacher_attn.o_proj

        # --- HFP parametreleri (egitilen kisim) ---
        D = head_dim
        self.key_dim = 2 * D * cfg.dpfp_nu if cfg.key_feature_map == "dpfp" else D

        # Binding conv (kernel 3, depthwise): associative recall icin sart (K8).
        # Projeksiyon SONRASI q/k yoluna uygulanir (girdi yoluna uygulanamaz;
        # projeksiyonlar frozen). Kanal = num_heads*head_dim (k, rep sonrasi).
        kk = max(1, cfg.conv_kernel)
        self.conv_kernel = kk
        ch = num_heads * head_dim
        self.conv_q = nn.Conv1d(ch, ch, kk, groups=ch, bias=True, padding=0)
        self.conv_k = nn.Conv1d(ch, ch, kk, groups=ch, bias=True, padding=0)
        # Init: kimlige yakin (son tap 1, digerleri 0) -> zero-shot bozulmaz
        with torch.no_grad():
            for conv in (self.conv_q, self.conv_k):
                conv.weight.zero_()
                conv.weight[:, 0, -1] = 1.0
                conv.bias.zero_()

        # Cok-olcekli exp decay (kafa x key_dim) — K2b init
        lam = torch.linspace(cfg.lam_min, cfg.lam_max, self.key_dim)
        lam = lam.unsqueeze(0).expand(num_heads, -1).contiguous()
        self.decay = nn.Parameter(torch.log(lam / (1 - lam)))
        # Cubic eta (kafa x key_dim), logspace
        eta = torch.logspace(cfg.eta_log_min, cfg.eta_log_max, self.key_dim)
        self.log_eta = nn.Parameter(torch.log(eta).unsqueeze(0)
                                    .expand(num_heads, -1).contiguous())
        # Yazim kapilari
        self.beta_gate = nn.Linear(head_dim, 1)
        nn.init.constant_(self.beta_gate.bias, 1.0)          # sigmoid(1)~0.73
        self.alpha_logit = nn.Parameter(torch.full((num_heads,), cfg.alpha_init))
        # Okuma normu + cikis kazanci (zero-shot guvenligi: kucuk baslar)
        self.retrieval_norm = nn.LayerNorm(head_dim)
        self.out_gain = nn.Parameter(torch.full((num_heads, 1, 1), 1.0))

        # Distilasyon durumu
        self.mode = "student"                # "student" | "teacher_forcing" | "teacher"
        self.last_distill_loss: Optional[torch.Tensor] = None
        # [VRAM] Stage 1 katman-aninda backward: teacher_forcing'de her katmanin
        # ogrenci grafigi BAGIMSIZDIR; float verilirse (orn. 1/ACCUM) MSE katman
        # icinde hemen backward'lanir ve grafik serbest kalir -> tepe bellek
        # 13 katman yerine 1 katmanlik graf. last_distill_loss detached kalir
        # (yalnizca loglama icin); disaridan aux.backward() CAGRILMAZ.
        self.distill_backward_scale: Optional[float] = None
        # Streaming durumu (needle/uzun-akis eval): (M, z, cq_state, ck_state)
        self.streaming = False
        self._stream_state = None

    # ---------- yardimcilar ----------

    def _feat(self, u):
        """DPFP / elu+1 ozellik-haritasi, kafa-katlanmis girdi (..., head_dim)."""
        if self.cfg.key_feature_map == "dpfp":
            u = torch.cat([F.relu(u), F.relu(-u)], dim=-1)
            parts = [u * torch.roll(u, shifts=i + 1, dims=-1)
                     for i in range(self.cfg.dpfp_nu)]
            return torch.cat(parts, dim=-1)
        return F.elu(u) + 1.0

    def _conv(self, x, conv, state):
        """Causal depthwise conv, streaming state ile. x: (B, L, ch)."""
        kk = self.conv_kernel
        if kk == 1:
            return x, state
        B, L, ch = x.shape
        if state is None:
            state = x.new_zeros(B, kk - 1, ch)
        xp = torch.cat([state, x], dim=1)
        y = conv(xp.transpose(1, 2)).transpose(1, 2)
        return y, xp[:, xp.size(1) - (kk - 1):, :].detach()

    def _alpha(self, dtype):
        if self.cfg.write_rule == "additive":
            return None                       # alpha=0 yolu (payda-normalize)
        if self.cfg.write_rule == "delta":
            return torch.ones(self.num_heads, dtype=dtype,
                              device=self.alpha_logit.device)
        return torch.sigmoid(self.alpha_logit).to(dtype)

    # ---------- cekirdek recurrence (kafa-katlanmis batch: BH = B*num_heads) ----------

    def _memory(self, q, k, v, beta, M, z):
        """q,k: (BH,L,key_dim)  v: (BH,L,D)  beta: (BH,L,1)  M: (BH,key_dim,D)  z: (BH,key_dim)
        Cikti: (BH,L,D), yeni M, z. Chunkwise; hfp_bulk_state.py cebiriyle ayni."""
        cfg = self.cfg
        BH, L, Dk = q.shape
        dt, dev = q.dtype, q.device
        alpha = self._alpha(dt)               # None (additive) | (num_heads,)
        if alpha is not None:
            # kafa-katlanmis batch'e yay: (BH,1,1)
            alpha = alpha.repeat(BH // self.num_heads).view(BH, 1, 1)
            kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        outs = []

        # per-token log-decay dizisi
        if cfg.decay_mode == "exp":
            lam = torch.sigmoid(self.decay).to(dt)                    # (H,key_dim)
            lam = lam.repeat(BH // self.num_heads, 1)                 # (BH,key_dim)
            loglam = torch.log(lam.clamp_min(1e-12)).unsqueeze(1).expand(BH, L, Dk)
        else:  # cubic_flux_chunked: GECIS 1 — sirali z-taramasi (elementwise, ucuz)
            eta = torch.exp(self.log_eta).to(dt).repeat(BH // self.num_heads, 1)
            zs = z
            lam_list = []
            K_scan = kn if alpha is not None else k
            for t in range(L):
                lam_t = 1.0 / torch.sqrt(1.0 + 2.0 * eta * zs * zs)
                lam_list.append(lam_t)
                zs = zs * lam_t + K_scan[:, t]
            loglam = torch.log(torch.stack(lam_list, dim=1).clamp_min(1e-12))

        for s0 in range(0, L, cfg.rec_block):                          # GECIS 2 — paralel
            qb = q[:, s0:s0 + cfg.rec_block]
            vb = v[:, s0:s0 + cfg.rec_block]
            m = qb.size(1)
            cs = torch.cumsum(loglam[:, s0:s0 + m], dim=1)             # (BH,m,Dk)
            A = torch.exp(cs)
            ii = torch.arange(m, device=dev).view(m, 1)
            jj = torch.arange(m, device=dev).view(1, m)
            causal = (ii >= jj).to(dt)
            Dm = torch.exp(cs.unsqueeze(2) - cs.unsqueeze(1))          # (BH,m,m,Dk)

            if alpha is None:
                # --- additive: payda-normalize okuma (orijinal HFP semantigi) ---
                kb = k[:, s0:s0 + m]
                Q_dec = qb * A
                num_cross = torch.bmm(Q_dec, M)
                den_cross = (Q_dec * z.unsqueeze(1)).sum(-1)
                Dm_c = Dm * causal.view(1, m, m, 1)
                S = torch.einsum('bih,bijh,bjh->bij', qb, Dm_c, kb)
                num = num_cross + torch.bmm(S, vb)
                den = (den_cross + S.sum(dim=2) + 1e-6).unsqueeze(-1)
                outs.append(num / den)
                A_m = A[:, -1]
                K_rev = kb * torch.exp(cs[:, -1:] - cs)
                M = M * A_m.unsqueeze(-1) + torch.bmm(K_rev.transpose(1, 2), vb)
                z = z * A_m + K_rev.sum(dim=1)
            else:
                # --- alpha-gate melez delta (chunkwise-paralel triangular solve) ---
                knb = kn[:, s0:s0 + m]
                bb = beta[:, s0:s0 + m]                                # (BH,m,1)
                K_dec = knb * A
                A_cross = torch.bmm(K_dec, M)                          # u_t cross kismi
                strict = (ii > jj).to(dt)
                Dm_s = Dm * strict.view(1, m, m, 1)
                S = torch.einsum('bih,bijh,bjh->bij', knb, Dm_s, knb)
                T = torch.eye(m, device=dev, dtype=dt).unsqueeze(0) + alpha * bb * S
                W = torch.linalg.solve_triangular(
                    T, bb * (vb - alpha * A_cross), upper=False)       # (BH,m,D)
                Q_dec = qb * A
                out_cross = torch.bmm(Q_dec, M)
                Dm_c = Dm * causal.view(1, m, m, 1)
                Sq = torch.einsum('bih,bijh,bjh->bij', qb, Dm_c, knb)
                outs.append(out_cross + torch.bmm(Sq, W))
                A_m = A[:, -1]
                K_rev = knb * torch.exp(cs[:, -1:] - cs)
                M = M * A_m.unsqueeze(-1) + torch.bmm(K_rev.transpose(1, 2), W)
                z = z * A_m + K_rev.sum(dim=1)

        return torch.cat(outs, dim=1), M, z

    # ---------- HFP ogrenci yolu ----------

    def _student_forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states)                                  # (B,L,H*D)
        k = self.k_proj(hidden_states)                                  # (B,L,KV*D)
        v = self.v_proj(hidden_states)
        # GQA -> tum kafalara genislet (softmax attention'in eslemesiyle ayni)
        k = k.view(B, L, self.num_kv_heads, D).repeat_interleave(self.n_rep, dim=2)
        v = v.view(B, L, self.num_kv_heads, D).repeat_interleave(self.n_rep, dim=2)
        k = k.reshape(B, L, H * D)
        v = v.reshape(B, L, H * D)
        # NOT: RoPE uygulanmaz (bypass karari; bkz. modul docstring §2)

        st = self._stream_state if (self.streaming and self._stream_state is not None) else None
        cq = st[2] if st else None
        ck = st[3] if st else None
        q, cq = self._conv(q, self.conv_q, cq)                          # binding conv
        k, ck = self._conv(k, self.conv_k, ck)

        # kafa-katlanmis batch: (B*H, L, D)
        def fold(x):
            return x.view(B, L, H, D).permute(0, 2, 1, 3).reshape(B * H, L, D)
        qh, kh, vh = fold(q), fold(k), fold(v)
        qf, kf = self._feat(qh), self._feat(kh)                         # (BH,L,key_dim)
        beta = torch.sigmoid(self.beta_gate(vh))                        # (BH,L,1)

        if st is not None:
            M, z = st[0], st[1]
        else:
            M = qf.new_zeros(B * H, self.key_dim, D)
            z = qf.new_zeros(B * H, self.key_dim)

        out, M, z = self._memory(qf, kf, vh, beta, M, z)                # (BH,L,D)
        if self.streaming:
            self._stream_state = (M.detach(), z.detach(), cq, ck)

        out = self.retrieval_norm(out.view(B, H, L, D)) * self.out_gain # kafa-basi kazanc
        out = out.permute(0, 2, 1, 3).reshape(B, L, H * D)
        return self.o_proj(out)

    # ---------- HF-uyumlu forward ----------

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        need_teacher = self.mode in ("teacher_forcing", "teacher", "student_forcing")
        t_out = None
        if need_teacher:
            with torch.no_grad():
                t_ret = self.teacher(
                    hidden_states, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_value=past_key_value,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs)
                t_out = t_ret[0] if isinstance(t_ret, tuple) else t_ret

        if self.mode == "teacher":
            out = t_out
        else:
            s_out = self._student_forward(hidden_states)
            if self.mode in ("teacher_forcing", "student_forcing"):
                # Stage 1 distilasyon: katman-ici MSE hedefi = teacher(bu-katmanin-girdisi).
                #   teacher_forcing: ileri OGRETMEN ciktisi gider -> katmanlar bagimsiz, hepsi
                #     TEMIZ ogretmen girdisi gorur (ama cikarimda BOZUK girdi gelir = exposure bias).
                #   student_forcing [§25]: ileri OGRENCI ciktisi (detach) gider -> her katman
                #     cikarimda gorecegi GERCEK (student-uretimi, bozuk) girdi dagilimini gorur
                #     ve ustundeki hatayi sogurmayi ogrenir = compounding'in KOK cozumu.
                #     detach: katman-arasi graf kesilir -> ucuz per-katman backward korunur.
                l = F.mse_loss(s_out, t_out)
                if (self.distill_backward_scale is not None and self.training
                        and torch.is_grad_enabled()):
                    (l * self.distill_backward_scale).backward()   # grafigi hemen bosalt
                    self.last_distill_loss = l.detach()
                else:
                    self.last_distill_loss = l
                out = t_out if self.mode == "teacher_forcing" else s_out.detach()
            else:
                out = s_out

        if self.cfg.ret_len == 3:
            return out, None, past_key_value
        return out, None


# ==================== model-duzeyi yardimcilar ====================

def default_graft_layers(num_layers: int) -> List[int]:
    """Hibrit varsayilani: katmanlarin YARISI HFP (tek indeksliler), ilk ve son
    katman full-attention kalir (sentaks + cikti kalitesi; Mamba-in-Llama bulgusu)."""
    return [i for i in range(1, num_layers - 1) if i % 2 == 1]


def graft_llama(model, cfg: Optional[GraftConfig] = None,
                layers: Optional[List[int]] = None):
    """model.model.layers[i].self_attn -> HFPGraftAttention (secili katmanlarda).
    Tum base parametreleri dondurur; HFP parametreleri egitilebilir kalir."""
    cfg = cfg or GraftConfig()
    mcfg = model.config
    num_layers = mcfg.num_hidden_layers
    layers = default_graft_layers(num_layers) if layers is None else layers
    head_dim = getattr(mcfg, "head_dim", None) or mcfg.hidden_size // mcfg.num_attention_heads

    for p in model.parameters():                     # once hepsini dondur
        p.requires_grad_(False)

    grafted = []
    for i in layers:
        blk = model.model.layers[i]
        hfp_attn = HFPGraftAttention(
            blk.self_attn, i, cfg,
            hidden_size=mcfg.hidden_size,
            num_heads=mcfg.num_attention_heads,
            num_kv_heads=getattr(mcfg, "num_key_value_heads", mcfg.num_attention_heads),
            head_dim=head_dim)
        hfp_attn.to(next(blk.parameters()).device, next(blk.parameters()).dtype)
        blk.self_attn = hfp_attn
        grafted.append(i)

    # HFP parametrelerini ac (paylasilan frozen projeksiyonlar haric)
    frozen_shared = {"q_proj", "k_proj", "v_proj", "o_proj", "teacher"}
    for m in model.modules():
        if isinstance(m, HFPGraftAttention):
            for name, p in m.named_parameters():
                if not any(name.startswith(f) for f in frozen_shared):
                    p.requires_grad_(True)
    print(f"[graft] {len(grafted)}/{num_layers} katman HFP'ye cevrildi: {grafted}")
    print(f"[graft] egitilebilir param: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return grafted


def set_distill_backward(model, scale: Optional[float]):
    """Stage 1 VRAM modu: scale=1/ACCUM -> katman-aninda backward. None -> kapat."""
    for m in model.modules():
        if isinstance(m, HFPGraftAttention):
            m.distill_backward_scale = scale


def set_graft_mode(model, mode: str):
    assert mode in ("student", "teacher_forcing", "teacher", "student_forcing")
    for m in model.modules():
        if isinstance(m, HFPGraftAttention):
            m.mode = mode


def distill_loss(model):
    """teacher_forcing forward'indan sonra katman-ici MSE'lerin toplami."""
    losses = [m.last_distill_loss for m in model.modules()
              if isinstance(m, HFPGraftAttention) and m.last_distill_loss is not None]
    return torch.stack(losses).mean() if losses else None


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def enable_streaming(model, flag: bool = True):
    """O(1) akis modu (needle testi): grafted katman state'leri cagrilar arasi tasinir.
    Full-attention katmanlar icin HF past_key_values ayrica yonetilmelidir."""
    for m in model.modules():
        if isinstance(m, HFPGraftAttention):
            m.streaming = flag
            if not flag:
                m._stream_state = None


def reset_streaming(model):
    for m in model.modules():
        if isinstance(m, HFPGraftAttention):
            m._stream_state = None


# (dosya sonu)
