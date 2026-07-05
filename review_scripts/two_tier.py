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

"""[2TIER] Iki-kademeli bellek prototipi + uzun-ufuk testi.

Mimari (cekirdege dokunmadan, subclass):
  - M_fast/z_fast : mevcut mekanizma (exp decay + additive yazim), per-token.
  - M_slow/z_slow : her C token'da bir HIZLI BELLEKTEN konsolidasyon alir
                    (token'lardan degil): M_slow = lam_s (.) M_slow + g*M_fast.
                    Yavas yasa "cubic" (kucuk eta -> seyrek yazimda plato) veya "exp".
  - Okuma: out = qM_f/(q.z_f) + sigmoid(mix) (.) qM_s/(q.z_s + 1)

Kullanim:
  python review_scripts/two_tier.py verify
  python review_scripts/two_tier.py run <baseline|twotier_cubic|twotier_exp> <lr> <seed> [budget]
Gorev: dense format, ctx 320, 8 cift/8 sorgu; kovalar <48 / 48-127 / 128-223 / 224+.
"""
import os
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

import os, random, sys, time
import numpy as np, torch
import torch.nn as nn
from hfp.core.hfp_bulk_state import HFPBulkState
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM


class TwoTierState(HFPBulkState):
    def __init__(self, *args, consolidate_every=16, slow_law="cubic", **kw):
        super().__init__(*args, **kw)
        self.consolidate_every = consolidate_every
        self.slow_law = slow_law
        # yavas kademe parametreleri
        eta_s = torch.logspace(-6.0, -4.0, self.key_dim)      # cok uzun plato
        self.log_eta_slow = nn.Parameter(torch.log(eta_s))
        self.decay_slow = nn.Parameter(torch.full((self.key_dim,), 7.6))  # sigmoid~0.9995
        self.gamma = nn.Parameter(torch.tensor(-2.0))          # konsolidasyon orani ~0.12
        self.mix = nn.Parameter(torch.full((self.hidden_size,), -1.0))    # yavas katki ~0.27

    def get_initial_state(self, batch_size, device, dtype):
        base = super().get_initial_state(batch_size, device, dtype)
        M_s = torch.zeros(batch_size, self.key_dim, self.hidden_size, device=device, dtype=dtype)
        z_s = torch.zeros(batch_size, self.key_dim, device=device, dtype=dtype)
        return base + (M_s, z_s)

    def update(self, x, past_state=None, detach_state=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, L, _ = x.size()
        device, dtype = x.device, x.dtype
        if past_state is None or past_state[0].size(0) != B:
            past_state = self.get_initial_state(B, device, dtype)
        (short_memory, M, z, token_count, short_len_dyn, write_idx, conv_state, M_s, z_s) = past_state
        if detach_state:
            short_memory = short_memory.detach(); M = M.detach(); z = z.detach()
            conv_state = conv_state.detach(); M_s = M_s.detach(); z_s = z_s.detach()

        short_memory, write_idx = self._write_ring_buffer(short_memory, x, write_idx)
        n0 = token_count
        token_count += L
        active_len = min(token_count, short_len_dyn)

        kk = self.conv_kernel
        if kk > 1:
            x_pad = torch.cat([conv_state, x], dim=1)
            x_qk = self.short_conv(x_pad.transpose(1, 2)).transpose(1, 2)
            new_conv_state = x_pad[:, x_pad.size(1) - (kk - 1):, :]
        else:
            x_qk = x; new_conv_state = conv_state
        Q = self._feat(self.W_q(x_qk)); K = self._feat(self.W_k(x_qk))
        gate = torch.sigmoid(self.gate_dropout(self.importance_gate(x))).to(dtype)
        self._last_gate = gate.clone().detach()
        V = self.W_v(x) * gate

        lam_f = torch.sigmoid(self.decay).to(dtype).unsqueeze(0)              # (1,D)
        g = torch.sigmoid(self.gamma)
        mixv = torch.sigmoid(self.mix)
        if self.slow_law == "exp":
            lam_s_const = torch.sigmoid(self.decay_slow).to(dtype).unsqueeze(0)
        else:
            eta_s = torch.exp(self.log_eta_slow).to(dtype).unsqueeze(0)

        outs = []
        for t in range(L):
            kt, vt, qt = K[:, t], V[:, t], Q[:, t]
            M = M * lam_f.unsqueeze(-1) + torch.einsum('bd,bh->bdh', kt, vt)
            z = z * lam_f + kt
            # konsolidasyon: global token indeksi C'nin kati oldugunda
            if (n0 + t + 1) % self.consolidate_every == 0:
                if self.slow_law == "exp":
                    lam_s = lam_s_const
                else:
                    lam_s = 1.0 / torch.sqrt(1.0 + 2.0 * eta_s * z_s * z_s)   # (B,D)
                M_s = M_s * lam_s.unsqueeze(-1) + g * M
                z_s = z_s * lam_s + g * z
            of = torch.einsum('bd,bdh->bh', qt, M) / ((qt * z).sum(-1, keepdim=True) + 1e-6)
            os_ = torch.einsum('bd,bdh->bh', qt, M_s) / ((qt * z_s).sum(-1, keepdim=True) + 1.0)
            outs.append((of + mixv * os_).unsqueeze(1))
        retrieved = self.retrieval_norm(torch.cat(outs, dim=1))
        new_state = (short_memory, M, z, token_count, short_len_dyn, write_idx, new_conv_state, M_s, z_s)
        return short_memory[:, :active_len, :], retrieved, new_state


# ---------------- gorev (dense, ctx 320) ----------------
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, WIN, P = 320, 8, 16   # P=16 -> etiket yogunlugu ctx160/P8 ile ayni
ANS = 0

def make_seq():
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    slots = random.sample(range(CTX // 2), 2 * P); random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * CTX; meta = []
    for i in range(P):
        a, b = slots[2 * i], slots[2 * i + 1]
        if a > b: a, b = b, a
        wp, qp = 2 * a, 2 * b
        v = random.randint(VLO, VHI - 1)
        toks[wp] = keys[i]; toks[wp + 1] = v
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        lab[qp + 1] = v
        meta.append((qp - wp, qp + 1, v))
    return toks, lab, meta


def build(kind, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                    short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                    decay_mode="exp", rec_block=32)
    model = HFPForCausalLM(cfg)
    if kind.startswith("twotier"):
        law = "cubic" if kind.endswith("cubic") else "exp"
        for i in range(cfg.num_hidden_layers):
            model.hfp.bulk_states[i] = TwoTierState(
                hidden_size=64, short_len=8, rec_block=32, decay_mode="exp",
                slow_law=law, consolidate_every=16)
    return model


def verify():
    torch.manual_seed(0)
    m = TwoTierState(hidden_size=16, short_len=4, max_short_len=8, rec_block=4, slow_law="cubic")
    m.eval()
    with torch.no_grad():
        x = torch.randn(2, 40, 16)
        _, rf, sf = m.update(x)
        _, r1, s1 = m.update(x[:, :17]); _, r2, s2 = m.update(x[:, 17:], past_state=s1)
        d = (rf - torch.cat([r1, r2], 1)).abs().max().item()
        dMs = (sf[7] - s2[7]).abs().max().item()
    print(f"chunk-tutarlilik: out {d:.2e}  M_slow {dMs:.2e} ->", "PASS" if d < 1e-5 else "FAIL")
    model = build("twotier_cubic", 0); model.train()
    out = model(torch.randint(0, VHI + 4, (2, 48)), labels=torch.randint(0, VHI + 4, (2, 48)))
    out.loss.backward()
    bs = model.hfp.bulk_states[0]
    for nm in ("gamma", "mix", "log_eta_slow"):
        gnorm = getattr(bs, nm).grad
        print(f"grad({nm}) =", "YOK" if gnorm is None else f"{gnorm.norm().item():.2e}")
    model.eval()
    with torch.no_grad():
        x1 = torch.randint(0, VHI + 4, (1, 40)); x2 = x1.clone(); x2[0, -1] = (x1[0, -1] + 5) % (VHI + 4)
        d = (model(x1).logits[:, :-1] - model(x2).logits[:, :-1]).abs().max().item()
    print(f"causal sizinti: {d:.2e} ->", "PASS" if d == 0.0 else "FAIL")


def run():
    KIND, LR, SEED = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 33.0
    TAG = f"2t_{KIND}_p{P}_{LR:g}_{SEED}"
    CKPT = f"{CKDIR}/{TAG}.pt"
    STEPS = 600
    model = build(KIND, SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    start = 0
    if os.path.exists(CKPT):
        st = torch.load(CKPT, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
        start = st["step"]
        random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])

    def batch(bs=12):
        xs, ys = [], []
        for _ in range(bs):
            t, l, _ = make_seq(); xs.append(t); ys.append(l)
        return torch.tensor(xs), torch.tensor(ys)

    t0 = time.time(); model.train(); step = start
    while step < STEPS and time.time() - t0 < BUDGET:
        step += 1
        x, y = batch()
        out = model(x, labels=y)
        assert torch.isfinite(out.loss), f"NaN {TAG} step {step}"
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            print(f"[{TAG}] step {step} loss {out.loss.item():.3f}", flush=True)
    if step < STEPS:
        torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                        rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
        print(f"[{TAG}] CKPT {step}/{STEPS}"); sys.exit(0)

    model.eval()
    buckets = {"<48": [0, 0], "48-127": [0, 0], "128-223": [0, 0], "224+": [0, 0]}
    def bname(gp):
        return "<48" if gp < 48 else "48-127" if gp < 128 else "128-223" if gp < 224 else "224+"
    with torch.no_grad():
        for _ in range(60):
            toks, _, meta = make_seq()
            logits = model(torch.tensor([toks])).logits[0]
            for gp, ap, v in meta:
                b = buckets[bname(gp)]; b[1] += 1
                b[0] += int(logits[ap - 1].argmax().item() == v)
    res = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in buckets.items()}
    print(f"[{TAG}] FINAL {res} (n={ {k: n for k, (c, n) in buckets.items()} })")
    with open(f"{CKDIR}/2t_results.txt", "a") as f:
        f.write(f"{TAG} {res}\n")


if __name__ == "__main__":
    (verify if sys.argv[1] == "verify" else run)()
