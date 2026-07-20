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

"""[LEN-GEN] Uzunluk genellemesi: ctx 160'ta egit, 160/320/640'ta degerlendir.
O(1) state'in vaadi: egitim uzunlugu != cikarim uzunlugu. Recall transfer oluyorsa
ctx-320 'ogrenilebilirlik ucurumu' yalnizca egitim fenomenidir ve cozumu bedavadir.
Kullanim: python review_scripts/length_gen.py <train|eval> <seed> [budget]"""
import os
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

import random, sys, time
import numpy as np, torch
import os as _os
VARIANT = _os.environ.get("LG_VARIANT", "additive")  # additive|delta|dpfp
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
WIN, P = 8, 8
ANS = 0
TRAIN_CTX, STEPS, LR = 160, 600, 1e-3
SEED = int(sys.argv[2])
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 33.0
CKPT = f"{CKDIR}/lg_{VARIANT}_{SEED}.pt" if VARIANT != "additive" else f"{CKDIR}/lg_{SEED}.pt"
FINAL = f"{CKDIR}/lg_{VARIANT}_{SEED}_final.pt" if VARIANT != "additive" else f"{CKDIR}/lg_{SEED}_final.pt"


def make_seq(ctx):
    toks = [random.randint(1, FHI - 1) for _ in range(ctx)]
    slots = random.sample(range(ctx // 2), 2 * P); random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * ctx; meta = []
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


def build(max_pos):
    cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                    short_len=8, max_position_embeddings=max_pos, local_window=WIN,
                    decay_mode="exp", rec_block=32,
                    write_rule=("delta" if VARIANT == "delta" else "additive"),
                    key_feature_map=("dpfp" if VARIANT == "dpfp" else "elu"))
    return HFPForCausalLM(cfg)


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = build(TRAIN_CTX + 8)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    start = 0
    if os.path.exists(CKPT):
        st = torch.load(CKPT, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
        start = st["step"]
        random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])
    t0 = time.time(); model.train(); step = start
    while step < STEPS and time.time() - t0 < BUDGET:
        step += 1
        xs, ys = [], []
        for _ in range(16):
            t, l, _ = make_seq(TRAIN_CTX); xs.append(t); ys.append(l)
        out = model(torch.tensor(xs), labels=torch.tensor(ys))
        assert torch.isfinite(out.loss)
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 200 == 0:
            print(f"[lg-{VARIANT}-{SEED}] step {step} loss {out.loss.item():.3f}", flush=True)
    if step < STEPS:
        torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                        rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
        print(f"[lg-{VARIANT}-{SEED}] CKPT {step}/{STEPS}"); sys.exit(0)
    torch.save(model.state_dict(), FINAL)
    print(f"[lg-{VARIANT}-{SEED}] EGITIM BITTI -> {FINAL}")


def evaluate():
    sd = torch.load(FINAL, weights_only=True)
    for ctx in (160, 320, 640, 1280):
        random.seed(1000 + SEED)
        model = build(ctx + 8)
        sd2 = {k: v for k, v in sd.items() if "pos_encoder.pe" not in k}
        model.load_state_dict(sd2, strict=False)   # PE buffer yeni uzunlukta yeniden uretilir
        model.eval()
        buckets = {}
        with torch.no_grad():
            for _ in range(50):
                toks, _, meta = make_seq(ctx)
                logits = model(torch.tensor([toks])).logits[0]
                for g, ap, v in meta:
                    b = "<48" if g < 48 else "48-127" if g < 128 else "128-255" if g < 256 else "256+"
                    c = buckets.setdefault(b, [0, 0]); c[1] += 1
                    c[0] += int(logits[ap - 1].argmax().item() == v)
        res = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in sorted(buckets.items())}
        print(f"[lg-{VARIANT}-{SEED}] eval ctx={ctx}: {res} (n={ {k: n for k, (c, n) in sorted(buckets.items())} })", flush=True)


if __name__ == "__main__":
    (train if sys.argv[1] == "train" else evaluate)()
