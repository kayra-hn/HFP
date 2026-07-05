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

"""[DENSE-RET] Yogun-supervizyonlu retention: ayni dizide P cift + P sorgu.
Kullanim: python review_scripts/dense_retention.py <mode> <write_rule> <lr> <seed> [budget]
Yapi (ctx 160): 2-token'lik 2P slot rastgele yerlesir; her cift icin yazim slotu
[k v], sorgu slotu [k ANS] (yazimdan SONRA). Etiketler tum ANS pozisyonlarinda
-> sinyal ~Px. Eval: gap kovalarina gore acc (gap = v-pozisyonu ile ANS arasi).
Pencere 8, 2 katman -> alici alan ~14; gap>=16 saf bellek. Sans %3.3."""
import os
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

import os, random, sys, time
import numpy as np, torch
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

MODE, WRITE, LR, SEED = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 33.0
TAG = f"dense_{MODE}_{WRITE}_{LR:g}_{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"
STEPS = 600
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, WIN, P = 160, 8, 8
ANS = 0

def make_seq():
    """dizi + (label listesi) + [(gap, ans_pos, v)] uretir."""
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    nslot = CTX // 2
    slots = random.sample(range(nslot), 2 * P)
    random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * CTX
    meta = []
    for i in range(P):
        a, b = slots[2 * i], slots[2 * i + 1]
        if a > b: a, b = b, a
        wp, qp = 2 * a, 2 * b          # yazim/sorgu token indeksleri
        v = random.randint(VLO, VHI - 1)
        toks[wp] = keys[i]; toks[wp + 1] = v
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        lab[qp + 1] = v
        meta.append((qp + 1 - (wp + 1), qp + 1, v))   # gap, ans_pos, hedef
    return toks, lab, meta

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                decay_mode=MODE, rec_block=32, write_rule=WRITE)
model = HFPForCausalLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
start = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, weights_only=False)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
    start = st["step"]
    random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])

def batch(bs=16):
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
buckets = {"1-15": [0, 0], "16-47": [0, 0], "48-95": [0, 0], "96+": [0, 0]}
def bname(g):
    return "1-15" if g < 16 else "16-47" if g < 48 else "48-95" if g < 96 else "96+"
with torch.no_grad():
    for _ in range(80):
        toks, _, meta = make_seq()
        logits = model(torch.tensor([toks])).logits[0]
        for g, ap, v in meta:
            b = buckets[bname(g)]
            b[1] += 1
            b[0] += int(logits[ap - 1].argmax().item() == v)   # ANS'tan onceki logit hedefi tahmin eder
res = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in buckets.items()}
ns = {k: n for k, (c, n) in buckets.items()}
print(f"[{TAG}] FINAL {res} (n={ns})")
with open(f"{CKDIR}/dense_results.txt", "a") as f:
    f.write(f"{TAG} {res}\n")
