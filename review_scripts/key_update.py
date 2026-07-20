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

"""[KEY-UPDATE] Delta yaziminin adil testi: ayni anahtar iki kez yazilir
(k->v1 ... k->v2 ...), sorgu IKINCI yazimdan sonra gelir, dogru cevap v2.
Additive bellek k(x)v1 + k(x)v2 karisimini ayirt edemez (yapisal tavan);
delta ikinci yazimda v1'i siler. 2x2: write_rule x key_feature_map.
Kullanim: python review_scripts/key_update.py <write_rule> <fmap> <seed> [budget]"""
import os, random, sys, time
import numpy as np, torch
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

WRITE, FMAP, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 33.0
TAG = f"ku_{WRITE}_{FMAP}_{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"
STEPS = 600
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, WIN, P = 160, 8, 5
ANS = 0

def make_seq():
    """P anahtar; her biri icin: [k v1] ... [k v2] ... [k ANS] (cevap v2).
    3 olay/anahtar -> 3P slot. Sira: yazim1 < yazim2 < sorgu."""
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    slots = random.sample(range(CTX // 2), 3 * P); random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * CTX; meta = []
    for i in range(P):
        a, b, c = sorted([slots[3 * i], slots[3 * i + 1], slots[3 * i + 2]])
        w1, w2, qp = 2 * a, 2 * b, 2 * c
        v1 = random.randint(VLO, VHI - 1)
        v2 = random.randint(VLO, VHI - 1)
        while v2 == v1:
            v2 = random.randint(VLO, VHI - 1)
        toks[w1] = keys[i]; toks[w1 + 1] = v1
        toks[w2] = keys[i]; toks[w2 + 1] = v2
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        lab[qp + 1] = v2
        meta.append((qp + 1, v2, v1))
    return toks, lab, meta

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                decay_mode="exp", rec_block=32, write_rule=WRITE, key_feature_map=FMAP)
model = HFPForCausalLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
    assert torch.isfinite(out.loss), f"NaN {TAG} {step}"
    opt.zero_grad(set_to_none=True); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sch.step()
    if step % 200 == 0:
        print(f"[{TAG}] step {step} loss {out.loss.item():.3f}", flush=True)
if step < STEPS:
    torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                    rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
    print(f"[{TAG}] CKPT {step}/{STEPS}"); sys.exit(0)

model.eval()
ok = old = other = 0; T = 100
with torch.no_grad():
    for _ in range(T):
        toks, _, meta = make_seq()
        logits = model(torch.tensor([toks])).logits[0]
        for ap, v2, v1 in meta:
            pred = logits[ap - 1].argmax().item()
            ok += int(pred == v2); old += int(pred == v1)
            other += int(pred != v2 and pred != v1)
n = T * P
print(f"[{TAG}] FINAL dogru(v2)={100.0*ok/n:.1f}%  ESKI-deger(v1)={100.0*old/n:.1f}%  diger={100.0*other/n:.1f}%")
with open(f"{CKDIR}/ku_results.txt", "a") as f:
    f.write(f"{TAG} v2={100.0*ok/n:.1f} v1={100.0*old/n:.1f}\n")
