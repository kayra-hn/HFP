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

"""[HARD-RET] Ayristirici retention gorevi: uzun gap + distraktor girisimi.
Kullanim: python review_scripts/hard_retention.py <mode> <lr> <seed> [budget_sn]

Yapi (context=320, pencere=8, 2 katman -> alici alan ~14; gap>=32 saf bellek):
  [d1 v1 ... d6 v6]  ...filler...  [k* v*]  ...filler(gap)...  k* ANS
  - 6 distraktor cift basta (girisim), hedef cift sorgudan tam 'gap' once.
  - Egitim: gap ~ U{1..256}. Eval: gap {8,32,64,128,256}, acc + hedef log-prob.
Sans = 1/30 = %3.3. Gorev doyarsa (her yerde ~%100) daha da zorlastirilmali."""
import os
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

import os, random, sys, time
import numpy as np, torch
import torch.nn.functional as Fn
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

MODE, LR, SEED = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 33.0
CURR = len(sys.argv) > 5 and sys.argv[5] == "curriculum"
WRITE = os.environ.get("HR_WRITE", "additive")   # additive | delta
CTX_E = int(os.environ.get("HR_CTX", "320"))
MAXG_E = int(os.environ.get("HR_MAXGAP", "256"))
STEPS_E = int(os.environ.get("HR_STEPS", "800"))
TAG = f"hard{'c' if CURR else ''}_{MODE}_{WRITE}_c{CTX_E}_{LR:g}_{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"
STEPS = None  # asagida STEPS_E
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, WIN, MAXGAP, NDIS = None, 8, None, 6
CTX, MAXGAP, STEPS = CTX_E, MAXG_E, STEPS_E
ANS = 0

def make_seq(gap):
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    keys = random.sample(range(KLO, KHI), NDIS + 1)
    tgt_k = keys[0]
    # distraktorler basta
    for i, k in enumerate(keys[1:]):
        toks[2 * i] = k
        toks[2 * i + 1] = random.randint(VLO, VHI - 1)
    # hedef cift: sorgudan tam 'gap' once (v pozisyonu)
    vpos = CTX - 2 - gap
    kpos = vpos - 1
    assert kpos >= 2 * NDIS, "gap cok buyuk / context yetersiz"
    v = random.randint(VLO, VHI - 1)
    toks[kpos] = tgt_k; toks[vpos] = v
    toks[CTX - 2] = tgt_k; toks[CTX - 1] = ANS
    return toks, v

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

def batch(bs=16, step=0):
    # [CURRICULUM] gap tavani adimla buyur: 16 + step/2 (400. adimda 216, 480+'da 256)
    gmax = min(MAXGAP, 16 + step // 2) if CURR else MAXGAP
    seqs, labels = [], []
    for _ in range(bs):
        g = random.randint(1, gmax)
        toks, v = make_seq(g)
        lab = [-100] * CTX; lab[-1] = v
        seqs.append(toks); labels.append(lab)
    return torch.tensor(seqs), torch.tensor(labels)

t0 = time.time(); model.train(); step = start
while step < STEPS and time.time() - t0 < BUDGET:
    step += 1
    x, y = batch(step=step)
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

model.eval(); res = {}
with torch.no_grad():
    for g in [g for g in (8, 32, 64, 128, 256) if g <= MAXGAP]:
        ok = 0; lp = 0.0; T = 120
        for _ in range(T):
            toks, v = make_seq(g)
            logits = model(torch.tensor([toks[:-1]])).logits[0, -1]
            ok += int(logits.argmax().item() == v)
            lp += Fn.log_softmax(logits, -1)[v].item()
        res[g] = (round(100.0 * ok / T, 1), round(lp / T, 2))
print(f"[{TAG}] FINAL acc%/logprob: {res}")
with open(f"{CKDIR}/hard_results.txt", "a") as f:
    f.write(f"{TAG} {res}\n")
