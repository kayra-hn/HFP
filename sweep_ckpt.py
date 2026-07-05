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

"""[SWEEP] Checkpoint'li retention taramasi: python review_scripts/sweep_ckpt.py <mode> <lr> <seed>
Saf-bellek tasarimi: 2 katman x pencere 8 -> yigilmis attention alani ~14; gap 16/24 yalnizca bellek.
800 adimda egitir, gap {2,8,16,24} acc raporlar. Zaman butcesi asilirsa checkpoint'e yazar, cikar."""
import os, random, sys, time
import numpy as np, torch
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
from run_experiment import make_retention_sequence

MODE, LR, SEED = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 33.0
TAG = f"{MODE}_{LR:g}_{SEED}"
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
CKPT = f"{CKDIR}/sw_{TAG}.pt"
STEPS = 800
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
CTX, WIN, MAXGAP = 48, 8, 24

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                decay_mode=MODE, rec_block=16)
model = HFPForCausalLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
start = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, weights_only=False)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
    start = st["step"]
    random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])

def batch():
    seqs, labels = [], []
    for _ in range(32):
        g = random.randint(1, MAXGAP)
        toks, tgt = make_retention_sequence(g, CTX, KLO, KHI, VLO, VHI, FHI)
        lab = [-100] * len(toks); lab[-1] = tgt
        seqs.append(toks); labels.append(lab)
    return torch.tensor(seqs), torch.tensor(labels)

t0 = time.time(); model.train(); step = start
while step < STEPS and time.time() - t0 < BUDGET:
    step += 1
    x, y = batch()
    out = model(x, labels=y)
    assert torch.isfinite(out.loss), f"NaN {TAG} step {step}"
    opt.zero_grad(set_to_none=True); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sch.step()
    if step % 200 == 0:
        print(f"[{TAG}] step {step} loss {out.loss.item():.3f}", flush=True)

if step < STEPS:
    torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                    rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
    print(f"[{TAG}] CKPT {step}/{STEPS}"); sys.exit(0)

model.eval(); res = {}
with torch.no_grad():
    for g in (2, 8, 16, 24):
        ok = 0; T = 150
        for _ in range(T):
            toks, tgt = make_retention_sequence(g, CTX, KLO, KHI, VLO, VHI, FHI)
            ok += int(model(torch.tensor([toks[:-1]])).logits[0, -1].argmax().item() == tgt)
        res[g] = round(100.0 * ok / T, 1)
print(f"[{TAG}] FINAL {res}")
with open(f"{CKDIR}/sweep_results.txt", "a") as f:
    f.write(f"{TAG} {res}\n")
