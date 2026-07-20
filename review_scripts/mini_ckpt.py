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

# Checkpoint'li mini deney kosucusu: her cagri ~BUDGET sn calisir, state kaydeder.
# Kullanim: python3 mini_ckpt.py <run_id>
#   run_id: mqar_ck3 | mqar_ck1 | ret_exp | ret_cubic
import random, sys, time, os
import numpy as np, torch
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
from run_experiment import make_mqar_sequence, make_mqar_batch, make_retention_sequence

RUN = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 33.0
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
CKPT = f"{CKDIR}/ck_{RUN}.pt"
STEPS = 800
DEV = "cpu"
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
VOCAB = VHI + 4
PAIRS, CTX, WIN = 4, 48, 16          # mqar
RCTX, RWIN, MAXGAP = 48, 8, 24       # retention

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

spec = {
    "mqar_ck3": dict(task="mqar", decay="exp", ck=3),
    "mqar_ck1": dict(task="mqar", decay="exp", ck=1),
    "ret_exp": dict(task="ret", decay="exp", ck=3),
    "ret_cubic": dict(task="ret", decay="cubic_flux", ck=3),
    "ret_exp_lr3e4": dict(task="ret", decay="exp", ck=3, lr=3e-4),
    "ret_cubic_lr3e4": dict(task="ret", decay="cubic_flux", ck=3, lr=3e-4),
}[RUN]

ctx = CTX if spec["task"] == "mqar" else RCTX
win = WIN if spec["task"] == "mqar" else RWIN
set_seed(0)
cfg = HFPConfig(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=ctx + 8, local_window=win,
                decay_mode=spec["decay"], conv_kernel=spec["ck"])
model = HFPForCausalLM(cfg).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=spec.get("lr", 1e-3))
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
start = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, weights_only=False)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
    start = st["step"]
    random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])

def batch():
    if spec["task"] == "mqar":
        return make_mqar_batch(32, PAIRS, CTX, KLO, KHI, VLO, VHI, FHI, DEV)
    seqs, labels = [], []
    for _ in range(32):
        g = random.randint(1, MAXGAP)
        toks, tgt = make_retention_sequence(g, RCTX, KLO, KHI, VLO, VHI, FHI)
        lab = [-100] * len(toks); lab[-1] = tgt
        seqs.append(toks); labels.append(lab)
    return (torch.tensor(seqs, device=DEV), torch.tensor(labels, device=DEV))

t0 = time.time()
model.train()
step = start
while step < STEPS and time.time() - t0 < BUDGET:
    step += 1
    x, y = batch()
    out = model(x, labels=y)
    assert torch.isfinite(out.loss), f"NaN step {step}"
    opt.zero_grad(set_to_none=True); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sch.step()
    if step % 100 == 0 or step == 1:
        print(f"[{RUN}] step {step} loss {out.loss.item():.4f}", flush=True)

if step < STEPS:
    torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                    rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
    print(f"[{RUN}] CHECKPOINT step={step}/{STEPS}")
    sys.exit(0)

# egitim bitti -> eval
model.eval()
if spec["task"] == "mqar":
    ok = 0; T = 300
    with torch.no_grad():
        for _ in range(T):
            toks, tgt = make_mqar_sequence(PAIRS, CTX, KLO, KHI, VLO, VHI, FHI)
            x = torch.tensor([toks[:-1]], device=DEV)
            ok += int(model(x).logits[0, -1].argmax().item() == tgt)
    print(f"[{RUN}] FINAL acc={100.0*ok/T:.1f}% (sans {100/30:.1f}%)")
else:
    res = {}
    with torch.no_grad():
        for g in (2, 4, 8, 16, 24):
            ok = 0; T = 150
            for _ in range(T):
                toks, tgt = make_retention_sequence(g, RCTX, KLO, KHI, VLO, VHI, FHI)
                x = torch.tensor([toks[:-1]], device=DEV)
                ok += int(model(x).logits[0, -1].argmax().item() == tgt)
            res[g] = 100.0 * ok / T
            print(f"[{RUN}] gap={g} acc={res[g]:.1f}%", flush=True)
    print(f"[{RUN}] FINAL {res} (sans {100/30:.1f}%; 2kat*7=14 -> gap 16/24 saf bellek)")
torch.save(dict(done=True), CKPT + ".done")
