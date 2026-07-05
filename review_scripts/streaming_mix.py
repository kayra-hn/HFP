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

"""[STREAM-MIX] dpfp x delta bilesiminin dogal rejimi: uzun akis + cok olgu +
guncellemeler karisik. Her anahtar %40 olasilikla iki kez yazilir (guncelleme);
sorgu SON degeri ister. Egitim ctx 160 / P 8; eval hem 160 hem 640 (P=24:
yuksek girisim + guncelleme). Kullanim: streaming_mix.py <write> <fmap> <seed> [budget]"""
import os, random, sys, time
import numpy as np, torch
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

WRITE, FMAP, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3])
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 33.0
TAG = f"sm_{WRITE}_{FMAP}_{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"
STEPS = 600
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
TRAIN_CTX, WIN, TRAIN_P = 160, 8, 8
ANS = 0

def make_seq(ctx, P):
    toks = [random.randint(1, FHI - 1) for _ in range(ctx)]
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * ctx; meta = []
    # anahtar basina 1 (%60) veya 2 (%40) yazim + 1 sorgu -> slot ihtiyaci degisken
    events = []
    for i in range(P):
        n_w = 2 if random.random() < 0.4 else 1
        events.append((keys[i], n_w))
    need = sum(nw + 1 for _, nw in events)
    slots = sorted(random.sample(range(ctx // 2), need))
    # anahtarlarin olaylarini rastgele serpistir ama kendi ici sirali kalsin:
    # slot listesinden her anahtara (nw+1) RASTGELE konum ata, sirala.
    idxs = list(range(need)); random.shuffle(idxs)
    ptr = 0
    for k, nw in events:
        mine = sorted(idxs[ptr:ptr + nw + 1]); ptr += nw + 1
        pos = [slots[i] * 2 for i in mine]
        vals = []
        for wpos in pos[:-1]:
            v = random.randint(VLO, VHI - 1)
            while vals and v == vals[-1]:
                v = random.randint(VLO, VHI - 1)
            vals.append(v)
            toks[wpos] = k; toks[wpos + 1] = v
        qp = pos[-1]
        toks[qp] = k; toks[qp + 1] = ANS
        lab[qp + 1] = vals[-1]
        meta.append((qp + 1, vals[-1], vals[0] if len(vals) > 1 else None))
    return toks, lab, meta

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=1288, local_window=WIN,
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
        t, l, _ = make_seq(TRAIN_CTX, TRAIN_P); xs.append(t); ys.append(l)
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
for ctx, P, trials in ((160, 8, 60), (640, 24, 30)):
    okS = nS = okU = nU = stale = 0
    random.seed(3000 + SEED)
    with torch.no_grad():
        for _ in range(trials):
            toks, _, meta = make_seq(ctx, P)
            logits = model(torch.tensor([toks])).logits[0]
            for ap, vlast, v1 in meta:
                pred = logits[ap - 1].argmax().item()
                if v1 is None:
                    nS += 1; okS += int(pred == vlast)
                else:
                    nU += 1; okU += int(pred == vlast); stale += int(pred == v1)
    print(f"[{TAG}] eval ctx={ctx} P={P}: tek-yazim={100.0*okS/max(1,nS):.1f}%  "
          f"guncellenen={100.0*okU/max(1,nU):.1f}%  bayat={100.0*stale/max(1,nU):.1f}%", flush=True)
    with open(f"{CKDIR}/sm_results.txt", "a") as f:
        f.write(f"{TAG} ctx{ctx} single={100.0*okS/max(1,nS):.1f} upd={100.0*okU/max(1,nU):.1f} stale={100.0*stale/max(1,nU):.1f}\n")
