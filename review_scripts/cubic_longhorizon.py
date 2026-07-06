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

"""[FAZ 1B] cubic_flux ADIL testi — 2x2 (retention x feature-map) uzun-ufukta.

Projenin asil ozgun iddiasinin karar deneyi. cubic+additive zaten kaybetti
(DENEY Ek 4); test edilecek dogru hucre cubic'in platosunun teorik avantaj
tasidigi yer: SEYREK (P=8) + UZUN GAP + DPFP (kanallari seyrek tutar -> plato
korunur). Egitim ctx 160 (ogrenilebilir zemin, DENEY Ek 6), eval uzun (640/1280)
-> gap 128-255 ve 256+ kovalari platonun bolgesi.

exp'in cok-olcekli lambda'si (0.90..0.999) guclu bir baseline oldugundan,
cubic'in kazanmasi icin bu seyrek-uzun bolgede exp'i ACIKCA gecmesi gerekir.

ONCEDEN YAZILAN BASARI KRITERI (post-hoc rasyonalizasyonu onler):
  "cubic_flux_chunked + dpfp, 256+ kovasinda exp + dpfp'yi 3 seed ortalamasinda
   >2 standart hata gecerse cubic'in uzun-ufuk avantaji DOGRULANDI; aksi halde
   hipotez reddedilir / parked kalir."

cubic LR'a duyarli (INCELEME: sabit LR kiyasi haksiz) -> mode basina LR verilir.

Kullanim: python review_scripts/cubic_longhorizon.py <retention> <fmap> <lr> <seed> [budget]
  retention: exp | cubic_flux_chunked
  fmap:      elu | dpfp
  orn: python review_scripts/cubic_longhorizon.py cubic_flux_chunked dpfp 1e-3 0 600
Sonuc: CKDIR/cubic_lh_results.txt'e eklenir + ekrana basilir.
"""
import os, random, sys, time
import numpy as np, torch
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

RET  = sys.argv[1]            # exp | cubic_flux_chunked
FMAP = sys.argv[2]            # elu | dpfp
LR   = float(sys.argv[3])
SEED = int(sys.argv[4])
BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 600.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100
WIN, P = 8, 8                 # seyrek rejim: P=8
ANS = 0
TRAIN_CTX, STEPS = 160, 600
EVAL_CTXS = (640, 1280)
TAG = f"clh_{RET}_{FMAP}_{LR:g}_{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"
FINAL = f"{CKDIR}/{TAG}_final.pt"


def make_seq(ctx):
    """dense: P cift + P sorgu, gap dogal 1..~ctx/2 dagilir."""
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
        meta.append((qp + 1 - (wp + 1), qp + 1, v))
    return toks, lab, meta


def build(max_pos):
    cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                    short_len=8, max_position_embeddings=max_pos, local_window=WIN,
                    decay_mode=RET, rec_block=32, write_rule="additive",
                    key_feature_map=FMAP, ffn_type="standard")
    return HFPForCausalLM(cfg)


def bname(g):
    return "<48" if g < 48 else "48-127" if g < 128 else "128-255" if g < 256 else "256+"


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = build(TRAIN_CTX + 8).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    start = 0
    if os.path.exists(CKPT):
        st = torch.load(CKPT, weights_only=False)
        model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
        start = st["step"]; random.setstate(st["rng"])
        np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])
    t0 = time.time(); model.train(); step = start
    while step < STEPS and time.time() - t0 < BUDGET:
        step += 1
        xs, ys = [], []
        for _ in range(16):
            t, l, _ = make_seq(TRAIN_CTX); xs.append(t); ys.append(l)
        out = model(torch.tensor(xs, device=DEV), labels=torch.tensor(ys, device=DEV))
        assert torch.isfinite(out.loss), f"NaN {TAG} step {step}"
        opt.zero_grad(set_to_none=True); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 100 == 0:
            print(f"[{TAG}] step {step} loss {out.loss.item():.3f}", flush=True)
    if step < STEPS:
        torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                        rng=random.getstate(), nprng=np.random.get_state(), trng=torch.get_rng_state()), CKPT)
        print(f"[{TAG}] CKPT {step}/{STEPS}"); return False
    torch.save(model.state_dict(), FINAL)
    return True


def evaluate():
    sd = torch.load(FINAL, weights_only=True)
    for ctx in EVAL_CTXS:
        random.seed(2000 + SEED)
        model = build(ctx + 8).to(DEV)
        sd2 = {k: v for k, v in sd.items() if "pos_encoder.pe" not in k}
        model.load_state_dict(sd2, strict=False)
        model.eval()
        buckets = {"<48": [0, 0], "48-127": [0, 0], "128-255": [0, 0], "256+": [0, 0]}
        with torch.no_grad():
            for _ in range(50):
                toks, _, meta = make_seq(ctx)
                logits = model(torch.tensor([toks], device=DEV)).logits[0]
                for g, ap, v in meta:
                    b = buckets[bname(g)]; b[1] += 1
                    b[0] += int(logits[ap - 1].argmax().item() == v)
        res = {k: round(100.0 * c / max(1, n), 1) for k, (c, n) in buckets.items()}
        print(f"[{TAG}] eval ctx={ctx}: {res}", flush=True)
        with open(f"{CKDIR}/cubic_lh_results.txt", "a") as f:
            f.write(f"{TAG} ctx{ctx} {res}\n")


if __name__ == "__main__":
    done = train()
    if done:
        evaluate()
        print(f"[{TAG}] EGITIM BITTI (eval yazildi)")
    else:
        # butce bitti; tekrar cagirinca kaldigi yerden devam eder
        raise SystemExit(0)
