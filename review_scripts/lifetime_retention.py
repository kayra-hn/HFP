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

"""[GOREV C v2 / OMUR TESTI]  (v1 GECERSIZDI: tek-denetimli egitim §1 geregi
ln(30) platosunda kaldi — 5/6 kol hic ogrenmedi; v2 egitimi dense_retention
tarzi YOGUN denetime cevirdi. Eval (streaming omur sondasi) degismedi.)

[GOREV C / OMUR TESTI] Cihaz-ici uzun-omurlu ajan rejimi: seyrek olgu,
yogun trafik, egitim ufkunun COK otesinde geri getirme.

MOTIVASYON (on-kayit, 2026-07-19): §15h graft kiyasinda cubic'in avantaji
cikmadi; ama o test kisa-orta ufuk + surekli-doygun state rejimiydi. Cubic'in
teorik vaadi BURADA yasiyor: bos/seyrek kanal neredeyse unutmaz (plateau),
dolu kanal kendini sinirlar. Exp'in yapisal siniri: sabit geometrik yari-omur
(cok-olcekli ogrenilmis kanal-basi lambda dahil — adil kontrol budur).

TASARIM
  Egitim (hard_retention ile ayni protokol): ctx 320, pencere 8, 2 katman;
  hedef cift + 6 distraktor, gap ~ U{1..256}; 3 seed; MODE ∈ {exp,
  cubic_flux_chunked}; write=additive (K2 kilitli recete), dpfp.
  Eval (yeni kisim): STREAMING (use_cache, chunk 256) ile hedef cifti yaz,
  ardindan G token TRAFIK akit (dolgu + her 64 tokenda bir distraktor kv —
  seyrek-yazim rejimi; hedef asla tekrarlanmaz), sonra sorgula.
  G ∈ {256, 1024, 4096, 16384, 65536}  (egitim tavani 256'nin 256 katina dek).
  Metrik: acc% (sans %3.3) + hedef log-prob, gap-egrisi olarak.

ON-KAYITLI KRITERLER (kosudan ONCE yazildi; sonuc ne cikarsa RESULTS'a):
  - Birincil: G >= 4096 bolgesinde acc(cubic) - acc(exp) seed-ortalama farki.
    "Cubic nisi kanitlandi" esigi: ardisik iki uzak gap'te >= +10 puan VE
    seed araliklari ayrisiyor. Aksi yonde ayni esik exp icin de gecerli.
  - Ikincil: egrinin SEKLI — exp geometrik sonum (log-acc ~ dogrusal dusus)
    vs cubic plato (uzak kuyrukta duzlesme) imzasi.
  - Iki mod da uzak kuyruklarda sansa cokerse: "iki yasa da bu rejimi
    tutamiyor" — durust sonuc, nis kanitlanmadi demektir.

KULLANIM
  python review_scripts/lifetime_retention.py <mode> <seed> [train_budget_sn]
    mode: exp | cubic_flux_chunked      seed: 0|1|2
  Tum kollar:
    for m in exp cubic_flux_chunked; do for s in 0 1 2; do
      python review_scripts/lifetime_retention.py $m $s 900; done; done
  Env: LT_GAPS="256,1024,4096,16384,65536"  LT_TRIALS=30  LT_CHUNK=256
       LT_STEPS=800  LT_DIST_EVERY=64  HFP_CKPT_DIR=checkpoints
Cikti: {CKDIR}/lifetime_results.txt (+ ayrintili CSV {CKDIR}/lifetime_{TAG}.csv)
"""
import os
CKDIR = os.environ.get("HFP_CKPT_DIR", "checkpoints")
os.makedirs(CKDIR, exist_ok=True)

import csv, random, sys, time
import numpy as np, torch
import torch.nn.functional as Fn
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

MODE, SEED = sys.argv[1], int(sys.argv[2])
assert MODE in ("exp", "cubic_flux_chunked"), MODE
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
GAPS = [int(g) for g in os.environ.get("LT_GAPS", "256,1024,4096,16384,65536").split(",")]
TRIALS = int(os.environ.get("LT_TRIALS", "30"))
CHUNK = int(os.environ.get("LT_CHUNK", "256"))
STEPS = int(os.environ.get("LT_STEPS", "800"))
DIST_EVERY = int(os.environ.get("LT_DIST_EVERY", "64"))

# hard_retention ile ayni soz dagarcigi/gorev sabitleri
KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100   # sans = 1/30
CTX, WIN, MAXGAP, NDIS = 320, 8, 256, 6
ANS = 0
TAG = f"lifetimev2_{MODE}_s{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                decay_mode=MODE, rec_block=32, write_rule="additive",
                key_feature_map="dpfp",
                pe_period=CTX)   # [§14 E2] period-PE: egitim-otesi uzunlukta stabil
model = HFPForCausalLM(cfg)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEV)

# ---------------- EGITIM (dense_retention protokolu — §1: YOGUN denetim sart) ----------------
P = 8   # dizide 8 olgu + 8 sorgu (tek-denetim ctx>300'de OGRENMIYOR; RESULTS §1)
def make_train_seq():
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    nslot = CTX // 2
    slots = random.sample(range(nslot), 2 * P)
    random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), P)
    lab = [-100] * CTX
    for i in range(P):
        a, b = slots[2 * i], slots[2 * i + 1]
        if a > b: a, b = b, a
        wp, qp = 2 * a, 2 * b
        v = random.randint(VLO, VHI - 1)
        toks[wp] = keys[i]; toks[wp + 1] = v
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        lab[qp + 1] = v
    return toks, lab

def batch(bs=16):
    seqs, labels = [], []
    for _ in range(bs):
        toks, lab = make_train_seq()
        seqs.append(toks); labels.append(lab)
    return torch.tensor(seqs, device=DEV), torch.tensor(labels, device=DEV)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
start = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, weights_only=False, map_location=DEV)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
    start = st["step"]
    random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])
    print(f"[{TAG}] resume @ {start}")

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
        print(f"[{TAG}] step {step}/{STEPS} loss {out.loss.item():.3f}", flush=True)

if step < STEPS:
    torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                    rng=random.getstate(), nprng=np.random.get_state(),
                    trng=torch.get_rng_state()), CKPT)
    print(f"[{TAG}] CKPT {step}/{STEPS} — butce bitti, tekrar calistir (resume eder)")
    sys.exit(0)

# ---------------- EVAL: STREAMING OMUR TESTI ----------------
# Hedef cift en basta yazilir; G token trafik (dolgu + seyrek distraktor kv);
# sonda sorgu. Egitimdekiyle ayni dagilim ailesi, SADECE mesafe/rejim degisir.
def traffic_tokens(n, forbid_key):
    toks = []
    while len(toks) < n:
        seg = min(DIST_EVERY, n - len(toks))
        toks.extend(random.randint(1, FHI - 1) for _ in range(seg - 2 if seg > 2 else seg))
        if seg > 2:
            k = random.choice([k for k in range(KLO, KHI) if k != forbid_key])
            toks.extend([k, random.randint(VLO, VHI - 1)])
    return toks[:n]

@torch.no_grad()
def probe(gap):
    tgt_k = random.randint(KLO, KHI - 1)
    v = random.randint(VLO, VHI - 1)
    seq = [tgt_k, v] + traffic_tokens(gap, tgt_k) + [tgt_k]
    ids = torch.tensor([seq], device=DEV)
    past = None; logits = None
    for s in range(0, ids.size(1), CHUNK):
        out = model(ids[:, s:s + CHUNK], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    ok = int(logits.argmax().item() == v)
    lp = Fn.log_softmax(logits, -1)[v].item()
    return ok, lp

# [PLATO KORUMASI] Egitim sans-platosunda kaldiysa kiyas GECERSIZ (v1 dersi).
with torch.no_grad():
    _x, _y = batch(bs=32)
    _l = model(_x, labels=_y).loss.item()
print(f"[{TAG}] egitim-sonu dogrulama loss: {_l:.3f} (sans ~3.40)")
assert _l < 2.5, (f"OGRENMEDI: loss {_l:.2f} ~ sans platosu; kiyas gecersiz. (RESULTS §1)")
model.eval()
rows = []; res = {}
for g in GAPS:
    ok = 0; lp = 0.0
    te = time.time()
    for t in range(TRIALS):
        o, l = probe(g); ok += o; lp += l
    res[g] = (round(100.0 * ok / TRIALS, 1), round(lp / TRIALS, 2))
    rows.append(dict(mode=MODE, seed=SEED, gap=g, acc=res[g][0], logprob=res[g][1],
                     trials=TRIALS, sec=round(time.time() - te, 1)))
    print(f"[{TAG}] gap {g:>6}: acc {res[g][0]:5.1f}%  logprob {res[g][1]:.2f}  "
          f"({time.time()-te:.0f}s)", flush=True)

print(f"[{TAG}] FINAL acc%/logprob (sans %3.3): {res}")
with open(f"{CKDIR}/lifetime_results.txt", "a") as f:
    f.write(f"{TAG} {res}\n")
with open(f"{CKDIR}/{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[{TAG}] yazildi: {CKDIR}/lifetime_results.txt + lifetime_{TAG}.csv")
