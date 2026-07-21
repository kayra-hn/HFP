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

"""[GOREV E] Tasima mufredati: "ogrendigi mesafede hatirlar" hipotezinin testi.

NEDEN (RESULTS §17-§18 zinciri):
  §17: >1024 tokende bellek sansa cokuyor.  §15h: sebep retention yasasi degil.
  §18: sebep kapasite/girisim de degil (64x seyreltme kurtarmadi).
  Geriye kalan aciklama: model EGITIMDE hicbir zaman uzun tasima gormedi
  (egitim ufku 256). Graft hattinda ayni sorun MESAFE MUFREDATI ile cozuldu:
  Run 4 (@512 var, @8192 yok) -> Run 5 (mufredat) -> @8192/@16384 BULDU (§15f).
  Bu script o mufredati kucuk-olcek modele uygular.

NASIL (v2 omur testinden TEK FARK: egitim; eval birebir ayni):
  Egitim artik STREAMING + chunklar arasi. Her ornek:
    CHUNK A : yogun-denetimli dizi (dense_retention) + sona gomulu HEDEF cift
    K adet  : dolgu chunk'i (state tasinir, grad yok — TBPTT sinirli)
    CHUNK B : sorgu + cevap (loss burada; attention A'yi GOREMEZ, use_cache=False)
  K ~ mufredat: ilk %30 adimda K∈{0,1}, sonra CARRY_MAX'a kadar buyur.
  Boylece egitilen tasima mesafesi ~ CARRY_MAX * CTX token olur.
  Ayrica her chunk icinde dense denetim korunur (§1: seyrek denetim ogrenmiyor).

ON-KAYITLI KRITERLER (kosudan ONCE yazildi):
  Karsilastirma tabani = §17 v2 (mufredatsiz) ayni gap egrisi:
    exp   256:15.6  1024:4.4  4096:1.1  16384:5.6  65536:1.1   (sans 3.3)
    cubic 256:18.9  1024:2.2  4096:2.2  16384:2.2  65536:2.2
  - BASARI: mufredatli modelde 4096 VE 16384 gaplerinde seed-ortalama acc
    >= %15 (sansin ~4.5 kati) -> "ogrendigi mesafede hatirlar" DOGRULANDI;
    cihaz-ici iddia "egitilen tasima mesafesi kadar omur" olarak ifade edilir.
  - KISMI: yalniz 4096'da >= %15 -> mufredat calisiyor ama menzil egitilen
    tavanla sinirli; CARRY_MAX buyutulup tekrarlanir.
  - BASARISIZ: her iki uzak gap de < %8 -> mufredat aciklamasi da REDDEDILIR;
    o zaman sorun mimari (state boyutu/okuma yolu), rapora oyle yazilir.
  - Yasa kiyasi ikincildir: exp vs cubic farki yine >= 10 puan degilse §15h/§17
    "yasa fark etmiyor" sonucu pekisir.

KULLANIM
  python review_scripts/carry_curriculum.py <mode> <seed> [train_budget_sn]
  Env: CC_CARRY_MAX=16   CC_STEPS=1200   CC_CTX=256   CC_P=6
       CC_GAPS="256,1024,4096,16384"  CC_TRIALS=30  HFP_CKPT_DIR=checkpoints
Cikti: {CKDIR}/carry_results.txt + {CKDIR}/carryv1_{mode}_s{seed}.csv
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
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0

CARRY_MAX = int(os.environ.get("CC_CARRY_MAX", "16"))   # max dolgu chunk sayisi
STEPS = int(os.environ.get("CC_STEPS", "1200"))
CTX = int(os.environ.get("CC_CTX", "256"))              # chunk uzunlugu
P = int(os.environ.get("CC_P", "6"))                    # chunk-ici dense cift sayisi
GAPS = [int(g) for g in os.environ.get("CC_GAPS", "256,1024,4096,16384").split(",")]
TRIALS = int(os.environ.get("CC_TRIALS", "30"))
DIST_EVERY = int(os.environ.get("CC_DIST_EVERY", "64"))

KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100      # sans = 1/30
WIN, ANS = 8, 0
TAG = f"carryv1_{MODE}_s{SEED}"
CKPT = f"{CKDIR}/{TAG}.pt"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                decay_mode=MODE, rec_block=32, write_rule="additive",
                key_feature_map="dpfp", pe_period=CTX)
model = HFPForCausalLM(cfg)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEV)
print(f"[{TAG}] cihaz={DEV} carry_max={CARRY_MAX} (~{CARRY_MAX*CTX} token tasima) ctx={CTX}")

# ---------------- EGITIM: streaming + chunklar arasi tasima ----------------
def dense_chunk(reserve_tail=0):
    """Chunk-ici yogun denetimli dizi (§1). reserve_tail: sona bosluk birak."""
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
    lab = [-100] * CTX
    usable = (CTX - reserve_tail) // 2
    slots = random.sample(range(usable), min(2 * P, usable))
    random.shuffle(slots)
    keys = random.sample(range(KLO, KHI), len(slots) // 2)
    for i in range(len(slots) // 2):
        a, b = slots[2 * i], slots[2 * i + 1]
        if a > b: a, b = b, a
        wp, qp = 2 * a, 2 * b
        v = random.randint(VLO, VHI - 1)
        toks[wp] = keys[i]; toks[wp + 1] = v
        toks[qp] = keys[i]; toks[qp + 1] = ANS
        lab[qp + 1] = v
    return toks, lab

def filler_chunk(forbid):
    """Dolgu trafigi: seyrek distraktor kv (omur testindeki trafikle ayni aile)."""
    toks = []
    while len(toks) < CTX:
        n = min(DIST_EVERY, CTX - len(toks))
        toks.extend(random.randint(1, FHI - 1) for _ in range(max(0, n - 2)))
        if n > 2:
            k = random.choice([k for k in range(KLO, KHI) if k != forbid])
            toks.extend([k, random.randint(VLO, VHI - 1)])
    return toks[:CTX]

def carry_example(K):
    """A: hedef cift sonda gomulu | K dolgu chunk | B: sorgu+cevap sonda."""
    tgt_k = random.randint(KLO, KHI - 1)
    v = random.randint(VLO, VHI - 1)
    a_tok, a_lab = dense_chunk(reserve_tail=4)
    a_tok[CTX - 2] = tgt_k; a_tok[CTX - 1] = v      # HEDEF yazimi A'nin sonunda
    a_lab[CTX - 2] = -100;  a_lab[CTX - 1] = -100
    fills = [filler_chunk(tgt_k) for _ in range(K)]
    b_tok, b_lab = dense_chunk(reserve_tail=4)
    b_tok[CTX - 2] = tgt_k; b_tok[CTX - 1] = ANS    # SORGU B'nin sonunda
    b_lab[CTX - 2] = -100;  b_lab[CTX - 1] = v      # loss: cross-chunk hedef
    return a_tok, a_lab, fills, b_tok, b_lab

def curriculum_K(step):
    """Ilk %30: K∈{0,1}; sonra dogrusal olarak CARRY_MAX'a buyur."""
    warm = int(0.3 * STEPS)
    if step <= warm: return random.choice([0, 1])
    frac = (step - warm) / max(1, STEPS - warm)
    return random.randint(0, max(1, int(frac * CARRY_MAX)))

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
start = 0
if os.path.exists(CKPT):
    st = torch.load(CKPT, weights_only=False, map_location=DEV)
    model.load_state_dict(st["m"]); opt.load_state_dict(st["o"]); sch.load_state_dict(st["s"])
    start = st["step"]
    random.setstate(st["rng"]); np.random.set_state(st["nprng"]); torch.set_rng_state(st["trng"])
    print(f"[{TAG}] resume @ {start}")

BS = int(os.environ.get("CC_BS", "8"))
t0 = time.time(); model.train(); step = start
while step < STEPS and time.time() - t0 < BUDGET:
    step += 1
    K = curriculum_K(step)
    A, AL, FS, B, BL = [], [], [], [], []
    for _ in range(BS):
        a, al, fills, b, bl = carry_example(K)
        A.append(a); AL.append(al); FS.append(fills); B.append(b); BL.append(bl)
    xa = torch.tensor(A, device=DEV); ya = torch.tensor(AL, device=DEV)
    xb = torch.tensor(B, device=DEV); yb = torch.tensor(BL, device=DEV)

    opt.zero_grad(set_to_none=True)
    # A: chunk-ici dense loss (yazma yolu ogrenir) + state uret
    out_a = model(xa, labels=ya, use_cache=True)
    past = out_a.past_key_values
    loss_a = out_a.loss
    loss_a.backward()                      # A'nin grafigini hemen bosalt
    # dolgu chunklari: state tasinir, gradyan yok
    with torch.no_grad():
        for j in range(K):
            xf = torch.tensor([FS[i][j] for i in range(BS)], device=DEV)
            past = model(xf, past_key_values=past, use_cache=True).past_key_values
    past = [tuple(t.detach() if torch.is_tensor(t) else t for t in st_) for st_ in past]
    # B: yalniz bellekten okuma (attention A'yi goremez) — asil hedef
    out_b = model(xb, labels=yb, past_key_values=past, use_cache=True)
    assert torch.isfinite(out_b.loss), f"NaN {TAG} step {step}"
    out_b.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sch.step()
    if step % 100 == 0:
        print(f"[{TAG}] step {step}/{STEPS} K={K} lossA {loss_a.item():.3f} "
              f"lossB(cross-chunk) {out_b.loss.item():.3f}", flush=True)

if step < STEPS:
    torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(), step=step,
                    rng=random.getstate(), nprng=np.random.get_state(),
                    trng=torch.get_rng_state()), CKPT)
    print(f"[{TAG}] CKPT {step}/{STEPS} — butce bitti, tekrar calistir (resume eder)")
    sys.exit(0)

# [FIX] Egitim TAMAMLANDIYSA da checkpointi kaydet — Gorev F (matched_probe)
# bu dosyayi kullanir; onceden yalniz "butce doldu" halinde kaydediliyordu.
torch.save(dict(m=model.state_dict(), o=opt.state_dict(), s=sch.state_dict(),
                step=step, rng=random.getstate(), nprng=np.random.get_state(),
                trng=torch.get_rng_state()), CKPT)
print(f"[{TAG}] egitim tamam -> checkpoint kaydedildi: {CKPT}")

# [PLATO KORUMASI] cross-chunk gorevi ogrenilmediyse kiyas gecersiz
with torch.no_grad():
    a, al, fills, b, bl = carry_example(2)
    xa = torch.tensor([a]*8, device=DEV); ya = torch.tensor([al]*8, device=DEV)
    o = model(xa, labels=ya, use_cache=True); past = o.past_key_values
    for j in range(2):
        xf = torch.tensor([fills[j]]*8, device=DEV)
        past = model(xf, past_key_values=past, use_cache=True).past_key_values
    xb = torch.tensor([b]*8, device=DEV); yb = torch.tensor([bl]*8, device=DEV)
    _l = model(xb, labels=yb, past_key_values=past, use_cache=True).loss.item()
print(f"[{TAG}] egitim-sonu cross-chunk dogrulama loss: {_l:.3f} (sans ~3.40)")

# ---------------- EVAL: §17 ile BIREBIR AYNI omur sondasi ----------------
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
    for s in range(0, ids.size(1), CTX):
        out = model(ids[:, s:s + CTX], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return int(logits.argmax().item() == v), Fn.log_softmax(logits, -1)[v].item()

model.eval(); rows = []; res = {}
for g in GAPS:
    ok = 0; lp = 0.0; te = time.time()
    for _ in range(TRIALS):
        o, l = probe(g); ok += o; lp += l
    res[g] = (round(100.0 * ok / TRIALS, 1), round(lp / TRIALS, 2))
    rows.append(dict(mode=MODE, seed=SEED, carry_max=CARRY_MAX, gap=g,
                     acc=res[g][0], logprob=res[g][1], trials=TRIALS))
    print(f"[{TAG}] gap {g:>6}: acc {res[g][0]:5.1f}%  logprob {res[g][1]:.2f} "
          f"({time.time()-te:.0f}s)", flush=True)

print(f"[{TAG}] FINAL acc%/logprob (sans %3.3): {res}")
with open(f"{CKDIR}/carry_results.txt", "a") as f:
    f.write(f"{TAG} carry_max={CARRY_MAX} {res}\n")
with open(f"{CKDIR}/{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[{TAG}] yazildi: {CKDIR}/carry_results.txt + {TAG}.csv")
