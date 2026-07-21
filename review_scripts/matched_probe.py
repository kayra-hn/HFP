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

"""[GOREV F] Eslesmis sonda: egitim/eval uyumsuzlugunun testi (harness duzeltmesi).

NEDEN (RESULTS §19 celiskisi):
  Gorev E'de egitim cross-chunk gorevini OGRENDI (lossB 3.45 -> 1.51-2.23,
  K=14 ~3600 token tasima) ama ayni mesafede EVAL SANSTA kaldi. Ogrenilmis
  beceri degerlendirmede gorunmuyorsa, ikisi ayni gorevi olcmuyordur.
  Tespit edilen farklar (§19):
    (1) YAZMA baglami: egitimde hedef yogun bir chunk'in SONUNA (sinira hizali,
        CTX-2/CTX-1) yazilir; eski evalde bombos akisin ILK iki tokeni.
    (2) SORGU baglami: egitimde sorgu yogun chunk'in sonunda (yerel ipuclariyla);
        eski evalde ciplak key tokeni.
    (3) Distraktor istatistikleri iki yolda farkli.

BU SCRIPT: sondayi EGITIM dagilimindan uretir; tek degisken MESAFE kalir.
  A chunk (dense, hedef sonda) -> K dolgu chunk -> B chunk (dense, sorgu sonda)
  Yani carry_curriculum'un egitim ornegiyle BIREBIR ayni sahne, sadece K degisir.
  Ayrica AYNI kosuda ESKI (eslesmemis) sondayi da olcer -> fark dogrudan gorulur.

ON-KAYITLI KRITERLER (kosudan ONCE yazildi):
  Model: carryv1_* checkpointleri (Gorev E; egitim TEKRARLANMAZ).
  - UYUMSUZLUK DOGRULANDI: eslesmis sondada K<=16 (egitilen menzil) icin
    seed-ort acc >= %15 iken eslesmemis sonda ayni K'da <= %8 kalirsa.
    -> §17-§19 cokusu buyuk olcude EVAL ARTEFAKTI; durust iddia:
       "egitildigi mesafede, egitildigi bicimde soruldugunda hatirlar".
  - MIMARI SUPHESI: eslesmis sondada da K=8/16'da acc < %8 ise, dort eleme
    (yasa/kapasite/mufredat/harness) tamamlanir; bas suphe state boyutu
    (bulk_dim=32) ve okuma yolu olur.
  - EKSTRAPOLASYON: K=32/64 (egitilen tavanin 2-4 kati) ayrica raporlanir;
    dusus varsa "egitilen menzil kadar omur" ifadesi sayilarla sinirlandirilir.

KULLANIM
  python review_scripts/matched_probe.py <mode> <seed>
  Env: MP_KS="0,1,2,4,8,16,32,64"  MP_TRIALS=30  MP_CTX=256  MP_P=6
       MP_DIST_EVERY=64  HFP_CKPT_DIR=checkpoints
Cikti: {CKDIR}/matched_results.txt + {CKDIR}/matchedv1_{mode}_s{seed}.csv
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
KS = [int(k) for k in os.environ.get("MP_KS", "0,1,2,4,8,16,32,64").split(",")]
TRIALS = int(os.environ.get("MP_TRIALS", "30"))
CTX = int(os.environ.get("MP_CTX", "256"))
P = int(os.environ.get("MP_P", "6"))
DIST_EVERY = int(os.environ.get("MP_DIST_EVERY", "64"))

KLO, KHI, VLO, VHI, FHI = 100, 130, 130, 160, 100     # sans = 1/30
WIN, ANS = 8, 0
TAG = f"matchedv1_{MODE}_s{SEED}"
SRC = f"{CKDIR}/carryv1_{MODE}_s{SEED}.pt"            # Gorev E checkpointi

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
cfg = HFPConfig(vocab_size=VHI + 4, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=256, bulk_dim=32,
                short_len=8, max_position_embeddings=CTX + 8, local_window=WIN,
                decay_mode=MODE, rec_block=32, write_rule="additive",
                key_feature_map="dpfp", pe_period=CTX)
model = HFPForCausalLM(cfg)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEV)

assert os.path.exists(SRC), (
    f"{SRC} yok. Once Gorev E kosulmali (carry_curriculum.py) — bu script "
    "egitim YAPMAZ, yalnizca sondayi degistirir.")
st = torch.load(SRC, weights_only=False, map_location=DEV)
sd = st["m"] if isinstance(st, dict) and "m" in st else st
model.load_state_dict(sd)
model.eval()
print(f"[{TAG}] yuklendi: {SRC}  (Gorev E modeli, egitim tekrarlanmadi)")

# ---- EGITIM dagilimindan uretilen parcalar (carry_curriculum ile ayni) ----
def dense_chunk(reserve_tail=0):
    toks = [random.randint(1, FHI - 1) for _ in range(CTX)]
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
    return toks

def filler_chunk(forbid):
    toks = []
    while len(toks) < CTX:
        n = min(DIST_EVERY, CTX - len(toks))
        toks.extend(random.randint(1, FHI - 1) for _ in range(max(0, n - 2)))
        if n > 2:
            k = random.choice([k for k in range(KLO, KHI) if k != forbid])
            toks.extend([k, random.randint(VLO, VHI - 1)])
    return toks[:CTX]

@torch.no_grad()
def probe_matched(K):
    """EGITIMLE AYNI SAHNE: A(dense, hedef sonda) -> K dolgu -> B(dense, sorgu sonda)."""
    tgt_k = random.randint(KLO, KHI - 1)
    v = random.randint(VLO, VHI - 1)
    a = dense_chunk(reserve_tail=4); a[CTX - 2] = tgt_k; a[CTX - 1] = v
    b = dense_chunk(reserve_tail=4); b[CTX - 2] = tgt_k; b[CTX - 1] = ANS
    past = model(torch.tensor([a], device=DEV), use_cache=True).past_key_values
    for _ in range(K):
        xf = torch.tensor([filler_chunk(tgt_k)], device=DEV)
        past = model(xf, past_key_values=past, use_cache=True).past_key_values
    out = model(torch.tensor([b[:-1]], device=DEV), past_key_values=past, use_cache=True)
    lg = out.logits[0, -1]
    return int(lg.argmax().item() == v), Fn.log_softmax(lg, -1)[v].item()

@torch.no_grad()
def probe_unmatched(K):
    """ESKI sonda (§17/§19): ciplak hedef en basta, ciplak sorgu sonda."""
    tgt_k = random.randint(KLO, KHI - 1)
    v = random.randint(VLO, VHI - 1)
    seq = [tgt_k, v]
    for _ in range(K):
        seq += filler_chunk(tgt_k)
    seq += [tgt_k]
    ids = torch.tensor([seq], device=DEV)
    past = None; lg = None
    for s in range(0, ids.size(1), CTX):
        out = model(ids[:, s:s + CTX], past_key_values=past, use_cache=True)
        past = out.past_key_values; lg = out.logits[0, -1]
    return int(lg.argmax().item() == v), Fn.log_softmax(lg, -1)[v].item()

rows = []
print(f"\n{'K':>4} {'~token':>8} | {'ESLESMIS acc':>13} {'logp':>7} | {'ESKI acc':>9} {'logp':>7}")
print("-" * 62)
for K in KS:
    t0 = time.time()
    m_ok = m_lp = u_ok = u_lp = 0.0
    for _ in range(TRIALS):
        o, l = probe_matched(K);   m_ok += o; m_lp += l
        o, l = probe_unmatched(K); u_ok += o; u_lp += l
    m_acc = 100.0 * m_ok / TRIALS; u_acc = 100.0 * u_ok / TRIALS
    rows.append(dict(mode=MODE, seed=SEED, K=K, tokens=K * CTX,
                     matched_acc=round(m_acc, 1), matched_logp=round(m_lp / TRIALS, 2),
                     unmatched_acc=round(u_acc, 1), unmatched_logp=round(u_lp / TRIALS, 2),
                     trials=TRIALS))
    print(f"{K:>4} {K*CTX:>8} | {m_acc:>12.1f}% {m_lp/TRIALS:>7.2f} | "
          f"{u_acc:>8.1f}% {u_lp/TRIALS:>7.2f}   ({time.time()-t0:.0f}s)", flush=True)

print(f"\n[{TAG}] sans %3.3 | ESLESMIS = egitim dagilimindan sonda, ESKI = §17/§19 sondasi")
with open(f"{CKDIR}/matched_results.txt", "a") as f:
    f.write(f"{TAG} " + str({r['K']: (r['matched_acc'], r['unmatched_acc']) for r in rows}) + "\n")
with open(f"{CKDIR}/{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[{TAG}] yazildi: {CKDIR}/matched_results.txt + {TAG}.csv")
