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

"""[§30b CPU] Hafiza organi sondasi — GPU'suz, egitimsiz, dusuk maliyetli.

NEDEN: §30/§31 hukmu icin GPU gerekmiyor. Kritik olcum sadece CIKARIM ve
kisa dizilerle yapiliyor (chunk 128, gap 0-2 -> deneme basina ~300-500 token).
Bu script onu tek basina, CPU'da kosturur.

NE OLCER (RESULTS §30b ile ayni protokol):
  chunk A (dolgu + olgu SONDA) -> K dolgu chunk -> chunk B (dolgu + sorgu SONDA)
  Her chunk'ta TAZE KV-cache; HFP streaming durumu tasinir.
  => mevcut chunk'tan eskisi YALNIZCA O(1) durumundan gelebilir.
  Egitim protokolu de boyleydi (S2 cross-chunk recall, SEQ=128), yani sonda
  EGITIM DAGILIMIYLA ESLESIK — §19/§26a'daki harness dersi uygulanmis halde.

KONTROL: ayni sonda 'teacher' modunda da kosulur (tam attention, tek parca).
  Ust sinir dusukse sorun hafizada degil, base modeldedir -> ayirt edilir.

KULLANIM
  python review_scripts/memory_probe_cpu.py <checkpoint.pt> [model_dizini]
  Env: MP_TRIALS=8  MP_GAPS=0,1,2  MP_LAYERS=3,7,11,15,19,23  MP_DEVICE=cpu

Ornek:
  python review_scripts/memory_probe_cpu.py hfp_graft_exp_g6W_s0_final.pt ./qwen_model

Cikti: gap basina birebir-geri-getirme yuzdesi + ust sinir + on-kayitli hukum.
"""
import os, sys, glob, random, time

# [FIX] 'python review_scripts/x.py' calistirildiginda Python sys.path'e SCRIPT'in
# dizinini ekler, repo kokunu degil -> 'No module named hfp'. Kokü kendimiz ekliyoruz.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from hfp.models.grafting import (GraftConfig, graft_llama, set_graft_mode,
                                 enable_streaming, reset_streaming, HFPGraftAttention)

CKPT = sys.argv[1] if len(sys.argv) > 1 else None
assert CKPT and os.path.exists(CKPT), (
    "Kullanim: python review_scripts/memory_probe_cpu.py <checkpoint.pt> [model_dizini]")
MODEL_DIR = sys.argv[2] if len(sys.argv) > 2 else None

DEV = os.environ.get("MP_DEVICE", "cpu")
TRIALS = int(os.environ.get("MP_TRIALS", "8"))
GAPS = [int(g) for g in os.environ.get("MP_GAPS", "0,1,2").split(",")]
LAYERS = [int(x) for x in os.environ.get("MP_LAYERS", "3,7,11,15,19,23").split(",")]
SEQ = 128

# ---------- model ----------
if MODEL_DIR is None:                      # yerelde Qwen ara
    for root in (".", os.path.expanduser("~"), "/kaggle/input", "/content"):
        for cfg in glob.glob(f"{root}/**/config.json", recursive=True):
            d = os.path.dirname(cfg)
            if glob.glob(f"{d}/*.safetensors"):
                MODEL_DIR = d
                break
        if MODEL_DIR:
            break
assert MODEL_DIR, ("Qwen2.5-1.5B klasoru bulunamadi. Ikinci argumanla verin veya\n"
                   "  huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./qwen_model")
print(f"model     : {MODEL_DIR}")
print(f"checkpoint: {CKPT}")
print(f"cihaz     : {DEV} | deneme {TRIALS} | gap {GAPS}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
graft_llama(model, GraftConfig(decay_mode="exp", write_rule="hybrid",
                               key_feature_map="dpfp", rec_block=16), layers=LAYERS)
for m in model.modules():
    if isinstance(m, HFPGraftAttention):
        m.out_gain.data.fill_(0.1)

sd = torch.load(CKPT, map_location=DEV)
own = dict(model.state_dict())
matched = [k for k in sd if k in own and own[k].shape == sd[k].shape]
assert len(matched) == len(sd), (
    f"Checkpoint uyusmuyor: {len(matched)}/{len(sd)} tensor eslesti. "
    "Katman listesi (MP_LAYERS) veya write_rule farkli olabilir.")
model.load_state_dict(sd, strict=False)
_og = torch.cat([m.out_gain.detach().flatten() for m in model.modules()
                 if isinstance(m, HFPGraftAttention)])
print(f"yukleme   : {len(matched)}/{len(sd)} tensor | out_gain ort {_og.mean():.3f} "
      f"std {_og.std():.3f} (init 0.1'den sapma = egitim izi)", flush=True)

# ---------- sonda ----------
FILLER = " The routine check completed and nothing of note happened during this step."
FILL_IDS = tok(FILLER, add_special_tokens=False).input_ids
VALS = ["tunis", "oslo", "lima", "ravenwood", "stonepark", "elmgate"]


def pad(n):
    b = []
    while len(b) < n:
        b.extend(FILL_IDS)
    return b[:n]


@torch.no_grad()
def probe(value, gap, upper_bound=False, gen=6):
    """upper_bound=False -> C kosulu: her chunk taze cache, bilgi yalniz O(1)'de.
       upper_bound=True  -> tam attention, tek parca (base modelin tavani)."""
    fact = tok(f" The license server is {value}. ", add_special_tokens=False).input_ids
    query = tok(" The license server is", add_special_tokens=False).input_ids
    a = pad(SEQ - len(fact)) + fact
    b = pad(SEQ - len(query)) + query

    if upper_bound:
        set_graft_mode(model, "teacher"); enable_streaming(model, False)
        ids = a + pad(SEQ) * gap + b
        cache = DynamicCache()
        out = model(torch.tensor([ids], device=DEV), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    else:
        set_graft_mode(model, "student")
        enable_streaming(model, True); reset_streaming(model)
        model(torch.tensor([a], device=DEV), past_key_values=DynamicCache(), use_cache=True)
        for _ in range(gap):
            model(torch.tensor([pad(SEQ)], device=DEV),
                  past_key_values=DynamicCache(), use_cache=True)
        cache = DynamicCache()
        out = model(torch.tensor([b], device=DEV), past_key_values=cache, use_cache=True)
        cache = out.past_key_values

    g, last = [], out.logits[:, -1:].argmax(-1)
    for _ in range(gen):
        g.append(last.item())
        out = model(last, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last = out.logits[:, -1:].argmax(-1)
    if not upper_bound:
        enable_streaming(model, False)
    return tok.decode(g).strip()


print(f"\n{'gap':>4} {'~token':>7} {'C: O(1) hafiza':>16} {'A: ust sinir':>14}   ornek (C)")
print("-" * 74)
res = {}
for gp in GAPS:
    t0 = time.time()
    okC = okA = 0
    ex = ""
    for t in range(TRIALS):
        r = random.Random(1000 + t)
        v = r.choice(VALS)
        ansC = probe(v, gp, upper_bound=False)
        okC += (v.lower() in ansC.lower())
        if t == 0:
            ex = f'"{ansC}" (hedef {v})'
        if gp == GAPS[0]:                       # ust sinir yalniz ilk gap'te (pahali degil ama gereksiz)
            okA += (v.lower() in probe(v, gp, upper_bound=True).lower())
    accC = 100.0 * okC / TRIALS
    accA = 100.0 * okA / TRIALS if gp == GAPS[0] else float("nan")
    res[gp] = accC
    print(f"{gp:>4} {gp*SEQ+SEQ:>7} {accC:>15.0f}% "
          f"{('%.0f%%' % accA) if gp == GAPS[0] else '   —':>14}   {ex}  ({time.time()-t0:.0f}s)",
          flush=True)
    if gp == GAPS[0]:
        UPPER = accA

print(f"\n=== ON-KAYITLI HUKUM (RESULTS §31) ===")
if UPPER < 50:
    print(f"UST SINIR DUSUK ({UPPER:.0f}%) -> sonda base model icin bile zor; hafiza hukmu VERILEMEZ.")
else:
    g0 = res[GAPS[0]]
    if g0 >= 60:
        print(f"YAZMA YOLU OGRENDI: gap {GAPS[0]}'da {g0:.0f}% (>=60).")
        print("  -> §30c teshisi DOGRULANDI ve mudahale ISE YARADI. O(1) durumu artik")
        print("     egitilen menzilde gercek bir depo. Sonraki: menzil genisletme.")
    elif g0 >= 30:
        print(f"KISMI: gap {GAPS[0]}'da {g0:.0f}% (30-60). Sinyal var ama zayif;")
        print("  -> yon dogru, kapasite/girisim kaldiraclari (delta yazim, buyuk state) denenir.")
    else:
        print(f"ETKI YOK: gap {GAPS[0]}'da {g0:.0f}% (<30).")
        print("  -> Gradyan tesisati GEREKLIYDI ama YETERLI DEGIL (kucuk-olcek §21 ile tutarli).")
        print("     Sirada: delta yazim (§32) ve/veya daha buyuk state; ikisi de sabit bellegi bozmaz.")
