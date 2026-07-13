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

"""[Faz 4.5] Kisa-baglam benchmark: graft'li model vs orijinal (MMLU + HellaSwag).

Strateji dokumani (LLM_GRAFTING_STRATEGY.md §4.2) kriteri: graft'li model,
klasik benchmark'larda orijinalin -%5 bandindan fazla kaybetmemeli.

lm-eval-harness'in HFModel sarmalayicisi yerine modeli DOGRUDAN veririz
(graft'li nn.Module HF-uyumlu oldugu icin HFLM(pretrained=model) calisir).
Zaman butcesi icin --limit ile ornek sayisi kisilabilir (ilk gecis 200 yeter;
tam kosum icin --limit 0).

Kullanim (Colab, Stage 2 sonrasi — model bellekte grafted + agirliklar yuklu):
    pip install lm-eval
    python review_scripts/graft_benchmark_eval.py            # yardim
Ya da notebook icinden:
    from review_scripts.graft_benchmark_eval import run_benchmarks
    df = run_benchmarks(model, tok, limit=200)               # graft'li
    set_graft_mode(model, 'teacher'); df0 = run_benchmarks(model, tok, limit=200)  # orijinal
"""

import os, sys
if '__file__' in globals():   # script olarak; notebook hucresine yapistirilirsa atlanir
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS = ["hellaswag", "mmlu"]          # ilk gecis; istege: arc_easy, winogrande
CRITERION = 0.05                        # -%5 bandi (strateji dok. §4.2)


def run_benchmarks(model, tokenizer, tasks=None, limit=200, batch_size=4):
    """Graft'li (ya da teacher moddaki) HF-uyumlu modeli lm-eval ile kosar.
    limit: gorev basina ornek sayisi (0/None = tam kosum, cok daha yavas)."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    res = simple_evaluate(model=lm, tasks=tasks or TASKS,
                          limit=(None if not limit else limit))
    out = {}
    for task, metrics in res["results"].items():
        for k, v in metrics.items():
            if k.startswith(("acc,", "acc_norm,")) and isinstance(v, float):
                out[f"{task}/{k.split(',')[0]}"] = v
    return out


def compare(orig: dict, grafted: dict):
    """-%5 kriterini uygular; her metrik icin hukum basar."""
    print(f"{'metrik':<30} {'orijinal':>9} {'graft':>9} {'oran':>7}  hukum")
    print("-" * 66)
    verdicts = []
    for k in sorted(orig):
        if k not in grafted:
            continue
        o, g = orig[k], grafted[k]
        ratio = g / o if o else float("nan")
        ok = g >= o * (1 - CRITERION)
        verdicts.append(ok)
        print(f"{k:<30} {o:>9.4f} {g:>9.4f} {ratio:>6.3f}x  "
              f"{'GECTI' if ok else 'KALDI (-%5 bandi disinda)'}")
    if verdicts:
        n_ok = sum(verdicts)
        print(f"\nToplam: {n_ok}/{len(verdicts)} metrik kriter icinde -> "
              f"{'K3-benzeri KISA-BAGLAM KORUNDU' if all(verdicts) else 'kisa-baglam kaybi var; hibrit orani / token butcesi gozden gecirilmeli'}")
    return verdicts


if __name__ == "__main__":
    print(__doc__)
    print("Bu script notebook icinden import edilerek kullanilir "
          "(model + tokenizer bellekte olmali).")
