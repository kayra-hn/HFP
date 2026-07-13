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

"""[K3-c] Alpha-gate / graft parametre analizi — GPU GEREKMEZ, checkpoint yeter.

Grafting Stage 1/2 checkpoint'inden (hfp_graft_*.pt) okur:
  1. ALPHA DAGILIMI (kafa basina, katman basina): 0 -> additive (arsiv),
     1 -> delta (calisma-bellegi). Hipotez (SONRAKI_ADIMLAR_PLANI.md K3-c):
     cogunluk additive'e, azinlik delta'ya kutuplasir -> "arsiv kafalari vs
     calisma-bellegi kafalari". Tek kutup da bulgudur.
  2. BETA (yazim siddeti) bias'lari.
  3. DECAY spektrumu (exp lam dagilimi) ve ETA spektrumu (cubic plato olcekleri
     t* ~ 1/sqrt(2*eta)) — egitim init'ten ne kadar uzaklasmis?
  4. OUT_GAIN — katman basina okuma kazanci (0'a coken katman = bellek yolu
     kullanilmiyor demektir; onemli teshis).

Kullanim (Colab'de Stage 1 sonrasi ya da lokalde):
    python review_scripts/alpha_gate_analysis.py /path/to/hfp_graft_final.pt
"""

import os, sys, math
if '__file__' in globals():   # script olarak; notebook hucresine yapistirilirsa atlanir
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def hist_ascii(vals, lo=0.0, hi=1.0, bins=10, width=40):
    counts = [0] * bins
    for v in vals:
        i = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
        counts[i] += 1
    mx = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        a, b = lo + i * (hi - lo) / bins, lo + (i + 1) * (hi - lo) / bins
        lines.append(f"  {a:4.2f}-{b:4.2f} | {'#' * int(width * c / mx):<{width}} {c}")
    return "\n".join(lines)


def main(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: {path}  ({len(sd)} tensor)\n")

    # katman indekslerini anahtar adlarindan cikar: model.layers.<i>.self_attn.alpha_logit
    def layer_of(k):
        for part in k.split('.'):
            if part.isdigit():
                return int(part)
        return -1

    alpha_keys = sorted([k for k in sd if k.endswith('alpha_logit')], key=layer_of)
    gain_keys  = sorted([k for k in sd if k.endswith('out_gain')], key=layer_of)
    decay_keys = sorted([k for k in sd if k.endswith('.decay')], key=layer_of)
    eta_keys   = sorted([k for k in sd if k.endswith('log_eta')], key=layer_of)
    beta_keys  = sorted([k for k in sd if 'beta_gate.bias' in k], key=layer_of)

    # ---- 1. ALPHA ----
    if alpha_keys:
        print("=" * 66)
        print("  1. ALPHA-GATE (0=additive/arsiv  1=delta/calisma-bellegi)")
        print("=" * 66)
        all_alphas = []
        for k in alpha_keys:
            a = torch.sigmoid(sd[k]).flatten()
            all_alphas += a.tolist()
            marks = ''.join('D' if v > 0.7 else ('a' if v < 0.3 else '~') for v in a)
            print(f"  katman {layer_of(k):>2}: ort {a.mean():.3f}  min {a.min():.3f} "
                  f"max {a.max():.3f}  kafalar [{marks}]  (a<0.3, ~ara, D>0.7)")
        n = len(all_alphas)
        n_add = sum(v < 0.3 for v in all_alphas)
        n_del = sum(v > 0.7 for v in all_alphas)
        print(f"\n  TOPLAM {n} kafa: additive {n_add} ({100*n_add/n:.0f}%), "
              f"delta {n_del} ({100*n_del/n:.0f}%), ara {n-n_add-n_del}")
        print(hist_ascii(all_alphas))
        init = torch.sigmoid(torch.tensor(-2.0)).item()
        moved = sum(abs(v - init) > 0.05 for v in all_alphas)
        print(f"  init {init:.3f}'ten >0.05 uzaklasan: {moved}/{n} "
              f"({'OGRENILIYOR' if moved > n * 0.2 else 'henuz init civari — daha fazla adim gerek'})")

    # ---- 2. OUT_GAIN ----
    if gain_keys:
        print("\n" + "=" * 66)
        print("  2. OUT_GAIN (katman okuma kazanci; ~0 = bellek yolu kullanilmiyor)")
        print("=" * 66)
        for k in gain_keys:
            g = sd[k].flatten()
            flag = "  <-- SONUK (bellek yok sayiliyor!)" if g.abs().mean() < 0.1 else ""
            print(f"  katman {layer_of(k):>2}: ort {g.mean():+.3f}  |g| ort {g.abs().mean():.3f}{flag}")

    # ---- 3. DECAY / ETA ----
    if decay_keys:
        print("\n" + "=" * 66)
        print("  3. EXP DECAY spektrumu (lam; init 0.90-0.999 lineer)")
        print("=" * 66)
        for k in decay_keys[:3] + (['...'] if len(decay_keys) > 3 else []):
            if k == '...': print('  ...'); break
            lam = torch.sigmoid(sd[k]).flatten()
            print(f"  katman {layer_of(k):>2}: lam medyan {lam.median():.4f}  "
                  f"[{lam.min():.4f}, {lam.max():.4f}]  ufuk medyan ~{1/(1-lam.median()):.0f} token")
    if eta_keys:
        print("\n  CUBIC eta -> plato gecis olcegi t* ~ 1/sqrt(2*eta):")
        for k in eta_keys[:3]:
            eta = torch.exp(sd[k]).flatten()
            t = (1.0 / torch.sqrt(2 * eta))
            print(f"  katman {layer_of(k):>2}: t* medyan {t.median():.0f} token  "
                  f"[{t.min():.0f}, {t.max():.0f}]")

    # ---- 4. BETA ----
    if beta_keys:
        print("\n" + "=" * 66)
        print("  4. BETA bias (yazim siddeti; init 1.0 -> sigmoid ~0.73)")
        print("=" * 66)
        for k in beta_keys:
            print(f"  katman {layer_of(k):>2}: bias {sd[k].item():+.3f} "
                  f"(sigmoid {torch.sigmoid(sd[k]).item():.3f})")

    print("\nBitti. Yorum icin: SONRAKI_ADIMLAR_PLANI.md K3-c.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("kullanim: python review_scripts/alpha_gate_analysis.py <hfp_graft_*.pt>")
        sys.exit(1)
    main(sys.argv[1])
