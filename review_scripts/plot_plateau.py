# Hyper Flux Projection (HFP) — plateau-retention figures (self-contained)
# Copyright (C) 2026 Kayrahan Yılmaz — AGPL-3.0
#
# PLATEAU_RETENTION_NOTE.md icin figurler. Veriler 8-seed GPU kosusundan
# HARDCODE (Kaggle/dosyaya bagimli degil). Calistir: python review_scripts/plot_plateau.py
# Gereksinim: numpy, matplotlib. Cikti: assets/figures/fig_plateau.png, assets/figures/fig_paired256.png

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 8-seed GPU sonuclari (ctx 1280), kova bazinda [seed x bucket] ---
buckets = ["<48", "48-127", "128-255", "256+"]
exp = np.array([  # exp/dpfp
    [100.0, 85.4, 80.6, 29.2],
    [ 82.1, 84.2, 54.0, 20.4],
    [ 86.7, 51.7, 36.2, 12.4],
    [ 93.1, 64.8, 39.5, 11.0],
    [ 82.6, 56.5, 41.9, 13.2],
    [ 81.0, 88.7, 69.6, 44.5],
    [ 80.0, 58.7, 44.4, 12.6],
    [ 74.1, 42.4, 45.9, 15.4],
])
cub = np.array([  # cubic_flux_chunked/dpfp
    [ 65.2, 64.6, 70.8, 52.1],
    [ 56.4, 47.4, 39.7, 33.1],
    [ 86.7, 51.7, 39.7, 16.1],
    [ 48.3, 46.3, 44.4, 46.6],
    [ 82.6, 73.9, 44.6, 30.7],
    [ 42.9, 45.3, 40.5, 33.6],
    [ 43.3, 43.5, 33.3, 43.3],
    [ 77.8, 50.8, 41.0, 14.6],
])
n = exp.shape[0]
exp_m, exp_se = exp.mean(0), exp.std(0, ddof=1) / np.sqrt(n)
cub_m, cub_se = cub.mean(0), cub.std(0, ddof=1) / np.sqrt(n)

# paired t-test on 256+
d = cub[:, -1] - exp[:, -1]
t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
print(f"256+  exp {exp_m[-1]:.1f}  cubic {cub_m[-1]:.1f}  Δ {d.mean():+.1f}  t(7)={t:.2f}  wins {int((d>0).sum())}/{n}")

# --- Figure 1: plateau vs cliff ---
x = np.arange(len(buckets))
plt.figure(figsize=(6.4, 4.2))
plt.errorbar(x, exp_m, yerr=exp_se, marker="o", capsize=4, lw=2, label="exp + dpfp (geometric)")
plt.errorbar(x, cub_m, yerr=cub_se, marker="s", capsize=4, lw=2, label="cubic_flux + dpfp (plateau)")
plt.axhline(3.3, ls="--", c="gray", lw=1, label="chance (3.3%)")
plt.xticks(x, buckets)
plt.xlabel("write→query gap (tokens)")
plt.ylabel("recall accuracy (%)")
plt.title("Plateau vs cliff: recall by gap (train@160 → eval@1280, 8 seeds)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
import os; os.makedirs("assets/figures", exist_ok=True)
plt.savefig("assets/figures/fig_plateau.png", dpi=150)
print("yazildi: assets/figures/fig_plateau.png")

# --- Figure 2: paired 256+ (exp vs cubic) ---
plt.figure(figsize=(4.6, 4.6))
lim = max(exp[:, -1].max(), cub[:, -1].max()) + 8
plt.plot([0, lim], [0, lim], ls="--", c="gray", lw=1, label="eşit")
plt.scatter(exp[:, -1], cub[:, -1], s=60, zorder=3)
for i in range(n):
    plt.annotate(f"s{i}", (exp[i, -1], cub[i, -1]), textcoords="offset points", xytext=(5, 4), fontsize=8)
plt.xlim(0, lim); plt.ylim(0, lim)
plt.xlabel("exp + dpfp,  256+ accuracy (%)")
plt.ylabel("cubic + dpfp,  256+ accuracy (%)")
plt.title(f"256+ gap, paired by seed\ncubic wins {int((d>0).sum())}/{n},  Δ={d.mean():+.1f}, t={t:.2f}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("assets/figures/fig_paired256.png", dpi=150)
print("yazildi: assets/figures/fig_paired256.png")
