# Plateau Retention: A State-Magnitude-Dependent Decay for Extreme-Long-Range Recall in O(1)-Memory Sequence Models

**Kayrahan Yılmaz** · Technical note (small-scale study) · July 2026
Code & full logs: <https://github.com/kayra-hn/HFP> · License: AGPL-3.0

> **Scope & honesty note.** This is a small-scale, controlled study (≤1M
> parameters, synthetic multi-query recall, single consumer GPU). It reports a
> **pre-registered, statistically significant** effect at this scale. It does
> **not** claim a language-model benchmark result, and the physics that
> *motivated* the mechanism is inspiration only — nothing here validates it.
> The result is an ML claim, established by the experiment below.

## Summary

Constant-state ("O(1)-memory") recurrent sequence models — RetNet, GLA, Mamba,
RWKV, DeltaNet — must forget, and almost all of them forget via a decay that is
**geometric** (exponential in gap) with an input-dependent gate. Under geometric
decay a fact seen once decays as `λ^gap`, i.e. it vanishes exponentially with
distance. This note studies a different, less-explored axis: a decay whose rate
depends on the **accumulated state magnitude** rather than on the input, derived
as the exact one-step solution of a cubic relaxation `dθ/dτ = −η·θ³`:

```
λ_t = 1 / sqrt(1 + 2·η·z_{t-1}²)      # z = per-channel key accumulator
```

An empty channel (`z→0`) barely decays (a **plateau**); a full channel decays in
proportion to its magnitude (self-limiting). The retention envelope is
**plateau-then-power-law** instead of a geometric cliff.

**Finding (pre-registered).** On a synthetic dense-recall task, trained at
context 160 and evaluated at 1280 (train-short / infer-long), this cubic decay —
combined with a DPFP key feature map — beats the exponential-decay baseline
**specifically at the longest gaps (256+ tokens)**: mean accuracy **33.8% vs
19.8%** (≈1.7×) across 8 seeds, paired **t(7)=2.47, p≈0.043**, satisfying the
pre-registered ">2 SE" criterion. cubic wins in 6/8 seeds. The advantage is a
**trade-off**, not a uniform improvement: cubic is *worse* at every shorter gap
(it flattens the whole curve) and only overtakes exp once exp has decayed to
near-chance.

## 1. Setup

**Architecture (HFP).** Windowed local attention + a per-layer recurrent
linear-attention memory (`M ∈ ℝ^{D×H}`, `z ∈ ℝ^D`) whose inference-time state is
constant in context length. Long-range information must flow through the
recurrent memory; the windowed attention cannot reach beyond its receptive
field, so recall at gaps larger than the window is a pure test of the memory.

**Selectable retention law.**
- `exp` — geometric decay `λ = σ(decay)`, learned per channel, multi-timescale
  initialization (0.90–0.999). This is the standard efficient-recurrent baseline.
- `cubic_flux` — the state-magnitude-dependent decay above, `λ_t =
  1/√(1+2η z_{t-1}²)`, `η` a learned per-channel "flux". Implemented as an exact
  two-pass form (a sequential z-scan for the per-token decay, then a
  GLA-style chunkwise-parallel M-recurrence); the two-pass form is bit-consistent
  with the naive sequential recurrence.

**Capacity axis.** DPFP (Deterministic Parameter-Free Projection; Schlag et al.,
2021) expands the key dimension 4×, delaying rank collapse and keeping per-channel
usage sparse.

**Task.** Dense multi-query synthetic recall: a sequence contains P key–value
writes and P queries interleaved; the label is the value at each query position.
Chance = 3.3%. Supervision is dense (many labels per sequence) because
single-query supervision fails to learn at long context (an optimization
artifact, not a memory limit). Accuracy is bucketed by gap between write and
query. Train at context 160, evaluate unchanged at 640 and 1280 (train-short /
infer-long, enabled by the O(1) state). P=8 (sparse regime).

## 2. Main result — mean bucket profile (8 seeds, ctx 1280, dpfp)

| gap bucket | exp/dpfp | cubic/dpfp |
|---|---|---|
| <48       | **84.9** | 62.9 |
| 48–127    | **66.6** | 52.9 |
| 128–255   | **51.5** | 44.3 |
| **256+**  | 19.8     | **33.8** |

The signature is exactly the mechanism's prediction: **exp starts high and falls
off a cliff** (84.9 → 19.8); **cubic starts lower and stays flat** (62.9 → 33.8).
The curves cross between the 128–255 and 256+ buckets. cubic wins **only** at the
extreme tail — where a plateau matters and where exp has decayed to near-chance —
and pays for it everywhere shorter.

### Per-seed 256+ (the pre-registered decision bucket)

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| exp/dpfp   | 29.2 | 20.4 | 12.4 | 11.0 | 13.2 | 44.5 | 12.6 | 15.4 | **19.8** |
| cubic/dpfp | 52.1 | 33.1 | 16.1 | 46.6 | 30.7 | 33.6 | 43.3 | 14.6 | **33.8** |
| Δ (c−e)    | +22.9 | +12.7 | +3.7 | +35.6 | +17.5 | −10.9 | +30.7 | −0.8 | **+14.0** |

Paired difference: mean **+14.0**, SD 16.0, SE 5.6 → **t(7) = 2.47, p ≈ 0.043**
(two-tailed). Pre-registered criterion (Δ > 2·SE) **met**. cubic wins 6/8 seeds;
the two losses are one exp-lucky seed (s5) and one near-tie (s7).

Same-direction results were also observed on CPU (a separate 5-seed run) and are
robust to the compute backend in sign, though the magnitude is seed-sensitive.

## 3. Mechanism

Two ingredients are both necessary:

1. **The plateau.** Under geometric decay, a sparsely-written channel loses a
   stored fact as `λ^gap` — exponentially. Under cubic decay, an unused channel
   (`z` small) barely decays, so the stored fact leaves a **polynomially-decaying
   trace** that remains retrievable at gaps where the geometric envelope is
   already ≈0. This is why the advantage appears *only* at 256+.

2. **DPFP is required.** With the standard `elu` feature map, cubic's advantage
   at 256+ is small (~7% in our runs) — because in a shared, dense key space the
   accumulator `z` grows and the plateau is lost. DPFP keeps per-channel usage
   sparse, so `z` stays small on unused channels and the plateau survives. The
   effect is a **cubic × DPFP interaction**, not cubic alone.

A secondary, likely-robust property: cubic is **self-limiting** — over a 4000-token
adversarial stream, `max|M| ≈ 14` for cubic vs ≈254 for exp — so it does not blow
up over arbitrarily long streams, which matters for constant-memory streaming.

## 4. Limitations & honest ledger

- **It is a trade-off, not superiority.** cubic is worse at every gap < 256
  (mean <48: 62.9 vs 84.9). The claim is narrow: *if you need recall at gaps
  where an exponential-decay memory has decayed to near-chance, under constant
  memory, cubic's plateau helps.* It is not a better general-purpose memory.
- **Small scale, synthetic.** ≤1M parameters, one synthetic task family. No
  language-model benchmark is claimed; the LM-scale question is open.
- **Magnitude is seed-fragile.** Per-seed 256+ ranges 14.6–52.1; the effect is
  significant in aggregate but noisy per run. A targeted stabilization attempt
  (slower learning rate + warmup on the retention parameters) did **not** improve
  robustness — reported as a negative result.
- **Speed.** cubic's per-token decay is a genuinely nonlinear recurrence
  (`λ_t` depends on `z_{t-1}`), so it cannot use the parallel/associative scan
  that the exponential family enjoys; it is sequential and slower. A TorchScript
  fusion of the scan removes Python overhead (exact, ~2–4× faster) but the O(L)
  sequential dependency remains.

## 5. Positioning (related work)

The efficient-recurrent / linear-attention family is crowded and well-resourced:
linear attention (Katharopoulos et al., 2020), RetNet (Sun et al., 2023), GLA
(Yang et al., 2023), Mamba / Mamba-2 (Gu & Dao, 2023; Dao & Gu, 2024), RWKV,
DeltaNet / Gated DeltaNet (Yang et al., 2024). **O(1) memory, DPFP, and the delta
write rule are all pre-existing** and better developed at scale elsewhere; this
work does not claim them. What differs here is the **decay axis**: these models
gate on the *input* (data-dependent decay); the mechanism studied here gates on
the *accumulated state magnitude* (a self-regulating feedback), which is the
less-explored corner. The contribution is a controlled, honest, multi-seed
measurement of that axis at small scale — a positive, pre-registered effect in one
specific regime — not a new state-of-the-art model.

## 6. Reproducibility

All scripts, seeds, and raw logs are in the repository. The decisive experiment:

```bash
# 8-seed matched comparison, 256+ decision bucket
for s in 0 1 2 3 4 5 6 7; do
  python review_scripts/cubic_longhorizon.py exp                dpfp 1e-3 $s
  python review_scripts/cubic_longhorizon.py cubic_flux_chunked dpfp 1e-3 $s
done
```

Regression / exactness tests: `smoke_test.py`, `review_scripts/verify_claims.py`
(chunk-consistency and exactness of the cubic discretization).

## Suggested figures

1. **Plateau vs cliff.** Accuracy vs gap bucket (mean ± SE over 8 seeds), two
   lines: exp/dpfp (high-then-cliff) and cubic/dpfp (flatter, wins at 256+).
   The crossover between 128–255 and 256+ is the visual headline.
2. **Per-seed 256+.** Paired dot/line plot of the 8 (exp, cubic) pairs, showing
   6/8 above the diagonal and the mean gap +14.0.
3. **(Optional) Ablation.** 256+ accuracy for cubic/elu vs cubic/dpfp vs exp/dpfp,
   showing DPFP is required for the plateau to pay off.
