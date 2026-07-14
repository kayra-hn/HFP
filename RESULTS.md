# HFP — Experimental Results (v2.1)

> 🇹🇷 **Türkçe:** [Deney Sonuçları (Türkçe)](docs/tr/DENEY_SONUCLARI.md)

All experiments are small-scale (≤1M params, synthetic recall tasks, CPU),
multi-seed where stated, and fully reproducible with the scripts in
`review_scripts/`. Chance level is 3.3% throughout. These are architecture-level
findings, not language-model benchmarks; treat effect *patterns* as the result
and absolute numbers as scale-dependent.

## 1. Methodological finding: supervision density gates learnability

Single-query recall sequences (one supervised token per sequence) sit at the
`ln(vocab)` loss plateau and never learn, at any tested LR/curriculum, once the
context exceeds ~300 tokens — even though the same model learns the same task
easily with 8 queries per sequence. **All retention claims measured with
sparse-supervision tasks are optimization artifacts.** Every experiment below
uses multi-query ("dense") sequences (`dense_retention.py`).

## 2. Retention-law and write-rule comparison (3 seeds, ctx 160)

Mean accuracy over seeds {0,1,2}, 600 steps, lr 1e-3, gap buckets in tokens:

| configuration | 1-15 | 16-47 | 48-95 | 96+ |
|---|---|---|---|---|
| exp + additive (baseline) | 44.4 | 32.7 | 18.8 | 8.1 |
| exp + delta write | 52.1 | 31.1 | 16.4 | 6.6 |
| cubic_flux + additive | 31.3 | 23.0 | 11.8 | 4.7 |

- `cubic_flux` trails the exponential baseline in this dense, short-context setting. However, its theoretical regime (very long horizons, sparse channels) has now been tested and validated (see §6).
- `delta` writes help only at short range here; see §5 for why.

## 3. Length generalization (3 seeds) — main positive result

Train at ctx 160 (8 facts/sequence), evaluate the same weights at 2–8× the
training length. Fixed-gap accuracy *increases* with evaluation length in all
three seeds (gap<48 bucket shown):

| seed | eval 160 | 320 | 640 | 1280 |
|---|---|---|---|---|
| 0 | 38.2 | 63.2 | 71.7 | 75.0 |
| 1 | 32.4 | 39.3 | 45.5 | 42.9 |
| 2 | 40.0 | 54.7 | 75.4 | 85.7 |

The apparent "training-length cliff" (models fail to *train* at ctx ≥320 on a
fixed budget) is an optimization artifact, not an architectural limit: the O(1)
recurrent state supports **train-short → infer-long** deployment directly.
(`length_gen.py`)

## 4. The memory is interference-limited, not decay-limited

Holding length fixed (ctx 640) and scaling the number of stored facts
P = 8→16→24 monotonically degrades fixed-gap accuracy in all 3 seeds
(`interference_eval.py`). This explains §3: longer streams at fixed fact count
have *lower* fact density, hence less interference.

## 5. Capacity axis (DPFP feature map) — first clear mechanism win

DPFP (`key_feature_map="dpfp"`, key_dim 4×; Schlag et al. 2021) attacks
exactly the interference limit. Confirmed across 3 seeds:

- ctx 640, gap 256+, P=8: baseline (elu) at chance in all seeds {5.4, 3.2, 2.5};
  DPFP {10.7, 12.9, 31.1}.
- Highest-interference cell (P=24, gap 128-255): baseline 5.3 → DPFP 13.2 (seed 0).
- DPFP also removes the baseline's weak-seed instability (seed 1 peak accuracy
  39% → 95%).
- Compounds with length generalization (seed 2, train@160 → eval@1280):

| variant @1280 | <48 | 48-127 | 128-255 | 256+ |
|---|---|---|---|---|
| elu | 85.7 | 69.4 | 29.9 | 5.4 |
| dpfp | 88.1 | 87.8 | 70.1 | 33.5 |

Delta writes do **not** help on this task family because the interference is
cross-key feature overlap (a capacity problem), not same-key overwriting;
delta's fair test is a key-update task (pending).

## 6. cubic_flux long-horizon advantage (Validated)

In a targeted long-horizon experiment (ctx=1280, sparse retention P=8, gap ≥ 256), `cubic_flux` paired with DPFP dramatically outperforms the exponential baseline:
- `exp` + DPFP (best LR 1e-3): 20.7% recall
- `cubic_flux` + DPFP (best LR 3e-3): **63.9% recall**

This 3x absolute advantage (>4 SE) validates the core physical hypothesis that polynomial decay resolves the long-horizon forgetting problem that exponential decay suffers from, provided feature sparsity (DPFP) manages the interference.

## 7. Initial Language Modeling Viability

In a multi-seed benchmark on TinyShakespeare (~16M params, 300K tokens;
both models shared the same skip-one objective — ranking valid, absolute
values mislabeled, see §14):
- GPT-2 (Transformer baseline): Val Loss 5.703 (PPL ~300)
- **HFP** (`cubic_flux` + `delta` + `dpfp`): **Val Loss 5.548 (PPL ~257)**

HFP ranked ahead of the full-attention baseline under this shared historical objective. Because the objective was later identified as skip-one rather than next-token (§14), this is a useful viability signal and ranking record, not a final next-token/O(1) LM claim.

## 8. Current recipe

`decay_mode="cubic_flux_chunked"`, `write_rule="additive"`, `key_feature_map="dpfp"`,
`ffn_type="standard"`.

> **Write-rule: LOCKED to `additive`** by the pre-registered K2 decision
> experiment (§13): at eval 2048 delta does not beat additive by >2 SE — it is
> numerically *worse* (additive ahead by 1.8 SE), and additive also leads at
> 256 and 1024. Delta remains the tool for key-update/streaming niches
> (2x multi-seed win, key-update task), and survives in grafting as the
> learnable per-head alpha-gate hybrid (independent of this lock).

## 9. Parked / negative results (honest ledger)

- Two-tier consolidation memory: prototype verified; could not be evaluated
  fairly yet (its target regime requires a model that learns long contexts
  first). (`two_tier.py`)
- Single-seed results anywhere in this file are labeled as such; everything marked 3-seed is seed-robust in pattern, not in absolute numbers.

## 10. Language Modeling Validation (WikiText-2)

> **[Metric note, 2026-07-13]** All arms shared a double-shifted target
> (skip-one objective, §14): the component *ranking* is valid, but the
> absolute values are not next-token perplexities.

A definitive multi-seed (seeds 0, 1, 2) ablation was conducted on the WikiText-2 dataset (16M parameters, seq length 256) to validate the architectural components on dense language modeling. 

**Summary of PPL Results:**
* `exp + additive + elu` (baseline): PPL **193.9**
* `exp + additive + dpfp`: PPL **196.6** (+2.7 PPL, capacity interference)
* `exp + delta + dpfp`: PPL **193.6** (-3.0 PPL vs dpfp, delta fixes interference)
* `cubic_flux + delta + dpfp`: PPL **191.2** (-2.4 PPL vs exp)
* **`cubic_flux + additive + dpfp`**: PPL-label **183.6** (best value in this shared skip-one-objective table; see metric note above)

**Component Analysis:**
1. **DPFP Effect (alone):** In the standard exponential additive setup, DPFP degrades LM performance (193.9 -> 196.6) due to capacity overlap/interference in dense text.
2. **Delta Effect:** The Delta-write rule successfully resolves this DPFP interference (196.6 -> 193.6).
3. **Cubic Effect:** The `cubic_flux` retention law creates a massive synergistic win when paired with `additive + dpfp`, dropping PPL to 183.6. It also improves the `delta + dpfp` setup (193.6 -> 191.2).

**Conclusion:**
The architecture combination of **`cubic_flux + additive + dpfp`** is the best variant in this controlled shared-objective table. Because the absolute values are skip-one scores rather than next-token PPL (§14), use the table for component ranking only; do not cite the numbers as final language-model perplexities.

## 11. Training-length cliff applies to LM as well (3 seeds, negative result)

Training directly at seq 1024 on WikiText-2 (16M params, lr 5e-4, 2500 iters,
batch 8) leaves **both** `cubic+additive+dpfp` and `cubic+delta+dpfp` at the
`ln|V|` plateau (val loss 10.85 ≈ ln 50257) in all 3 seeds — no learning at
all, while the identical models train fine at seq 256. This extends the §3
finding (retention tasks) to language modeling: **train-short → infer-long is
required**; long-context comparisons must evaluate short-trained weights at
long lengths rather than train at length.

## 12. External family baseline: GLA (K1 decision — WITHDRAWN, see §14)

> **[Revision 2026-07-13]** This comparison mixed objectives: a metric artifact
> (double-shifted labels, §14) means the HFP values below are *skip-one*
> scores while GLA's are correct next-token perplexities. The original
> "passed" verdict is withdrawn until a clean matched-objective re-run.
> A corrected single-seed probe suggests the verdict will survive
> (next-token PPL 55.4 vs 226.7) but with a fairness caveat: the HFP LM
> config ran full attention, not the O(1) windowed configuration (§14).

Equal-parameter pure-PyTorch GLA baseline (data-dependent per-channel forget
gates, chunkwise parallel; Yang et al. 2023 family), WikiText-2, seq 256,
per-mode LR sweep {3e-4, 5e-4, 1e-3} on seed 0, then 3 seeds at best LR (3e-4).
GLA required three stabilizations to train at all (output LayerNorm, pre-LN,
1/sqrt(H) logit scale — see CHANGELOG v2.2); it is deliberately a plain family
representative (no windowed attention, elu+1 features).

| model (3 seeds) | val loss | PPL |
|---|---|---|
| GLA (best LR 3e-4) | 5.4238 ± 0.0531 | 226.7 |
| **HFP `cubic+additive+dpfp`** | **5.2127 ± 0.0035** | **183.6** |
| HFP `cubic+delta+dpfp` | 5.2534 ± 0.0248 | 191.2 |

**Pre-registered criterion (K1):** HFP-best within −2 SE of the GLA mean or
better. **Result: passed decisively** — HFP-best − GLA = −0.2111 val loss
(combined SE 0.0307, ≈ 6.9 SE in HFP's favor; 43 PPL). At equal parameters HFP
does not merely match the efficient-recurrent family representative, it beats
it, while adding the O(1)-state extra axes (retention law, capacity map).
Honest note: GLA's seed variance is ~15x HFP's (0.053 vs 0.0035).

## 13. Write-rule decision at long evaluation lengths (K2 — recipe locked)

> **[Metric note, 2026-07-13]** Both HFP arms shared the same (artifact)
> objective (§14), so the additive-over-delta decision **stands**; absolute
> values are skip-one scores, and the GLA column (correct next-token) is not
> directly comparable to the HFP columns. The degradation *pattern* is real
> and is diagnosed in §14.

Train@256 → eval@{256, 1024, 2048}, 3 seeds each, per §11's train-short →
infer-long requirement (val loss, PPL in parentheses):

| eval len | cubic+additive+dpfp | cubic+delta+dpfp | GLA |
|---|---|---|---|
| 256 | **5.2404** (189) | 5.3008 (200) | 5.4443 (231) |
| 1024 | **5.3052** (201) | 5.3623 (213) | 5.4148 (225) |
| 2048 | **5.3618** (213) | 5.4154 (225) | 5.4195 (226) |

**Pre-registered criterion (K2):** delta must beat additive by >2 SE at eval
2048. **Result: hypothesis rejected** — additive − delta = −0.0536 (combined
SE 0.0291): delta is numerically *worse* at every length. The official recipe
is locked to `cubic+additive+dpfp` (§8); delta is reserved for
key-update/streaming niches.

**Honest observation (scaling input):** HFP degrades with evaluation length
(PPL 189 → 213 at 8x train length) while GLA is flat (~225-231). HFP still
leads at 2048 (213 vs 226), but the gap narrows from 42 to 13 PPL —
long-length robustness is the next thing to attack (window size, decay
horizons), not raw short-context quality.

## 14. Metric-artifact disclosure and length-degradation diagnosis (probe, single seed)

**The artifact.** All HFP LM numbers in §7, §10, §12, §13 were produced with a
double-shifted target: training/eval code passed pre-shifted labels into models
that also shift internally (`HFPForCausalLM`, `GPT2LMHeadModel`), making the
effective objective *skip-one* prediction (x[t+2]), not next-token. Same-objective
comparisons (§7, §10, §13) remain valid as rankings; the §12 GLA comparison mixed
objectives and is withdrawn as published. Empirical calibration (probe): a
correctly-trained model scores next-token 4.017 (PPL 55.5) but 9.599 (PPL 14745)
on the skip-one pairing — the metrics are not interchangeable. Fixed in
`train.py` (FIX M1); probe code: `notebooks/degradation_probe_cell.py`.

**Corrected single-seed numbers** (train@256, correct next-token, otherwise the
§13 protocol): 4.0145 (PPL 55.4) @256 → 4.0452 (57.1) @1024 → 4.0677 (58.4)
@2048. Caveat: this LM configuration runs **full causal attention** plus the
recurrent memory (`local_window` was never set in LM runs, unlike the retention
experiments which used `local_window=8`) — it is a hybrid, not the O(1)
configuration. Re-establishing K1 requires the matched-objective GLA comparison
and an O(1)-windowed HFP LM run.

**Length-degradation diagnosis** (pre-registered probe design):

| eval variant | @256 | @2048 | gap |
|---|---|---|---|
| E0 standard | 4.0145 | 4.0677 | +0.053 |
| E1 eval-time `local_window=256` | 4.0295 | 4.0380 | +0.009 |
| E2 window + PE tiled mod-256 | 3.9973 | 3.9829 | −0.014 |

Per-position loss @2048 is flat (4.01–4.11; no cliff at 256, no monotonic
growth). Verdict: the degradation is **attention-driven, not memory-driven** —
imposing the training-time attention range at eval removes most of the gap,
tiling the positional encoding removes the rest (E2 @2048 matches or beats the
model's own @256 score, within sampling noise ±0.02). No evidence of
memory-state OOD: the recurrent memory path is length-robust, consistent with §3.
Practical consequence: train-short/full-attention → deploy with eval-time
window + tiled PE is a zero-training fix that is simultaneously length-stable
and O(1) at inference. (Single seed; diagnostic, not a headline claim.)

## Reproduction

```bash
python smoke_test.py
python review_scripts/verify_claims.py
python review_scripts/dense_retention.py exp additive 1e-3 0
python review_scripts/length_gen.py train 0 && python review_scripts/length_gen.py eval 0
LG_VARIANT=dpfp python review_scripts/length_gen.py train 0
python review_scripts/interference_eval.py 0
```
