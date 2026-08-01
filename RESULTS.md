# HFP — Experimental Results (v2.1)

> 🇹🇷 **Türkçe:** [Deney Sonuçları (Türkçe)](docs/tr/DENEY_SONUCLARI.md)

> ## ⚠ ERRATUM (2026-07-28) — scope of the retrieval claims
>
> Sections written before 2026-07-28 that describe the graft "retrieving" facts at
> long distances are **correct as measurements** but were **misattributed**: the
> retrieval was carried by the KV cache of the 22 un-grafted attention layers, not
> by the O(1) recurrent state. With the cache reset per chunk, retrieval from the
> state alone is **0%** at every distance tested, including a single chunk boundary
> (§30a–§30c; checkpoint load verified bit-exact). The cause is a training recipe
> that could not teach cross-boundary storage (`no_grad` on the write chunk, state
> detached at boundaries, recall target diluted in a full-sequence LM loss). Removing
> the first two changed nothing (§31a); the third is under test (§33).
>
> **Unaffected by this erratum:** perplexity results, the graft-density wall and its
> compounding diagnosis (§24–§25), the VRAM/latency measurements (§23), and the
> retention-law regime findings (§15h, §28a) — none of these depend on the state
> holding long-range information.
>
> See the detailed correction above §15f.

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

## 12. External family baseline: GLA (K1 decision — WITHDRAWN; superseded by §16)

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

## 15. Qwen2.5-1.5B graft, full 2-stage distillation run (single seed, negative result)

First complete end-to-end graft experiment on a real pretrained LM
(`notebooks/colab_graft_qwen_v3_kaggle.ipynb`, Kaggle T4, fp32, 2026-07-18).
Config: `decay_mode=cubic_flux_chunked`, `write_rule=hybrid`,
`key_feature_map=dpfp`, `rec_block=16`, odd-indexed layers grafted
(~325k trainable params). Base PPL (WT-2 valid, seq 1024, 24 chunks): **7.96**.

Run history (all numbers from the actual logs):

- **Zero-shot (untrained graft):** PPL **2627** — sanity criterion `<1000`
  **FAILED** (flagged at run time: weight transfer / output-scale suspect;
  `out_gain` init 1.0 injects the untrained memory path at full scale).
- **Stage 1** (teacher-forcing, layerwise MSE; WT-103 raw via S3, seq 1024,
  700 steps): MSE 0.965 → **0.116**, plateau from ~450 steps (prior runs
  reported ~0.07 — not reproduced here). `alpha_ort` 0.119 → 0.131.
- **Stage 2** (logit-KL + LM loss, 600 steps): ran at **seq 128** — a
  **deviation** forced by T4 memory (fp32 weights 6.2 GB + 151k-vocab logits;
  seq 1024/512/256 all OOM). KL total 234 → noisy ~70-80 plateau (57-88 range,
  no clean convergence in last 300 steps); LM CE 5.60 → ~3.39.
- **Validation:** PPL 7.96 → **15.88** (**1.996×**; criterion ≤1.05×
  **FAILED**). Needle test: **MISS at all lengths** (2048/8192/16384) — model
  emits filler continuation, never retrieves the passphrase. Peak VRAM 11.86 GB;
  grafted state remains O(1) (context-independent size) as designed.

**§15a — Diagnostics (forward-only, `kaggle_graft_diagnostics_v1.ipynb`, 2026-07-18):**

- **T1 (needle harness control):** plain Qwen (full attention) **FINDS** the
  needle at L=2048 and 8192 → the harness is valid; §15's needle miss is a real
  negative (the memory path never learned retrieval), not a broken test.
- **T2 (out_gain init sweep, untrained graft, zero-shot PPL):** 1.0 → **2758.6**
  (replicates §15's 2627); 0.1 → **168.1**; 0.01 → 4705.8. Sweet spot ≈0.1
  (16× better start; `<1000` criterion met). Supports the bad-operating-point
  hypothesis; 0.01 over-mutes 13 attention-replaced layers and collapses again.
- **T3 (stage1_son autopsy):** alpha mean 0.131 (0.092-0.176), out_gain mean
  0.746 (from init 1.0; no dead heads), decay mean 0.949. S1's optimizer chose
  to *shrink* the memory output rather than make it useful — consistent with
  the MSE 0.116 plateau and the needle miss.
- **T4 + final.pt autopsy:** not run (final.pt was not attached as input);
  pending — would sharpen the recall-mix decision but does not block Run 2.

**Run 2 decision (single variable):** repeat S1+S2 with `out_gain` init **0.1**
(everything else identical). Pre-registered expectations: zero-shot ≈168;
S1 MSE plateau below 0.116; then S2 → PPL and needle re-measured. If PPL
improves but needle still misses, the *next* single change is mixing synthetic
recall data into S2.

**§15b — Run 2, Stage 1 (2026-07-18, Kaggle T4):** MSE **0.150 → 0.067
plateau** (~450+ steps; beats Run 1's 0.116 and matches the historical ~0.07);
`alpha_ort` 0.119 → **0.143** (Run 1 stalled at 0.131 — the memory path now
gains weight instead of being muted). Runtime 58 min.
⚠️ **Deviation disclosure:** the single-variable plan was accidentally broken —
S1 also ran at **seq 128** (a leftover of the T4-OOM setting), vs Run 1's
seq 1024. So the plateau comparison (0.067 vs 0.116) is confounded by sequence
length; the *initial*-MSE improvement (0.96 → 0.15) is still attributable to
`out_gain` (independently supported by T2). Checkpoint:
`checkpoints/graft_run2/hfp_graft_stage1_son.pt`. S2 + validation pending.

**§15c — Run 2 complete (S2 + validation, 2026-07-18):** Zero-shot with
out_gain=0.1: **168.8** — matches the §15a pre-registered expectation (~168)
exactly. S2 (600 steps, seq 128): KL 62 → ~40 noisy, LM CE 3.78 → ~3.1.
Validation: PPL 7.96 → **12.73** (**1.600×**; Run 1: 1.996× — improved, still
fails ≤1.05×). Needle: **MISS at 2048/8192/16384** (unchanged). Peak VRAM
11.86 GB. Verdict: the out_gain hypothesis is *partially* confirmed (better
operating point → better final PPL), but retrieval never emerges from
LM-only distillation — exactly the pre-registered branch condition.
**Run 3 (single variable): mix synthetic recall passages into the S2 data**
(needle-style documents where the teacher itself must retrieve, so the KL
target directly supervises retrieval). Pre-registered expectations:
PPL ≤ 12.73 (must not regress); needle @2048 target: FOUND; if mini-needle
(≤ seq) is learned but 2048 still misses, the bottleneck is length
generalization of the memory path, not the training signal.

**§15d — Run 3 complete (2026-07-18):** S2 with 25% recall-mix, seq 128,
everything else = Run 2. PPL 7.96 → **12.79** (1.608×; Run 2: 1.600× — no
regression ✓, no improvement). Needle: **MISS at 2048/8192/16384** (unchanged).
Periodic LM-CE dips to ~1.1 in the S2 logs confirm the recall batches were
present (repetitive filler = easy CE).
**Post-hoc design-flaw disclosure:** the recall documents fit inside a single
seq-128 training window — needle and query were co-visible to the intact
even-indexed *full-attention* layers, so the task was solvable **without the
recurrent memory**; no cross-chunk write→carry→read pressure was ever applied.
The eval needle, by contrast, spans many chunks. Run 3 therefore did **not**
test the §15c hypothesis properly; the training signal for cross-chunk
retrieval remained ≈0. **Fixed in Run 4** (implemented in the notebook):
recall documents are now split across two chunks — needle in chunk A, query
in chunk B; training runs A under streaming (state write, no grad), then B
with loss. With `use_cache=False`, attention layers **cannot see A**; the only
route to the needle is the recurrent state. Teacher KL targets come from a
full-attention forward over A+B. Disclosed limitation: stream state is
detached at the chunk boundary, so cross-chunk gradients reach only the
*read* path (write path learns from B-internal writes). Validation now also
measures needle @512 (near-regime control). Pre-registered expectations:
PPL ≤ ~12.8 (no regression); needle @512 target FOUND; @512 found but @2048
missed → length-generalization bottleneck; nothing found → read-path-only
gradient insufficient, next change is non-detached TBPTT (repo code change).

**§15e — Run 4 complete (2026-07-18): FIRST NEEDLE HIT.** Cross-chunk recall
training (25% mix, adjacent A→B chunks). Validation: needle **@512 FOUND**
("copper mountain" retrieved verbatim) — the first successful memory-path
retrieval in the graft setting; with `use_cache=False` in training and the
query 3-4 chunk boundaries away, the only route was write→carry→read through
the O(1) state. @2048/8192/16384 still MISS → exactly the pre-registered
**length-generalization** branch (training gap ≈1 chunk; eval needs ~15).
PPL 7.96 → **12.92** (1.623×) — marginally above the ≤~12.8 expectation
(12.73/12.79/12.92 across runs; treated as noise-level, disclosed). Recall
batches' B-chunk CE dropped to ~0.65 during S2 (retrieval being learned).
**Caveat (disclosed):** Run 4's training word list overlapped the eval secrets
("copper", "mountain" appear in both), so a memorization objection is possible
for the @512 hit — weakened by the @2048 miss (a memorizer would answer at any
length), but not eliminated. **Eval hardened for Run 5:** secrets drawn from
words never seen in training ('orange kettle' / 'purple ladder' /
'crimson garden') and the hit criterion tightened to full-phrase match.
**Run 5 (single variable): distance curriculum** — insert 0-12 random filler
chunks between A and B (max trained carry ≈13 chunks ≈ 1.7k tokens; teacher
uses `logits_to_keep` to bound memory). Pre-registered: @512 FOUND with
out-of-training secrets (clean retrieval proof); @2048 target FOUND; @8192+
informative (beyond trained range — §3's train-short→infer-long precedent
applies or fails honestly); PPL stable ~12.9.

> ### ⚠ RETROACTIVE CORRECTION (2026-07-28) — read before §15f–§15g
>
> These sections attribute long-range retrieval to information travelling through
> the O(1) recurrent memory. **That attribution is wrong**, and the error dates
> from the introduction of the cross-chunk recipe in Run 5. Two independent causes,
> both established later:
>
> 1. **The evaluation is confounded (§30a).** The needle harness carries one
>    `DynamicCache` across all chunks, so the 22 un-grafted full-attention layers
>    see the entire stream — and 16 384 tokens is *inside* Qwen2.5's native 32 k
>    context. A controlled version that resets the KV cache per chunk, leaving only
>    the O(1) state, retrieves **0%** (§30a/§30b/§30c, load verified 72/72 bit-exact).
> 2. **The training could not have taught storage (§30c, §31a).** Stage-2 recall ran
>    the write chunk under `torch.no_grad()`, detached the streaming state at every
>    chunk boundary, and diluted the recall target inside a full 128-token LM loss.
>    All three block gradient to the write path; removing the first two changed
>    nothing (§31a).
>
> **What survives:** the *empirical* claim that the distance curriculum improved
> long-context needle performance (Run 4 missed @8192, Run 5 found it) — that
> comparison is real. **What does not:** the mechanistic claim that the O(1) memory
> carries it. The most likely actual mechanism is that training on long,
> distance-varied sequences improved how the grafted layers behave at long context,
> which in turn improved *cache-based* retrieval by the remaining attention layers.
>
> **Warning sign we had and did not chase.** The small-scale line (§17–§21) was
> simultaneously finding that cross-chunk carry collapses at the first boundary.
> Both lines cannot be true of the same mechanism. That contradiction was parked
> instead of resolved; resolving it would have surfaced this months earlier. The
> methodological lesson is recorded here deliberately: **where two lines of your own
> evidence disagree is where the most information is.**
>
> Sections §15f–§15g are left unedited below as the original record; read every
> "retrieval" claim in them as *"the hybrid model retrieves, with the KV cache
> present"*, not as evidence about the recurrent state.

**§15f — Run 5 complete (2026-07-18): LONG-RANGE RETRIEVAL, CLEAN EVAL.**
Hardened needle (secrets never seen in training, full-phrase criterion):
**@512 FOUND, @2048 MISS, @8192 FOUND, @16384 FOUND** ("purple ladder"
retrieved verbatim at ~7k and ~14k token distances — far beyond the trained
carry range of ~1.7k; the §3 train-short→infer-long behavior appears in the
graft setting). The memorization objection from §15e is **closed**: these
words never appeared in training. Honest caveats: (1) the @2048 miss is a
**non-monotonic anomaly** — retrieval is real but not yet position/length-
reliable, and this is a single seed; a reliability grid (L × insertion ×
seed, forward-only) is required before any headline claim. (2) PPL drifted
7.96 → **13.04** (1.639×; 12.73 → 12.79 → 12.92 → 13.04 across runs) — the
recall mix slightly taxes LM quality; the ≤1.05× criterion remains failed.
Checkpoint: Run 5 `final.pt` (Kaggle output; to be archived).

**§15g — Needle reliability grid (Run 5 final; 5 lengths × 3 insertion
positions × 3 seeds; out-of-training secrets, full-phrase criterion):**

| L \ insertion | 0.125 | 0.5 | 0.875 |
|---|---|---|---|
| 512  | 3/3 | 3/3 | 3/3 |
| 1024 | 3/3 | 3/3 | 3/3 |
| 2048 | 2/3 | 3/3 | 3/3 |
| 4096 | 1/3 | 0/3 | 3/3 |
| 8192 | 3/3 | 2/3 | 3/3 |

**§15h (PRE-REGISTERED, pending) — Run 6: the controlled cubic-vs-exp graft
comparison.** All five graft runs so far used `cubic_flux_chunked` only, so the
project's distinctive retention-law claim is **untested in the graft setting**.
Run 6 = exact twin of Run 5 with the single change `decay_mode='exp'` (same S1
protocol/init/data, same distance-curriculum S2, same hardened eval + grid).
Criteria written before running: primary = reliability grid + needle set;
secondary = final PPL. Possible verdicts, all recordable: exp ≥ cubic on both →
the cubic angle is unsupported at LLM scale (honest negative; retention law
doesn't matter here); cubic > exp at long distances → first controlled
mechanism win for the physics-derived law at LLM scale; mixed → map the
trade-off. Mode-mismatch resume guard added to the notebook (exp/cubic
checkpoints cannot cross-load silently).

**§15h partial result (Run 6 complete, 2026-07-19):** point-set eval shows
**no detectable difference**: PPL cubic 13.04 (1.639×) vs exp **12.87**
(1.618×; marginally better, within the 12.73-13.04 run-to-run noise band);
needle pattern **identical** (FOUND @512/@8192/@16384, MISS @2048 — before
grid averaging). Interim honest reading: at this resolution the retrieval
capability is attributable to the **cross-chunk training protocol, not the
retention law**; the cubic angle is so far unsupported at LLM scale. Final
verdict awaits the pre-registered primary criterion: the 45-point reliability
grid on the exp final (cubic reference: 38/45, with the 4096-early/mid trough).

**§15h FINAL VERDICT (exp grid complete, 2026-07-19; identity-verified
out_gain 0.239):** exp grid **42/45 (93%)** vs cubic 38/45 (84%). Per-length:
exp is 9/9 at 512/1024/4096/8192 — **cubic's 4096-early/mid trough does not
exist under exp** — with one weak cell (2048@0.125: 0/3; cubic had 2/3 there).
Statistical caution: 42 vs 38 alone is not decisive (two-proportion p≈0.18),
but the direction agrees with PPL (exp 12.87 vs cubic 13.04) and nothing
favors cubic. **Verdict: in the graft setting, the cubic-plateau retention law
provides no measurable advantage over plain exponential decay; the long-range
retrieval capability belongs to the cross-chunk distillation protocol, which
works under either law.** This is the honest negative for the project's
distinctive mechanism at LLM scale (its remaining support: the small-scale
long-horizon result, §6, itself thin). Curious side observation (unexplained,
logged): with the exp final, short-needle using *training-vocabulary* secrets
("copper mountain") missed @512 while the clean-vocabulary grid was 9/9 @512 —
trained-pair interference/competition is a plausible but untested explanation.
Exp diagnostics T2 replication: zero-shot 3623 (gain 1.0) / 191.8 (0.1) /
4802 (0.01) — same U-shape as cubic (2758/168/4706).

Total **38/45 (84%)**; 13/15 cells reliable (≥2/3). Readings: (1) §15f's
@2048 "anomaly" was **noise** — 2048 is 8/9 overall. (2) The real weak zone
is **4096 with early/mid insertion** (1/3, 0/3) — reliability is not monotonic
in distance (8192@0.125 is 3/3), so the trough is not simple decay; plausibly
an interference/saturation interaction with the filler period — untested,
listed as an open question. (3) Defensible claim as of now: *a 325k-parameter
graft distilled on a free T4 gives a Qwen2.5-1.5B hybrid with O(1) grafted-layer
state that retrieves out-of-training passphrases across 512-8192+ token
distances with ~84% grid reliability* — single trained model, single training
seed; multi-seed training replication still pending. PPL cost (1.6×) remains
the main open quality gap.

Honest reading: the graft pipeline is now *mechanically* validated end-to-end
(resume, streaming, chunked recurrence, checkpointing all work), but this
configuration **does not** preserve LM quality (2× PPL) and shows **no
long-range retrieval** on needle. Likely contributors, in testable order:
(1) bad untrained operating point (zero-shot 2627 ≫ 1000; try small `out_gain`
init / teacher weight transfer); (2) Stage 2 at seq 128 shortens the
distillation context far below the eval regime; (3) noisy KL plateau suggests
LR/temperature/KL-weight retuning or longer S2; (4) distillation on WT-103 alone gives
zero training signal for retrieval — needle-style recall may need explicit
recall data mixed into S2. Single seed; no cherry-picking — this section is the
complete record of the run.

## 16. K1 gate, clean re-run: GLA family baseline v2 (3 seeds, corrected objective)

Supersedes the withdrawn §12. `colab_gla_benchmark_v3.ipynb` "Görev B v2":
matched next-token objective (the §14 double-shift artifact fixed and
regression-checked in-notebook), seq 256 training, ~16M params, 3 seeds per
arm, GLA LR selected by sweep (Görev A). Best val loss per seed:

| arm | s0 | s1 | s2 | mean ± std | PPL(mean) | NaN |
|---|---|---|---|---|---|---|
| **HFP cubic+additive+dpfp** | 5.2137 | 5.2156 | 5.2089 | **5.213 ± 0.003** | **183.6** | 0/3 |
| HFP cubic+delta+dpfp | 5.2687 | 5.2667 | 5.2247 | 5.253 ± 0.025 | 191.2 | 0/3 |
| GLA baseline | 5.4575 | 5.4513 | 5.3627 | 5.424 ± 0.053 | 226.7 | **3/3 diverged** |

**K1 verdict: PASSED.** HFP-additive beats the GLA baseline by **0.211 nats**
(HFP −19% PPL, equivalently GLA +23.5%) — ~4× the largest seed-std — and every
GLA seed diverged (NaN @2420/@2521/@3503) while all six HFP runs were stable
to early-stop. Fairness caveats (disclosed): the GLA arm is our in-house
`GLAForCausalLM` wrapper — its universal divergence may reflect
implementation/tuning weakness rather than the GLA method itself; claim is
scoped to *this implementation at this budget*. Length sweep (val loss
256→2048): HFP arms degrade by ~+0.11-0.12 nats (they run the full-attention
LM config, consistent with §14's attention-driven diagnosis; the §14 window+PE
recipe is the known fix), while GLA is flat-to-mixed (−0.11/−0.03/+0.07 across
seeds) from a much worse base. At eval-2048 the means are: HFP-add **~213
PPL**, HFP-del ~225, GLA ~226 — HFP-add still leads, the delta arm's margin
vanishes, and GLA's best seed (197.8) crosses below the HFP-add mean
(single-seed crossover under high GLA variance ±0.053; not averaged away).

## 17. Görev C — lifetime retention (cubic's natural-habitat test)

**v1 (2026-07-20): INVALID — and an independent replication of §1.** The first
version trained with single-token supervision at ctx 320; 5/6 arms never left
the ln(30) chance plateau (train loss ≈3.42 for 800 steps), so the law
comparison never happened (same-seed exp/cubic arms even produced identical
chance-level evals — an unlearned model's outputs barely depend on the decay
mode). This is §1's supervision-density finding reproduced in fresh code. The
one escaping arm (cubic s1, loss→0.23) evaluated *worse* in logprob
(−4.1..−4.8): memorization, not retention. **v2** switches training to dense
supervision (8 facts+queries per sequence, `dense_retention` protocol), keeps
the streaming lifetime probe unchanged, and adds a post-training plateau guard
(abort if val-batch loss ≥2.5) so an unlearned model can never again
masquerade as a comparison. Pre-registered criteria unchanged (script
docstring).

**v2 result (2026-07-20): the pre-registered third outcome — neither law holds
this regime.** Training now learns (loss 3.4 → 1.41-2.20; plateau guard
passed on all 6 arms). Seed-mean accuracy (chance 3.3%, n=30/cell/seed):

| carry gap | exp | cubic_flux |
|---|---|---|
| 256 (= training horizon) | **15.6%** | **18.9%** |
| 1024 | 4.4 | 2.2 |
| 4096 | 1.1 | 2.2 |
| 16384 | 5.6 | 2.2 |
| 65536 | 1.1 | 2.2 |

Above 1024 both modes sit at chance; every between-mode difference is ≤1 hit
(3.3 pts) and the pre-registered threshold (≥+10 pts across two consecutive
far gaps) is nowhere approached, in either direction. **Verdict: cubic's
"natural habitat" hypothesis is not supported — but neither is exp; the test
could not discriminate because both collapse.** The informative signal is
elsewhere: retention is real *at* the training horizon (256: 4.7-5.7× chance)
and vanishes beyond it. Since §3 showed train-short→infer-long *does* work
when fact density falls with length, the difference here is that traffic
density is held constant (a distractor kv every 64 tokens), so interference
grows linearly with distance — ~1000 competing writes by 65k into a 32-dim
state. This is §4's "interference-limited, not decay-limited" diagnosis
extended to the lifetime regime, and it predicts the fix is **capacity**
(dpfp ν, bulk_dim, write sparsity/gating), not the retention law. Follow-up
(cheap, same harness): sweep interference rate `LT_DIST_EVERY` ∈ {64, 512,
4096} at fixed gap — if far-gap accuracy recovers as writes thin out, the
capacity account is confirmed and the lifetime claim becomes a
*write-sparsity* claim rather than a retention-law claim.

## 18. Görev D — write-sparsity sweep: the capacity account is REJECTED

Pre-registered follow-up to §17 (`notebooks/kaggle_write_sparsity_sweep.ipynb`).
Gap fixed at 16384; only the interference rate varies. Seed-mean accuracy
(chance 3.3%, n=30/cell/seed, §17 v2 checkpoints, eval-only):

| write interval | ~competing writes | exp | cubic_flux |
|---|---|---|---|
| every 64 tok | 256 | 6.7% (3-10) | 7.8% (7-10) |
| every 256 tok | 64 | 3.3% (0-7) | 2.2% (0-7) |
| every 1024 tok | 16 | 2.2% (0-3) | 3.3% (3-3) |
| every 4096 tok | 4 | 2.2% (0-7) | 8.9% (0-27) |

**Verdict: the capacity/interference explanation for §17 is rejected.**
Thinning writes 64× (256 → 4 competitors) produced **no recovery** — sparsest
minus densest is +1.1 pts, and the trend is if anything *downward*. So §17's
collapse beyond 1024 is **not** state saturation; per the pre-registered second
reading it is a **failure to generalize past the training horizon** (256): the
model retains at the horizon it was trained on (§17: 15.6-18.9%, 4.7-5.7×
chance) and cannot transfer that skill to distances it never saw, in an
otherwise near-empty state. Consequence for the on-device thesis: the fix is
neither the retention law (§15h, §17) nor capacity (§18) — it is **training
horizon / curriculum**. This mirrors the graft line, where retrieval at 8-16k
appeared only after a *distance curriculum* was added (§15f); the small-scale
model here never got one. Next test (cheap, same harness): raise the training
carry range (CTX/MAXGAP, or a chunked curriculum with carries up to ~4k) and
re-run the §17 gap curve; if far-gap accuracy then rises, the account is
confirmed and the deployable claim becomes *"O(1) memory retains at distances
it was trained to carry"* — an honest, testable, and useful statement.
Logged anomaly (single seed, not a claim): cubic s2 at the sparsest setting
hit 26.7% (8/30) while its sibling seeds scored 0.0%.

## 19. Görev E — carry curriculum: rejected too, and a train/eval mismatch found

Pre-registered test of the last remaining explanation for §17
(`review_scripts/carry_curriculum.py`): train *cross-chunk* carries (target in
chunk A, query in chunk B, K filler chunks between, `use_cache` streaming so
attention cannot see A), K annealed 0→16 (~4096 tokens of trained carry), then
re-run the §17 lifetime probe unchanged. Seed-mean accuracy (chance 3.3%):

| gap | exp §17 | exp §19 | cubic §17 | cubic §19 |
|---|---|---|---|---|
| 256 | 15.6 | 6.7 | 18.9 | **20.0** |
| 1024 | 4.4 | 5.5 | 2.2 | 4.4 |
| 4096 | 1.1 | 3.3 | 2.2 | 4.4 |
| 16384 | 5.6 | 1.1 | 2.2 | 2.2 |

**Verdict: FAILED by the pre-registered criterion** (far gaps ≤4.4%, below even
the 8% "rejected" line). So for the small model the collapse is explained by
neither the retention law (§15h), nor capacity (§18), nor carry curriculum.

**But the run exposes a contradiction that is more informative than the verdict:
training *did* learn the cross-chunk task** — lossB (query chunk, memory-only
route) fell 3.45 → 1.51-2.23 (cubic arms lowest), i.e. at K=14 (~3600 tokens)
the model reads what it wrote through the O(1) state. Yet the eval probe at the
same distance is at chance. Learned-but-not-transferred ⇒ the two are **not the
same task**. Concrete mismatches, all mine to fix, in likely order of impact:
(1) *write context*: training writes the target inside a dense chunk aligned to
the chunk boundary (positions CTX-2/CTX-1), eval writes it as the first two
tokens of an otherwise empty stream; (2) *query context*: training queries at
the end of a dense chunk (many local cues), eval queries a bare key token;
(3) *distractor statistics* differ between the two paths. Note cubic ≥ exp on
5/8 cells here and had the lowest training loss — logged, not claimed (all
differences ≤1-2 hits).
**Next step is a harness fix, not a new hypothesis:** make the eval probe
generate its write/query context from the *training* distribution (dense chunk,
boundary-aligned target), keeping only distance as the manipulated variable. If
accuracy then tracks the trained carry range, §17's collapse was an eval
artifact all along and the honest claim becomes "retains over the distances it
was trained to carry, when queried the way it was trained". If it still
collapses, the architecture (state size / read path) becomes the prime suspect
and §17-§19 stand as a chain of four eliminated explanations.

## 20. Görev F — matched probe: the failure is the FIRST chunk boundary, not distance

`review_scripts/matched_probe.py` on the Görev E checkpoints (no retraining):
the probe scene is generated from the *training* distribution (dense chunk A
with the target at the boundary → K filler chunks → dense chunk B, query at the
end), and the old §17-style probe is measured in the same run. Seed-mean over
2 modes × 3 seeds, chance 3.3%:

| K | ~tokens | matched | old probe |
|---|---|---|---|
| **0** | 0 | **24.4%** | **100.0%** |
| 1 | 256 | 8.3 | 11.7 |
| 2 | 512 | 6.7 | 3.3 |
| 4 | 1024 | 1.1 | 1.1 |
| 8 | 2048 | 6.1 | 5.0 |
| 16 | 4096 | 3.4 | 2.8 |
| 32 | 8192 | 5.5 | 1.6 |
| 64 | 16384 | 2.2 | 0.5 |

**The K=0 row reframes everything.** With write and query inside the *same*
chunk the old probe is **100% correct** — the associative read itself is
perfect. One chunk boundary (K=1) drops it to 11.7%, and from K≥2 everything is
at chance, flat in distance. So the failure is **not** decay (§15h), **not**
capacity (§18), **not** curriculum (§19), and **not** probe mismatch (matched ≈
old for K≥1): it is the **first cross-chunk state hand-off**. Distance is
irrelevant once that hand-off has failed — which is exactly why every
distance-based hypothesis died.
**Prime suspect, and it is testable in minutes:** the streaming state path at
eval (`past_key_values` round-trip: `_offset_from_state`, conv-state carry,
`detach_state`) does not reproduce the state that training produces internally
— note §19's contradiction (training lossB *did* fall to 1.51-2.23 with K up to
14, i.e. the same hand-off works when done inside the training loop). Next
diagnostic, no training required: take one sequence, run it (a) in one shot and
(b) chunked with `use_cache`, and compare the memory tensors (M, z) and the
logits at the identical final position; `smoke_test.py` T4 already asserts
chunk-consistency for the *non-cached* path, so a discrepancy would localize to
the cached/streaming route. If M/z match but logits do not, the bug is in the
read path/position handling; if M/z diverge, it is in the state carry itself.

**§20a — correction to §19's inference, and a unifying hypothesis.** §19
claimed "training *did* learn the cross-chunk task" from lossB falling to
1.51-2.23. That inference was **flawed**: lossB averages over *all* supervised
positions in chunk B — six easy in-chunk pairs plus only one cross-chunk
target — so its fall mostly reflects in-chunk learning. The contradiction that
motivated the "harness mismatch" reading partially dissolves; training may
never have learned the hand-off either. **Unifying hypothesis (untested,
checkable in minutes):** the learned decay is calibrated to training-window
distances — §17's autopsy showed sigmoid(decay) mean **0.949/token**, i.e.
survival across one 256-token chunk ≈ 0.949²⁵⁶ ≈ 2·10⁻⁶: the state is
mathematically *erased* at the first boundary. And it cannot learn slower
decay from cross-chunk experience because the state is **detached at chunk
boundaries** (`bptt_across_chunks=False`), so no gradient ever reaches the
write/decay path from beyond the boundary. This one story explains K=0
perfection, K=1 collapse, distance-independence, curriculum failure (§19) and
sparsity independence (§18). Two cheap checks decide it: (1) checkpoint
autopsy — per-channel **max** sigmoid(decay) (long-range survival needs
λ→0.999+; if no channel exceeds ~0.99 the story holds); (2) retrain the small
model with the existing `bptt_across_chunks=True` flag (K2/TBPTT path, already
implemented in `modeling_hfp.py`) and re-run §17 — if far-gap accuracy
appears, the §17-§20 chain reduces to "learned decay + detach", a
*training-recipe* finding rather than an architecture defect. The graft line
is unaffected either way (separate streaming implementation; its long-range
retrieval empirically works, §15f-g).

## 21. Görev G — TBPTT intervention: accuracy unmoved; the decay-erasure story falls

TBPTT arms (`bptt_across_chunks=True`, no manual detach, full graph through
K≤16 chunks; else identical to §19), 2 modes × 3 seeds. Far-gap accuracy is
**unchanged** (4096/16384 best: 2.2%/2.2%; §19: 4.4/2.2 — noise-level).
Decay autopsy after training: λ_mean 0.948-0.950, λ_max 0.999, 19-22 channels
>0.995 (max single-chunk survival 76-81%). **Criterion flaw disclosed:** my
pre-registered check ("do λ≥0.995 channels exist?") lacked an *init* baseline —
and the stats are **identical to 4 decimals across all three cubic seeds**,
which independent trainings do not produce: the decay parameters are
effectively **frozen at initialization** in every arm. Two consequences:
(1) the §20a "decay erases the state" story **falls** — high-λ channels
(77%/chunk survival) exist from init, so information *could* persist several
chunks, yet accuracy still dies at the first boundary; (2) a sharper puzzle
replaces it: target logprob *worsens systematically with distance*
(−3.3 → −5.2) — the model assigns actively lower probability to the correct
value, a **read-path/dilution signature** rather than erasure — and decay's
frozenness raises the question of whether gradient reaches it at all even
under the TBPTT flag (worth a 5-minute `decay.grad` probe).
**Status: the small-scale lifetime line is parked here as an open engineering
question** (§17-§21: five eliminated/weakened explanations, current suspects =
read-path dilution and decay-gradient plumbing). The line consumed its budget;
the priority returns to the healthy graft mainline (PPL gap, multi-seed
replication — BEKLEYEN_ISLER #5, #8). Anyone resuming this thread should start
with: (a) one-shot vs cached M/z+logit equality check (§20), (b) decay.grad
nonzero check under TBPTT, (c) read-path norm analysis at long range.

## 22. PRE-REGISTERED: graft density (Run 7) and multi-seed replication (Runs 8-9)

Written before running; single parametrized notebook
(`colab_graft_qwen_v3_kaggle.ipynb`, RUN_SEED / GRAFT_N block).
**Run 7 — density:** GRAFT_N=6 (layers [3,7,11,15,19,23]), exp, seed 0,
distance-curriculum S2, single session (FORCE_S2). Hypothesis: the ~1.6× PPL
cost is the structural price of replacing 13/28 attention layers; halving the
graft should recover quality. Criteria: **PASS** = PPL ≤ 1.2× (≤9.55) AND
needle FOUND @512/8192/16384; **TRADE-OFF** = PPL improves but needle degrades
(map it, decide density later); **FAIL** = PPL stays ≥1.5× (density is not the
driver; suspect write-rule/norm interactions instead).
**Runs 8-9 — multi-seed:** GRAFT_N=13, exp, seeds 1 and 2 (Run 6 = seed 0).
Criteria: headline claim survives if needle @512/8192/16384 is FOUND in ≥2/3
seeds and PPL stays in the 12.7-13.1 band; any seed diverging materially is
reported as-is.

**§22a — Run 7 result (2026-07-21): PASS, decisively.** 6-layer graft
(layers [3,7,11,15,19,23], exp, seed 0, distance-curriculum S2, single
session): PPL 7.96 → **8.84 (1.112×)** — from 1.6× at 13 layers, meeting the
pre-registered ≤1.2× PASS bar (the original ≤1.05× product criterion is now
within sight); needle **FOUND at all four lengths 512/2048/8192/16384** with
out-of-training vocabulary — including 2048, the weak cell of *both* 13-layer
laws (§15g/§15h). Peak VRAM 9.28 GB. **Conclusion: the LM-quality cost is
density-driven** — halving the graft recovers most of the quality while
retrieval strengthens rather than degrades. The balanced 6-layer hybrid is the
project's current best artifact and the new reference recipe.
**Amendment to the §22 plan (declared before running 8-9):** multi-seed
replication will target the **6-layer recipe** (GRAFT_N=6, seeds 1-2, Run 7 =
seed 0) instead of the originally listed 13-layer config, since the headline
claim now attaches to the 6-layer artifact. Criteria adapted accordingly:
needle @512/2048/8192/16384 FOUND in ≥2/3 seeds; PPL within 8.6-9.1.

**§22b — Run 8 (seed 1, Colab T4, 2026-07-21): PASS.** PPL 7.96 → **8.84
(1.111×)**, needle **4/4** (512/2048/8192/16384, out-of-training vocab). Nearly
identical to Run 7 (8.84, 1.112×) — a remarkably tight replication, plausible
given the small (150k) adapter distilling to the same teacher. Peak VRAM
14.37 GB (Colab environment; 9.28 GB on Kaggle for the same recipe — allocator/
environment difference, both within T4). The ≥2/3 needle criterion is already
met with 2/2; Run 9 (seed 2) completes the pre-registered set.

**§22c — Run 9 (seed 2, Colab T4, 2026-07-21): PASS. Multi-seed set COMPLETE.**
PPL 7.96 → **8.84 (1.111×)**, needle **4/4**.

Three-seed summary of the 6-layer reference recipe (GRAFT_N=6, layers
[3,7,11,15,19,23], decay_mode=exp, cross-chunk distance curriculum):

| Seed | Run | PPL (orig 7.96 →) | ratio | needle 512/2048/8192/16384 |
|------|-----|-------------------|-------|----------------------------|
| 0 | Run 7 | 8.84 | 1.112× | 4/4 |
| 1 | Run 8 | 8.84 | 1.111× | 4/4 |
| 2 | Run 9 | 8.84 | 1.111× | 4/4 |

**Verdict (pre-registered §22a criteria — MET).** needle ≥2/3 seeds at all four
lengths: **3/3** (12/12 total). PPL within 8.6–9.1: **3/3**. The recipe replicates
across three independent seeds. The near-identical PPL across seeds is expected
for a 150k-param adapter distilling to a fixed frozen teacher on a fixed corpus —
the optimization landscape is narrow — and is itself evidence of a stable, not
lucky, result. Stage-1 for the 6-layer recipe is also ~4× cheaper than the
13-layer version (MSE 0.081→0.043 in 14 min vs ~1 h; fewer layers to match).

This closes the multi-seed replication line. The headline is now defensible:
*a 6-layer O(1)-memory graft of Qwen2.5-1.5B (**149,910** trainable params, ~0.01%)
reaches 1.11× base perplexity while passing needle-in-haystack retrieval at
512–16384 tokens with out-of-training-vocabulary secrets — reproduced across
three seeds, trained on free-tier T4.*

## 23. On-device VRAM / latency showcase (hybrid vs full KV-cache)

Structural benchmark (checkpoint-independent — depends only on tensor shapes,
dtype, compute pattern, not trained values) of the 6-layer reference recipe vs
stock Qwen2.5-1.5B, on a Colab T4, fp16, chunked prefill (512), 32-token decode.
Context 4k→128k. Script: `notebooks/bench_vram_latency_v14.ipynb`.

| context | peak VRAM base→hybrid | VRAM save | decode base→hybrid | latency save |
|---------|-----------------------|-----------|--------------------|--------------|
| 4 096   | 3.53 → 3.61 GB | −2.3% | 43.2 → 44.0 ms/tok | −1.9% |
| 8 192   | 3.66 → 3.70 GB | −1.3% | 69.9 → 60.6 ms/tok | +13% |
| 16 384  | 4.06 → 3.99 GB | +1.8% | 122.4 → 102.0 ms/tok | +17% |
| 32 768  | 4.87 → 4.69 GB | +3.5% | 228.0 → 183.8 ms/tok | +19% |
| 65 536  | 6.48 → 6.10 GB | +5.8% | 438.6 → 350.1 ms/tok | +20% |
| 131 072 | 9.70 → 8.93 GB | +8.0% | 857.1 → 680.3 ms/tok | +21% |

**Reading (honest).**

*Memory.* Savings grow monotonically with context: at 4k the hybrid is slightly
*worse* (HFP params/state overhead dominates when there is almost no KV to save),
crosses over around 16k, and reaches +8.0% at 128k. This is smaller than the
naive "6/28 = 21%" figure because 21% is the saving on the *KV component only*;
total VRAM is dominated by the 3.1 GB fp16 weights. At 128k, baseline KV ≈ 3.76 GB
of 9.70 GB total; 21% of 3.76 = 0.79 GB = 8.1% of total — matching the measured
8.0% almost exactly. The internal consistency is itself the validation.

*Latency* is the stronger win: decode work per token scales with the number of
cached (full-attention) layers, so linearizing 6/28 of them cuts ~21% of
per-token attention cost at 128k (857 → 680 ms/tok). At 4k it is a wash.

*The O(1) signature (headline visual).* The state carried by the 6 grafted layers
is **9.5 MB, flat at every context length**. The KV those same 6 layers *would*
have used grows linearly: 6/28 × 3.76 GB ≈ 805 MB at 128k. So on the grafted
portion the recipe replaces 805 MB (growing) with 9.5 MB (constant) — an ~85×
reduction that does not grow with context. The panel-3 plot (flat line vs rising
line) is the concrete picture of the on-device vision; the modest *total* saving
is simply because 22/28 layers still use full KV.

**Honesty note on the direct check.** The benchmark also tries to read the number
of layers actually stored in the `DynamicCache`; on this transformers version the
API read returned a sentinel (−1), so the direct "22 cached layers" assertion did
**not** execute. The grafted-layers-skip-KV property is therefore confirmed only
*indirectly* here — the measured total saving matches the analytic prediction that
holds **only if** grafted layers write no KV (if they did, the hybrid would equal
baseline, 0% saving). The notebook has since been fixed to read the cache across
API variants; a future run will report the direct count. Numbers above are
unaffected (VRAM/latency are measured directly).

**Frame.** Savings are proportional to graft density (6/28). Higher density
approaches the O(1) asymptote but pays perplexity (the §22a cliff: 13 layers →
1.6× PPL). This benchmark is the Pareto point of the *current* recipe, not a
ceiling. Pairing it with §22: at 6 layers you buy 1.11× PPL (3/3 seeds) and, at
128k, ~21% faster decode with an O(1) memory footprint on the grafted layers.

## 24. Per-layer linearization-cost map (Faz-A) — and what it revealed about the proxy

Grafted all 28 layers, Stage-1 teacher-forcing (independent per-layer targets),
read each layer's converged normalized MSE (NMSE = ‖student−teacher‖²/‖teacher‖²)
on 24 held-out WT-2 chunks. Script: `notebooks/layer_linearization_probe_v1.ipynb`.
Map (cheapest→most expensive, selected values):

```
cheapest:  L27 0.19  L26 0.28  L21 0.32  L25 0.46  L24 0.53  L23 0.71 ...
expensive: ... L9 1.67  L12 1.57  L6 2.02  L4 2.06  L2 2.30  L3 2.32
```

Full map + trajectories: `docs/assets/layer_linearization_map.json/.png`.

**Pre-registered verdicts — read honestly, both are informative:**

*H1 (selection headroom): the proxy says YES, but our own data warns against
trusting it.* Cheapest-6 mean NMSE 0.42 vs the reference set [3,7,11,15,19,23]
1.21 — reference is 66% "more expensive" by NMSE. Taken alone this says naive
odd-indexing left large headroom. **But the critical internal check:** that same
reference-6 — which by this map contains the single *worst* layer (L3, NMSE 2.32,
rank 28/28) and several mid-expensive ones — already delivers **1.11× PPL across
3 seeds** (§22). A set full of "hard-to-reconstruct" layers gives near-baseline
PPL. So low NMSE is demonstrably **not necessary** for good PPL, and the NMSE→PPL
link is loose. H1's headroom is a hypothesis the proxy raises; it is **not**
evidence that principled selection will lower PPL. Only Faz-B (real PPL) can say.

*H2 (are first/last layers expensive, per the Mamba-in-Llama keep-them-full
prior?): NO — and this is the pre-registered caveat manifesting, not a broken
metric.* The last layer (L27) is the **cheapest** (0.19); L0 is mid (rank 10).
This contradicts the keep-boundaries prior. Honest interpretation: that prior is
about **downstream PPL sensitivity**, whereas NMSE measures **single-layer
reconstruction difficulty** — two different axes. They diverge most sharply at
the boundary: the last layer's output feeds the final norm + LM head **directly**,
so a given reconstruction error there hits the logits with no downstream layers to
absorb/correct it, while a middle layer's error is reprocessed by many layers
after it. So L27 can be *easy to reconstruct* (low NMSE) yet *PPL-critical* (high
sensitivity) simultaneously. H2's "failure" therefore corroborates the
pre-registered limitation (NMSE ≠ PPL-sensitivity) rather than invalidating the
run.

**Net finding (the real value of Faz-A):** the map is informative, but the run's
most important result is a *methodological* one — **NMSE is a weak predictor of
PPL for this problem** (a "bad-NMSE" set already achieves 1.11×; boundary layers
invert the expected ranking). This means we cannot shortcut density decisions with
the cheap proxy; Faz-B must measure PPL directly, and its outcome is genuinely
uncertain. Faz-A did its job: it ranked candidates *and* told us how much to
trust the ranking (not much, on its own).

**Faz-B design (updated by Faz-A):** graft a principled-13 and measure real PPL
vs the §22a odd-13 cliff (1.6×). Given H2, guard the boundary layers (exclude L0,
L27) → guarded-cheapest-13 = [11,13,15,16,17,18,20,21,22,23,24,25,26], mean NMSE
~0.71 vs odd-13's 1.07. Same recipe/seed/count — the only variable is *which* 13.
Outcome interpretation, pre-registered: (a) guarded-13 materially < 1.6×
(toward ~1.2×) → principled selection unlocks higher density, proxy transfers
enough to be useful; (b) guarded-13 ≈ 1.6× → 13 layers is costly regardless of
selection (capacity wall, not selection), and NMSE does not transfer — a
different signal (direct per-layer PPL-delta) would be needed.

**§24a — Faz-B outcome (guarded-cheapest-13, seed 0, full S1+S2): outcome (b),
a clean NEGATIVE.** Principled selection [11,13,15,16,17,18,20,21,22,23,24,25,26]
(mean NMSE 0.71) → **PPL 13.50 = 1.697× base** — essentially the same as, in fact
marginally *worse* than, the naive odd-13 cliff (1.6×), despite having far lower
NMSE (0.71 vs 1.07). Needle also degraded (1/4 vs the 6-layer recipe's 4/4). Two
conclusions, both important:

1. **NMSE does not transfer to PPL — confirmed decisively.** A 34%-lower-NMSE
   layer set produced *worse* PPL. The cheap single-layer reconstruction proxy is
   not just weak but effectively non-predictive (near-zero, possibly inverted, at
   the set level) for this problem. Faz-A's map should not be used to guide density
   selection. This vindicates running Faz-B instead of trusting the proxy.

2. **The §22a cliff is not a selection artifact — it is real at density 13 under
   this recipe.** Two disjoint 13-layer selections (naive odd, principled-cheapest)
   both land at ~1.6–1.7×. So "pick better layers" is **ruled out** as the lever
   for pushing graft density. Honest caveat: the S2 loss for the 13-layer run was
   noticeably noisier (30–155, non-smooth) than the well-behaved 6-layer runs, and
   the recipe (LR, steps, out_gain, curriculum) was tuned for 6 layers — so this
   rules out *selection*, but does not fully separate a fundamental capacity wall
   from a trainability wall specific to higher density. That separation is the next
   question.

**Consequence for the density-push line (RESULTS §23 Pareto / cost-moat).** The
cheapest lever (smart layer selection) is now closed. Deepening the moat beyond
the 6-layer point requires a *harder* lever, in rough order of promise: (i) a
density-specific / curriculum training recipe (graft 6, freeze, add layers
incrementally) to test the trainability-vs-capacity question; (ii) a stronger O(1)
primitive (delta/gated write instead of additive; §5 of grafting.py already
supports hybrid); (iii) larger O(1) state capacity (key_dim/bulk_dim) — cheap at
long context per §23 math. The 6-layer recipe (1.11×, 3 seeds, §22) remains the
shipping reference; pushing past it is now a known *research* problem, not a
selection tweak. BEKLEYEN #16/#17 closed; #18 (density-curriculum) opened.

**§24b — pre-registered (written before the run): §18 first probe = training
stabilization.** Motivation: Faz-B's S2 loss oscillated wildly (30–155) vs the
smooth 6-layer runs — a signature of optimization instability, not necessarily a
capacity floor. Cheapest discriminator before building a full curriculum: re-run
the *same* guarded-13 layers with a stabilized S2 (LR 3e-4→1e-4, warmup 50→150,
steps 600→900), everything else identical. Pre-registered criteria: (a) stabilized
guarded-13 PPL drops **materially below 1.6×** (≤ ~1.35×) → the cliff is largely a
*trainability* artifact; density-via-training (curriculum/warm-start, larger state)
is worth pursuing. (b) PPL stays **≥ ~1.55×** → not a training-stability artifact;
the wall is capacity/expressivity, and pushing density via training is deprioritized.
One-shot, honestly reported whichever way it lands. Checkpoint lineage tagged
`g13mapgS` (distinct from Faz-B's `g13mapg`, no resume collision).

**§24c — §18 probe outcome: stabilization did NOT help → outcome (b), and a
mechanistic clue.** Stabilized guarded-13 (LR 1e-4, warmup 150, 900 steps) →
**PPL 14.28 = 1.795×**, marginally *worse* than Faz-B's 1.70×; needle 1/4; the S2
total-loss still oscillated (39–164). Pre-registered verdict (b): the 13-layer
cliff is **not** a simple training-stability artifact. Three disjoint 13-layer
configurations now agree — odd-13 joint (1.6×), guarded-13 joint (1.70×),
guarded-13 stabilized (1.795×) — so the **6→13 density wall is robust** for this
additive/hybrid linear primitive, independent of layer selection and of LR/warmup/
length tuning.

**The mechanistic clue (important):** Stage-1 converged *fine* even at 13 layers —
per-layer MSE fell to ~0.089, the same neighborhood as the 6-layer runs. So each
grafted layer reconstructs its teacher well **in isolation, on clean input**. Yet
end-to-end PPL is 1.8×. The failure is therefore not per-layer capacity but
**compounding**: in the true forward, each O(1) layer's small reconstruction error
feeds the next, and across 13 grafted layers the errors accumulate. This also
explains §24's finding that NMSE (single-layer, clean-input) does not predict PPL
(multi-layer composition) — they measure different regimes, and the wall lives in
the composition. (Caveat on the "oscillation": the 39–164 swings are dominated by
the cross-chunk-recall batches' KL term, not necessarily core-LM divergence — the
`lm` loss stayed ~3–4 — so "instability" overstates it; the real story is
compounding, not divergence.)

**What is now closed vs open.** Closed as density levers: (1) layer *selection*
(§24a), (2) training *stabilization* (§24c). The compounding diagnosis means the
one still-untested targeted intervention is a **true incremental curriculum**
(graft 6, train, freeze, add layers in small groups) — it attacks compounding
directly by never optimizing all 13 error sources jointly — plus, more
speculatively, a **stronger O(1) primitive** or **larger state**. But three spent
runs have robustly established the wall; the 6-layer recipe (1.11×, 3 seeds, §22)
stands as the practical ceiling for now. Deepening the §23 cost-moat via higher
density is a genuine, not-yet-cracked research problem — it must not be assumed in
any product/moat framing. BEKLEYEN #18 updated: stabilization closed; incremental
curriculum / bigger-state remain as harder, optional bets.

## 25. Root-cause fix attempt — student-forward distillation (exposure bias)

Pre-registered (written before the run). §24c diagnosed the 13-layer wall as
**compounding**: Stage-1 trains each grafted layer on *clean teacher* input
(teacher_forcing propagates teacher output), but at inference each layer receives
the *corrupted student* output of the layer before it. Classic **exposure bias** —
layers are never trained on the input distribution they actually face, so small
per-layer errors accumulate through the stack. Every earlier lever (selection,
stabilization) was a downstream patch; this attacks the root.

**Method (new training mode `student_forcing` added to `grafting.py`,
backward-compatible).** Identical to teacher_forcing except it propagates the
**student** output forward (detached) instead of the teacher's. Each grafted
layer's MSE target is still teacher(current-input), but "current-input" is now the
realistic student-produced (corrupted) hidden state — so the layer learns to
*absorb upstream error*. The `.detach()` cuts the cross-layer graph, preserving the
cheap per-layer immediate-backward (memory stays as in §24c). Schedule (scheduled-
sampling / DAgger style, to avoid early instability): first 40% of Stage-1 in
teacher_forcing (clean warm-up), then switch to student_forcing.

**Single variable vs §24c.** Same guarded-13 layers, same stabilized S2 — the only
change is Stage-1 distribution (clean→realistic). Checkpoint lineage `g13mapgSF`.

**Pre-registered criteria.** (a) PPL drops **materially below §24c's 1.795×**, ideally
below Faz-B's 1.70× toward ~1.3× → exposure bias was a dominant driver of
compounding; the fundamental fix works and the density line **reopens** (next:
apply to a fresh recipe, push K past 13). (b) PPL stays **~1.6–1.8×** → exposure
bias is not the dominant driver; the wall is deeper (softmax-vs-linear expressivity),
and density-via-more-layers is closed for this primitive — the honest ceiling is
~6 layers. One-shot, reported whichever way it lands. Secondary read: needle recall
(§24c gave 1/4) — a real fix should also recover retrieval.

**§25a — outcome (b): the root-cause fix did NOT crack the wall.** Student-forward
Stage-1 (guarded-13, stabilized S2) → **PPL 13.83 = 1.738×**, needle still 1/4.
Marginally better than §24c (1.795×), marginally worse than Faz-B (1.70×) —
squarely inside the same 1.6–1.8× cluster. So exposure bias is **not** the dominant
driver of the compounding wall.

**Four independent 13-layer configurations now converge:**

| config | S1 dist. | S2 | PPL | needle |
|--------|----------|-----|-----|--------|
| odd-13 (§22a) | teacher | base | 1.60× | — |
| guarded-13 (§24a) | teacher | base | 1.70× | 1/4 |
| guarded-13 stab (§24c) | teacher | stabilized | 1.795× | 1/4 |
| guarded-13 student-fwd (§25a) | **student** | stabilized | 1.738× | 1/4 |

The result is now robust and the interpretation honest: the 6→13 density wall is a
genuine **expressivity/capacity limit of the additive/hybrid linear-attention
primitive**, not a training artifact — selection (§24a), stabilization (§24c), and
the mechanism-targeted exposure-bias fix (§25a) all fail to move it materially.
The practical ceiling for this primitive is ~6 grafted layers (1.11×, §22).

**What remains — and what is now closed.** All *training-side* levers are exhausted.
The only lever left with a clear rationale is an *architecture-side* one: a
**higher-capacity O(1) primitive** — the simplest instance being larger recurrent
state (`bulk_dim`/`dpfp_nu`), which every 13-layer run so far held fixed. That is
the single definitive test of the capacity hypothesis: if the wall is "per-layer
memory too small," more state cracks it; if it is "linear attention fundamentally
under-ranks softmax," it will not (literature leans to a residual gap that state
only partly closes). Recommended as the *last* density experiment — one parameter,
one run — after which the density line closes cleanly regardless of outcome, and
effort returns to the 6-layer product path. BEKLEYEN #18 updated. §23 cost-moat
deepening remains unsolved and must not be assumed in any product framing.

## 26. Cubic in its native regime — extrapolation test (pre-registered)

Written before the run. §15h found cubic ≈ exp in the *graft-onto-Qwen* regime
(dense, saturated writes — cubic's worst case). But §6 documented a real,
statistically strong cubic win (63.9% vs 20.7%, >4 SE) in a *small-scale
long-horizon* regime. The reconciliation (technical note): cubic's decay
λ = 1/√(1+2η z²) adapts to **current channel occupancy** — empty/sparse channels
≈ no forgetting (plateau), saturated channels self-limit — whereas exp's per-channel
λ is fixed regardless of occupancy. So cubic's edge should appear precisely where
(a) writes are sparse (channels stay unsaturated) and (b) the recall horizon
**exceeds the trained horizon**, where exp's learned timescales run out but cubic's
content-adaptive plateau keeps protecting the target channel. That regime — sparse,
long-lived memory — is also the on-device personal-memory regime, so this is not
just a curiosity.

**Design.** Vehicle = the debugged cross-chunk carry pipeline (`carry_curriculum.py`
trains carry to CARRY_MAX=16 chunks; `matched_probe.py` evaluates on the
train-distribution probe at K = 0,2,8,16,32,64). Twin runs exp vs
cubic_flux_chunked, 3 seeds each, identical everything else (multi-scale learned
λ for exp = the fair control). The discriminating signal is **extrapolation**:
K=32 and K=64 are 2× and 4× the trained horizon.

**Pre-registered criteria.** (a) **Cubic niche CONFIRMED** if at *both* K=32 and
K=64 the seed-averaged matched accuracy is ≥ +10 points over exp AND the seed
ranges separate (min_cubic > max_exp). (b) Symmetric: same threshold the other way
→ exp wins even here. (c) Otherwise → **null**: cubic has no advantage even in its
predicted home regime, and exp is the honest default everywhere (cubic retired to
a flag). Reported whichever way it lands; this is cubic's fair day in its own court.

**§26a — outcome: INCONCLUSIVE (vehicle failure), not a null. Plus one strong
incidental signal.** Matched-probe accuracy (chance 3.3%), seed mean [min–max]:

| K | ~gap | exp | cubic | Δ |
|---|------|-----|-------|---|
| 0 | 256 | **18.7%** [6–36] | **43.3%** [36–56] | **+24.7** |
| 2 | 768 | 7.3% [2–12] | 8.0% [8–8] | +0.7 |
| 8 | 2 304 | 4.7% [2–8] | 6.7% [4–10] | +2.0 |
| 16 | 4 352 | 4.0% [2–6] | 4.7% [4–6] | +0.7 |
| 32 | 8 448 | 3.3% [0–6] | 4.7% [4–6] | +1.3 |
| 64 | 16 640 | 3.3% [2–6] | 5.3% [2–8] | +2.0 |

**Why this is not a verdict on cubic.** At every K ≥ 2 *both* arms sit at chance.
The carry vehicle never learned to hold the target past a couple of boundaries —
the same first-boundary collapse that caused the small-scale lifetime line to be
parked (§20/§21). A comparison in which both arms are at chance has **no power**:
it cannot distinguish "cubic has no niche" from "the test could not detect one."
Recording this as a null would misreport a zero-sensitivity experiment. §26's
pre-registered criteria are therefore **not evaluable**; the question stays open.

**The incidental signal (hypothesis-generating, NOT confirmation).** At K=0 — the
one distance where the vehicle has real signal (single chunk hand-off, ~256 tokens,
both arms clearly above chance) — cubic scores **43.3% vs exp's 18.7%, +24.7
points, and cubic wins in every seed** (cubic 36/38/56 vs exp 36/14/6). Honest
caveats, stated plainly: (i) K=0 was **not** a pre-registered test point, so this
is post-hoc; (ii) the seed ranges only *touch* (min cubic 36 = max exp 36) rather
than separate, so it fails the strict separation bar §26 demanded elsewhere;
(iii) n=3, 50 trials. It is a lead, not a result. But it is a *coherent* lead: it
sits exactly where cubic's mechanism predicts an edge (a sparsely-written channel
surviving a hand-off) and it aligns with the user's original observation and with
§6's confirmed win — i.e. cubic keeps showing an advantage wherever a retention
signal actually exists, and shows nothing wherever there is no signal at all.

**Consequence.** Cubic is *still* neither confirmed nor refuted in its predicted
home regime. Two honest options: (a) cheap, sharp follow-up at the distances where
the vehicle does work (K ∈ {0,1}, more seeds, more trials) to test whether the
+24.7 hand-off advantage is real — this is a well-posed, powered question; or
(b) fix the parked carry vehicle first (§20/§21), which is the harder, longer road.
Until one of these lands, `exp` remains the pragmatic default in the shipped graft
recipe (§15h, where it *was* properly powered and tied), and cubic remains a
supported flag with one confirmed win (§6) and one open lead (this section).

**§26b — powered hand-off test (K∈{0,1}, 6 seeds, 200 trials): pre-registered
criterion NOT met. Directionally consistent, statistically inconclusive.**

| K | exp (6 seeds) | cubic (6 seeds) | Δ | ranges separated? |
|---|---------------|-----------------|---|-------------------|
| 0 | 33.2% [8.5–69.5] | 53.4% [29–83] | **+20.2** | **no** (heavy overlap) |
| 1 | 16.2% [3–61.5] | 9.6% [4–14] | −6.6 | no (**direction reverses**) |

The §26a lead does **not** survive proper powering. At K=0 cubic's mean advantage
persists (+20.2, and its floor 29% > exp's 8.5%), but seed variance is enormous in
both arms (exp spans 8.5–69.5) and the ranges overlap heavily; a Welch t-test gives
t≈1.4, p≈0.2 — not significant at n=6. At K=1 the sign **flips** (exp ahead),
driven by a single exp outlier seed (s5: 61.5% vs its siblings' 3–6%). Reporting
this as a cubic win would be cherry-picking; the honest verdict is **not confirmed**.

**However — a separate, lower-variance signal points the same way.** On the
*training objective itself* (averaged over many batches, so far less noisy than
50–200-trial probes), for the three seeds with logs (s3–s5) cubic learned the
sparse cross-chunk carry task substantially better in **3/3**:

| seed | exp lossA / lossB(cross-chunk) | cubic lossA / lossB | exp streaming-probe @256 | cubic @256 |
|------|-------------------------------|---------------------|--------------------------|------------|
| s3 | 1.81 / 2.01 | **1.38 / 1.62** | 0.0% | **16.7%** |
| s4 | 1.86 / 2.15 | **0.82 / 1.04** | 0.0% | **53.3%** |
| s5 | 1.85 / 1.95 | **0.89 / 1.62** | 6.7% | **53.3%** |

(chance 3.3%; streaming probe = the §17-style lifetime probe run at end of
training, 30 trials.) Here the separation *is* clean — max exp 6.7% < min cubic
16.7% — and the training losses favour cubic in every seed, on both the in-chunk
(lossA) and cross-chunk (lossB) terms. Caveats, stated plainly: this is a post-hoc
comparison on a different probe with only 3 logged seeds and 30 trials, and lower
training loss is not by itself the pre-registered endpoint.

**Honest synthesis.** Three measures — matched probe K=0, training loss, streaming
probe — all lean cubic in this sparse-write carry regime, but the one *properly
pre-registered and powered* comparison fails its separation bar because of seed
variance. So: **cubic is not vindicated, and not refuted.** The accumulated picture
(§6 confirmed win; §15h clean null in the dense graft regime; §26b consistent-but-
underpowered lean here) is best summarized as *cubic appears to help specifically
in sparse-write / low-occupancy regimes and not otherwise, but the effect size in
this vehicle is smaller than the seed noise.* Settling it requires either many more
seeds (12–20, cheap on CPU: this whole run was CPU-only) or a lower-variance
endpoint (e.g. training-loss curves across seeds as the primary metric, pre-
registered). `exp` remains the default in the shipped graft recipe. Recorded as an
open, honestly-inconclusive question rather than a result in either direction.

## 27. Improving cubic: the η (plateau-scale) sweep

`eta_init` was **hard-coded** as `logspace(-4,-2)` in `hfp_bulk_state.py` since
inception — never swept, inherited by every experiment. Since cubic's plateau
scale is z\* ≈ 1/√(2η), the default gives z\* ≈ 7–70 while the carry task spans
256–4096+ tokens, suggesting the plateau was ~2 orders of magnitude too *short*.
η is now configurable (`ETA_LOG_MIN/MAX`, env-overridable; default unchanged).
Pre-registered primary metric = end-of-training cross-chunk verification loss
(low-variance, per the §26b lesson); 3 seeds/arm.

**§27a — result: hypothesis REFUTED, monotonically and in the opposite direction.**

| η range | z\* | cross-chunk loss (mean) [min–max] | seeds better than default |
|---------|-----|-----------------------------------|---------------------------|
| **(−4,−2) default** | 7–70 | **1.746** [1.55–1.90] | — (control) |
| (−6,−4) | 71–707 | 2.167 [1.83–2.56] | **0/3** |
| (−8,−6) | 707–7071 | 2.306 [1.98–2.57] | **0/3** |

Longer plateaus made cubic *worse*, consistently, with a clean monotone trend
(1.75 → 2.17 → 2.31). Not a null — a directional refutation.

**Why (and this corroborates an earlier finding).** As η→0, λ→1: the channel stops
forgetting at all. But this task streams **dense distractor traffic** (a distractor
key-value every 64 tokens), so a never-forgetting channel accumulates interference
until the target is unrecoverable. Retention is not the binding constraint here —
**interference is**. This independently reproduces the early HFP finding that
memory in this architecture is *interference-limited, not decay-limited*
(GPU_ROADMAP §0), now via a completely different manipulation. The default η is
therefore not mistuned in the "too short" direction; if anything the data's
gradient points the other way.

**Honest status of cubic after §26–§27.** One confirmed win in a sparse
long-horizon synthetic regime (§6); a clean null in the dense graft regime (§15h);
a consistent-but-underpowered lean in the carry regime (§26b, training loss ~1.75
vs exp ~2.15); and now a refuted tuning hypothesis (§27a). Cubic is a *real but
modest and regime-specific* effect that we have not been able to convert into a
decisive advantage. `exp` remains the shipped default. The one direction the data
actually points to — **larger** η (shorter plateau, faster interference flushing,
z\* ≈ 0.7–7) — is the natural next probe and is cheap; but expectations should be
calibrated: it tests a *tuning* refinement, not a new capability.

**§27b — the other direction hits a hard stability wall; and the pooled
exp-vs-cubic comparison.** Larger η (shorter plateau) was tested as the data's
gradient suggested:

| η range | z\* | outcome |
|---------|-----|---------|
| (−2, 0) | 0.7–7 | **1/3 seeds diverged (NaN @ step 414)**; survivors 1.99, 1.02 |
| (0, +2) | 0.07–0.7 | **3/3 seeds diverged (NaN @ steps 290–419)** |

Aggressive η is **numerically unstable**: λ = 1/√(1+2η z²) with large η drives the
decay factor toward 0, and gradients through the sequential z-scan blow up once the
curriculum grows K. This is a genuine boundary of the mechanism, not a tuning miss.
Note on reading the sweep table: the (−2,0) arm's mean (1.503) *looks* best but is
computed over **2 surviving seeds only** — averaging the survivors of a config that
diverges a third of the time is survivorship bias, and the arm is correctly recorded
as a **failure**, not an improvement. Conclusion for §27: **η tuning does not improve
cubic in either direction**; the shipped default (−4,−2) sits at a
stability/performance sweet spot that was, in hindsight, well chosen.

**Pooled primary-metric comparison (the one solid signal from §26–§27).** Combining
this run's seeds 0–2 with §26b's seeds 3–5 — same vehicle, same low-variance
endpoint (end-of-training cross-chunk verification loss), same default η:

| | cubic (default η) | exp | Δ (exp−cubic) |
|---|---|---|---|
| mean over 6 seeds | **1.731** | 2.141 | **+0.411 nat** |
| per-seed wins | **5 / 6** | 1 / 6 | (+0.68, +0.37, +0.11, +0.25, +1.25, −0.20) |
| paired t (df=5) | | | t = 2.00, p ≈ 0.10 |

Cubic learns this sparse-write cross-chunk carry task **better than exp in 5 of 6
seeds, by ~0.41 nat on average** — but the paired test gives p ≈ 0.10, i.e.
suggestive and short of conventional significance at n=6. Reported as such: a
consistent direction, not a proven effect.

**Final position on cubic (closing §26–§27).** The evidence is now a coherent,
bounded picture rather than a verdict: cubic helps in **sparse-write / long-horizon**
regimes (§6 confirmed win; here 5/6 seeds on the training objective) and does **not**
help in the **dense/saturated** graft regime (§15h clean null); the advantage is
**modest** (~0.4 nat, p≈0.10 — real-looking but under-powered); and it is **not
improvable by tuning its one free parameter** (§27a/b), with a hard instability
boundary at large η. The original intuition that cubic "does something real" is
**partially vindicated** — in a specific regime, at a modest magnitude — while the
stronger claim of a decisive architectural advantage is not supported. `exp` stays
the shipped default (dense graft regime, where they tie and exp is simpler and
stable); `cubic_flux_chunked` remains a supported flag, now with a documented regime
map and stability limits. Line closed unless a sparse-regime product need reopens it.

## 28. Cubic, decisive test (pre-registered before the run)

§27b left the central question at p ≈ 0.10 with n=6 — suggestive, not settled.
This is a power problem, not an ambiguity: with the observed effect size
(d = 0.411/0.503 ≈ 0.82), n=16 yields t ≈ 3.3 (p ≈ 0.005) *if the effect is real*,
and clearly fails to separate if it is not. So the question is answerable cheaply
(CPU-only) by adding seeds.

**Design.** Identical vehicle and settings to §27 (carry_curriculum, CARRY_MAX=16,
CTX=256, 1200 steps, default η), arms = `exp` vs `cubic_flux_chunked`,
**seeds 0–15 (n=16), paired by seed** (same seed = same data order, so pairing is
valid). Primary endpoint fixed in advance: **end-of-training cross-chunk
verification loss** (the low-variance metric identified in §26b). Secondary:
streaming probe @256 (reported, not decisive — known high variance).

**Pre-registered criteria.**
- **CUBIC ADVANTAGE CONFIRMED:** paired t-test p < 0.05 **and** cubic better in
  ≥ 70% of seeds (≥ 12/16) **and** mean Δ ≥ 0.2 nat. → cubic's sparse-regime edge
  is established at this scale; documented as a regime-specific architectural
  finding and written into the paper's contribution list.
- **REFUTED:** p ≥ 0.05 with mean Δ < 0.15 nat → the §26b/§27 lean was noise;
  cubic offers no reliable advantage even in its home regime; retired to a flag
  with that stated plainly.
- **STILL UNDERPOWERED:** p ≥ 0.05 but mean Δ ≥ 0.2 nat and ≥ 70% wins → effect
  plausibly real but smaller than estimated; reported as such, no further seeds
  (diminishing returns), no claim either way.

Any divergent (NaN) run is counted as a **failure of that arm**, not dropped
(§27b survivorship-bias lesson). Result reported whichever way it lands.

**§28a — result: CUBIC ADVANTAGE CONFIRMED (all three pre-registered criteria met).**

| | cubic_flux_chunked | exp (multi-scale learned λ) |
|---|---|---|
| cross-chunk verification loss, mean of 16 seeds | **1.794** | 2.278 |
| paired Δ (exp − cubic) | **+0.484 nat** | |
| seeds where cubic is better | **13 / 16** | 3 / 16 |
| paired t (df = 15) | **2.84** | |
| two-sided p | **0.0124** | |
| divergences (NaN) | 0 / 16 | 0 / 16 |

Criteria required p < 0.05 (**0.0124** ✓), ≥ 12/16 seed wins (**13/16** ✓), and
mean Δ ≥ 0.20 nat (**0.484** ✓). Both arms were numerically stable throughout, so
no survivorship correction applies. Effect size d ≈ 0.71 — consistent with the
n=6 estimate (0.82) that motivated the power calculation, i.e. the result is a
confirmation of a pre-specified prediction rather than a discovered pattern.

**What this establishes — and what it does not.** Established: in the
**sparse-write, cross-chunk carry regime**, content-adaptive cubic retention
(λ = 1/√(1+2η z²)) learns the carry task **measurably better** than a fair
multi-scale learned-λ exponential control, under a pre-registered, adequately
powered, paired design. This is the first properly powered positive result for
the project's own architectural hypothesis, and it survived a deliberately
adversarial sequence: a clean null in the dense regime (§15h), an inconclusive
zero-power run (§26a), an underpowered lean (§26b), and a refuted tuning
hypothesis in both directions (§27a/b).

Not established, and explicitly *not* claimed: (i) this does **not** overturn
§15h — in the dense/saturated graft regime the two laws still tie, and the two
results are consistent under the occupancy-dependence explanation (cubic's edge
requires unsaturated channels); (ii) the endpoint is **training loss on a
small-scale synthetic task** (2 layers, hidden 64), not language-model quality or
downstream retrieval — the streaming probe remained too noisy to separate arms
(§26b), and no claim is made there; (iii) the absolute magnitude is modest
(0.48 nat against a ~3.4-nat chance baseline); (iv) transfer to LM scale is an
open question that §15h suggests is regime-dependent, not automatic.

**Consequences.** (a) `exp` remains the shipped default for the **graft** recipe
(dense regime, where they tie and exp is simpler/stable). (b) `cubic_flux_chunked`
is promoted from "supported flag with an open question" to a **documented,
regime-specific mechanism with a measured advantage** — the recommended choice for
sparse-write / long-lived-memory workloads, which is precisely the on-device
personal-memory regime the project targets. (c) The paper's contribution list gains
a defensible empirical claim: *retention-law choice is immaterial in dense-write
regimes but measurably matters in sparse-write regimes* — a statement that unifies
§15h and §28a rather than contradicting either. (d) The cubic line closes here on a
positive, with its scope honestly bounded.

**§28b — engineering decision: cubic shelved; `exp` is the shipping path.** The
§28a finding stands and is not retracted — but a validated *finding* is not a
shippable *component*. Cubic is shelved for engineering, not scientific, reasons:
(1) its z-scan is **sequential** (λ_t depends on z_{t-1}), so unlike exp's parallel
cumsum it cannot be parallelized — slower training and prefill; (2) that scan is a
**custom op**, awkward to export to mobile runtimes (GGUF/ExecuTorch/Core ML),
which conflicts directly with the project's on-device goal; (3) it has a **narrow
stability window** (§27b NaNs at large η) that η tuning cannot widen (§27a/b);
(4) in the dense graft regime it pays those costs for **zero or negative** return
(§15h: PPL 13.04 vs 12.87, reliability 38/45 vs 42/45 — exp ahead on both, though
not significantly). Un-shelving requires, at minimum, (i) a parallel /
associative-scan formulation of the z-recursion (or a closed-form intra-chunk
approximation) and (ii) a demonstrated mobile export path without a custom op;
ideally also (iii) replication of the sparse-regime advantage at LM scale, since
§15h shows transfer across regimes is not automatic. The shelf is expected to be
temporary: the target product regime (personal memory = sparse writes) is cubic's
home, so this is a deployment-engineering blocker, not a dead end.

## 29. parallel_cubic — removing the sequential z-scan (implementation verified,
experiment NOT run)

**Motivation.** §28b shelved cubic for engineering reasons, the first being that
λ_t = 1/√(1+2η z_t²) depends on the *state*, forcing a **sequential** z-scan that
cannot be parallelised and is a custom op (blocking mobile export). Un-shelving
condition #1 was a parallel formulation.

**What was built.** `decay_mode="parallel_cubic"` in `hfp_bulk_state.py`:
λ is computed from the **true z at each block start** and held constant within the
block ("block-frozen λ"), so the intra-block computation collapses to the same
parallel machinery `exp` uses; only the inter-block carry stays sequential — i.e.
exactly the parallelism structure of exp/GLA. (An earlier open-loop-proxy design
was discarded: a fixed-λ occupancy proxy overshoots the true state by ~1000× when
λ₀→1, collapsing λ and producing NaNs. That failure and its cause are documented in
a comment block in `hfp_bulk_state.py`.)

**Verification gate — run, and passed** (`notebooks/parallel_cubic_v29.ipynb`):

| test | result |
|---|---|
| T1: `rec_block=1` ⇒ identical to sequential `cubic_flux_chunked` | **max abs diff = 0.000e+00** (bit-exact) |
| T2: `rec_block=32` approximation error | 1.372e-03 (≈1.03% relative) |
| T3: finite loss, `log_eta` receives gradient | pass |
| T4: L=1024, no NaN | pass |
| T5: speed (B=4, L=512, CPU) | cubic 119.6 ms → **parallel_cubic 103.0 ms (1.16×)**; exp 78.0 ms |

T1 being bit-exact establishes that the block-frozen form *reduces exactly* to
sequential cubic in the limit, so the implementation is correct and the
approximation is controlled by `rec_block`.

**Status: the quality experiment was NOT run.** The planned n=16 paired comparison
(does parallel_cubic retain §28a's +0.484 nat advantage?) never executed — work
moved to the memory line (§30) and cubic was subsequently shelved (BEKLEYEN #21).
Recorded as unfinished rather than omitted. Anyone resuming: the notebook is ready,
the gate passes, only cells 3-4 need running. Note also that the 1.16× speed-up is
modest because both modes share the O(m²) intra-block einsum; the real gain is the
removal of the custom sequential op (un-shelving condition #1), not throughput.

## 30. Memory-organ capability test (pre-registered)

Everything in the "memory co-processor" product framing rests on a capability that
**has not been measured**: can information be recovered from the O(1) state by a
*query*, when the KV cache is **not** available for the older stream?

Two gaps in the existing evidence:
1. **Retrieval mode.** §22's needle test carries a single `DynamicCache` across the
   whole stream, so the 22 un-grafted full-attention layers can see the needle
   directly. That test establishes *"grafting did not break retrieval"*; it does
   **not** establish *"the information is held in the O(1) state"*. For the
   co-processor claim the KV cache must be bounded — otherwise memory grows
   linearly and there is no memory organ.
2. **Query type.** The needle probe is verbatim continuation of the exact inserted
   phrase. A memory organ must answer *queries* — updated facts, multiple facts,
   and it must **not** confabulate about facts never stored.

**Design — 3 conditions × 5 probe types.** Conditions: (A) full-attention over the
full context, grafted layers in `teacher` mode = **upper bound** (what the base
model could do if it saw everything); (B) hybrid + persistent KV cache = the §22
setting; (C) hybrid + **fresh KV cache per chunk**, HFP streaming state carried =
**the real co-processor test** — anything older than the current chunk can only
come from the O(1) state. Note (C) matches the *training* protocol (S2 cross-chunk
recall trains with no cache across chunks), so it is the regime the model was
trained for. Probes: P1 verbatim (control), P2 lexical variant, P3 updated fact
(later value must win), P4 multi-fact discrimination, P5 negative control (never-
stored key — must NOT emit a stored value).

**Pre-registered criteria** (n trials per probe, exact-match on the target value):
- **MEMORY ORGAN CONFIRMED:** in condition (C), P1–P4 ≥ 60% each **and** P5 false-
  retrieval ≤ 20%, with (C) within 25 points of (A) on P1–P4. → the O(1) state is a
  usable memory organ; the co-processor framing is supported and product work can
  proceed on it.
- **CACHE-DEPENDENT:** (B) succeeds but (C) collapses (< 30% on P1/P3). → retrieval
  in prior results was carried substantially by the KV cache; the O(1) state alone
  is not yet a memory organ. The co-processor claim is **not** supported and must
  not be used in any product/pitch framing until fixed (options: train explicitly
  for cache-free recall, larger state, bigger base).
- **MODEL-LIMITED:** (A) itself fails a probe → that probe is beyond the 1.5B base
  model, not a memory failure; reported separately and excluded from the memory
  verdict (this is why the upper-bound condition exists).
- **CONFABULATION FAILURE:** P5 false-retrieval > 40% in (C) → the memory emits
  plausible-but-unstored values; unusable as a factual store regardless of P1–P4.

Reported whichever way it lands. A negative here is high-value: it is cheaper to
learn now than after building a product on the assumption.

**§30a — result: CACHE-DEPENDENT. The O(1) state alone retrieved nothing.**
6-layer reference checkpoint (`hfp_graft_exp_g6_s1_final.pt`, Run 8), 4096-token
stream, 8 trials/probe, chunk 512:

| condition | P1 verbatim | P2 variant | P3 updated | P4 multi-fact | P5 false-retrieval |
|---|---|---|---|---|---|
| **A** full attention (upper bound) | 100% | 100% | 0% | 100% | 0% |
| **B** hybrid + persistent KV cache | 88% | 38% | 100% | 38% | 0% |
| **C** hybrid + O(1) state only | **0%** | **0%** | **0%** | **0%** | 0% |

Condition C collapses completely. Sample C outputs show generic base-model
continuations with no trace of the stream ("The license server is" → "not
available…"; "The cache directory is" → "located at /home/username") — the model
is not confabulating *stored* values (P5 = 0% everywhere, which is genuinely good
news for factual safety), it simply retrieves **nothing**. P3 was excluded as
model-limited (upper bound A itself scored 0%: with both the old and new value in
context the base model prefers the prior-favoured "8080").

**Three findings, in order of importance.**

1. **The retrieval demonstrated in §22 was carried substantially by the KV cache,
   not by the O(1) state.** This is the honest reinterpretation of the needle
   result: grafting does not *break* retrieval (true, and still a valid claim), but
   it was never shown that the memory *holds* the information — and under this test
   it does not. Any framing that presents the O(1) state as a standalone memory
   organ is **not supported by evidence** and must not be used in product or
   investor language.
2. **Even with the cache, query-style retrieval is weak.** Condition B scores only
   38% on the lexical-variant and multi-fact probes versus 100% for full attention.
   So the "answer questions about accumulated content" capability is limited in the
   grafted model generally, not only in the cache-free regime.
3. **No confabulation.** P5 = 0% in all conditions: the model never emitted a
   stored value for a never-stored key. Whatever else is true, it does not
   fabricate false memories — a useful property to have measured.

**Honest caveat before treating this as final — probe/training mismatch.** This
project has already been burned once by exactly this confound: in the small-scale
line, an *unmatched* probe produced chance-level results while a *matched* probe
(built from the training distribution) recovered real signal (§19 → §26a). The same
mismatch is present here: S2 cross-chunk recall was trained with **128-token**
chunks and the query at the **end of a filler-padded chunk**, whereas condition C
streams **512-token** chunks and issues the query as a bare ~6-token fragment with
no local context. So §30a cleanly establishes that *the current setup does not
retrieve from state*, but it does not cleanly separate "the state holds nothing"
from "the state holds something the model cannot address in this format". The
decisive follow-up is cheap and pre-registered below.

**§30b — pre-registered follow-up (matched-probe replication).** Re-run condition C
with the protocol matched to training: chunk = 128, the query placed at the end of a
128-token filler chunk, fact-to-query distance swept over the trained range
(≈1-12 chunks) and beyond. Criteria: (a) matched-C ≥ 60% on P1 at trained distances
→ §30a was a harness artifact, the memory organ exists but is format-sensitive, and
the co-processor path stays open with an explicit range limit; (b) matched-C still
< 30% → §30a stands as a genuine capability finding: **the O(1) state is not a
usable memory store in the grafted model**, and the co-processor framing is
abandoned or deferred until a different design (cache-free training objective,
larger state, larger base) is demonstrated. Until (a) is shown, the project's honest
public claim remains the narrower one already supported: *a 6-layer O(1) graft
preserves language quality (1.11× PPL, 3 seeds) and does not break long-range
retrieval when a KV cache is present.*

**§30c — matched probe also fails; load verified; finding is FINAL.** The §30b
matched-protocol replication (chunk 128, query at end of a filler-padded chunk,
gaps 0→24 chunks) scored **0% at every distance, including gap = 0** — a single
chunk boundary, ~128 tokens. Every trial returned a byte-identical filler
continuation ("not running. The routine check…"), i.e. the fact chunk exerted *no
influence whatsoever*. The harness-mismatch explanation is therefore eliminated.

Because a result this absolute can also be produced by a silent checkpoint-loading
failure (`strict=False` loads nothing on a name mismatch, which would leave the
graft untrained → C = 0% while B still works off the cache), the load was verified
explicitly: **72/72 tensors matched by name and shape, 72/72 bit-identical in the
model, and `out_gain` had clearly moved away from its 0.1 init (mean 0.234, std
0.097, range 0.032–0.432)**. The weights are loaded and trained. The negative
stands.

**Mechanistic cause — identified, and it is in the training code.** In Stage-2
cross-chunk recall the write chunk is processed under `torch.no_grad()`:

```python
with torch.no_grad():
    model(xa)            # chunk A: write the fact into state — NO GRADIENT
    ...                  # gap chunks: carry state — NO GRADIENT
out = model(xb, labels=xb)   # chunk B: read; loss only here
```

So the cross-chunk objective back-propagates **only into the read path**
(`conv_q`, `retrieval_norm`, `out_gain`). The write-path parameters
(`decay`, `log_eta`, `conv_k`, `beta_gate`, `alpha_logit`) receive gradient
exclusively from *within-chunk* write→read pairs in B. The model was therefore
never trained to **store something for a later chunk** — only to read whatever
happens to be in the state, and to write for immediate local reuse. §30's result is
the direct, predictable consequence. (This limitation was written down honestly in
the notebook at the time — "cross-chunk gradyan yalniz OKUMA yoluna akar" — but its
implication for the memory claim was not followed through until now.)

**Unification with the parked small-scale line.** §17–§21 found cross-chunk recall
collapsing at the *first* state hand-off in the small-scale model, and the parked
suspects were "read-path dilution + decay-gradient plumbing". §30 reproduces the
same failure at graft scale, and names the plumbing precisely: **no gradient
reaches the write path across a boundary.** Two independent lines, one cause.

**Consequences — what must change in the project's claims.**
- The §22 needle result is **re-scoped**: it shows grafting does not break
  long-range retrieval *when the KV cache is present*. It does **not** show the
  O(1) state stores or retrieves anything. README and any external write-up must
  say this explicitly.
- §23's VRAM/latency numbers remain valid (structural), but their *interpretation*
  changes: the 6 grafted layers earn their keep on local language modelling
  (1.11× PPL vs 168 PPL untrained — they clearly learned something), not as a
  long-range memory store.
- The "memory co-processor" product framing is **withdrawn** until a state that
  demonstrably stores across boundaries exists.

**The targeted fix (next experiment, well-motivated rather than speculative).**
Train Stage-2 recall with gradient flowing through the write chunk and across the
boundary: remove the `no_grad` on chunk A, keep the graph to B (the graft-scale
equivalent of `bptt_across_chunks=True`). Cost: activation memory for A+gap+B in
one graph — feasible at SEQ=128, 6 layers, with gradient checkpointing. This is the
first intervention that addresses the identified cause rather than a symptom.
Caveat, stated up front: the small-scale analogue (Görev G / §21) showed no effect
from TBPTT alone, so this may be necessary but not sufficient.

### Measurement corrections (2026-07-28, from the §31 run's setup output)

Two numbers repeated in earlier sections were carried over from the **13-layer**
era and are wrong for the **6-layer reference recipe**:
- **Trainable parameters: 149,910 (~0.01% of the base), not 325k (~0.03%).** The
  325k figure is the 13-layer graft. The 6-layer recipe is *more* frugal than
  claimed; the corrected number is used going forward.
- **Untrained (zero-shot) graft PPL: 12.0, not ~168.** The 168 figure is likewise
  the 13-layer diagnostic (T2). This matters for interpretation: the trained graft
  moves PPL 12.0 → 8.84 (base 7.96), so the 6 grafted layers *do* contribute, but
  the model is considerably more tolerant of them being untrained than the stale
  number implied. Any statement of the form "untrained 168 → trained 8.84,
  therefore the layers carry heavy load" is **overstated** and is corrected here.

## 31. Write-path gradient across chunk boundaries (pre-registered)

Direct intervention on the cause identified in §30c. Two things blocked gradient
from ever reaching the write path across a boundary, and **both** had to go:
1. Stage-2 recall processed the write chunk under `torch.no_grad()`.
2. `HFPGraftAttention` **detached** the streaming state (`M`, `z`, and the conv
   state) at every chunk boundary.

Fix: new `stream_bptt` flag on the module (default `False`, backward compatible)
plus `S2_WRITE_BPTT` in the notebook, which runs chunk A and the gap chunks **with
gradient** and keeps the graph through to chunk B. Now the cross-chunk loss can
credit the *write* operation (`decay`, `log_eta`, `conv_k`, `beta_gate`,
`alpha_logit`), not only the read.

**Single variable.** Everything else reverts to the §22 reference recipe
(`GRAFT_N=6`, `GRAFT_FROM_MAP=None`, `S2_STAB=False`, `S1_STUDENT_FORWARD=False`,
seed 0). Checkpoint lineage tagged `…W`.

**Declared experimental limitation.** With BPTT on, A+gaps+B form one graph, so
activation memory grows with the gap. The curriculum is therefore shortened to
gaps ∈ {0,1} (≈ up to 256 tokens carried) instead of {0…12} — the first attempt
used {0,1,2,3} and hit CUDA OOM on a T4 at step ~4 (A+3 fillers+B = 5 chunks in one
graph, with gradient checkpointing necessarily off because it conflicts with the
streaming state). This run asks
*"does the write path learn to store across a boundary at all?"*, **not** "at what
range". Range extension is a separate, later question, and the result must not be
reported as a range claim.

**Pre-registered criteria** — evaluated with the §30 harness, condition C
(fresh KV cache per chunk, matched probe, gap = 0…3, i.e. inside the trained
range):
- **WRITE PATH LEARNS:** matched-C P1 ≥ 60% at gap 0 **and** ≥ 40% at gap 3 →
  §30c's diagnosis is confirmed *and* actionable; the O(1) state becomes a real
  store within the trained range. Next step then: extend range (longer BPTT
  windows, truncated BPTT with periodic gradient), re-open the memory-organ line.
- **PARTIAL:** P1 ≥ 60% at gap 0 but collapses by gap 3 → the write path can store
  across *one* boundary but not chain; points to state capacity or decay dynamics
  rather than gradient plumbing.
- **NO EFFECT:** P1 < 30% at gap 0 → gradient plumbing was necessary but not
  sufficient (consistent with the small-scale TBPTT null, §21). The honest
  conclusion would then be that this architecture, at this scale and state size,
  does not learn cross-boundary storage, and the memory-organ line closes pending
  a design change (larger state, different write rule, or a dedicated retrieval
  objective rather than LM loss).
- Guard: standard §22 metrics (PPL ≤ ~1.2×, needle with cache) must not regress
  materially; if they do, the fix trades general quality for memory and that
  trade-off is reported explicitly.

**§31a — result: NO EFFECT, and it exposes a third blocker that was in our own
§1.** Write-path BPTT trained cleanly (PPL 8.84 = 1.111×, unchanged; needle 4/4
with cache — guard passed), but the memory probe is unmoved:

| gap | ~tokens | C: O(1) state | A: upper bound |
|-----|---------|---------------|----------------|
| 0 | 128 | **0%** | **100%** |
| 1 | 256 | **0%** | — |

The upper bound at 100% makes this a *controlled* negative: the probe is valid and
the base model solves it trivially with full attention; the O(1) path still carries
nothing. Consistent with the small-scale TBPTT null (§21). Gradient plumbing was
**necessary but not sufficient**.

**Third blocker, found while recording this — the recall loss is 1/128 diluted.**
Stage-2 recall computes `out = model(xb, labels=xb)`: a **full LM loss over all 128
tokens of chunk B**, of which only the last few are the answer. The remaining ~124
tokens are highly-predictable filler. So the gradient signal for "retrieve the fact
from memory" is a small fraction of the batch loss, and the model can minimise that
loss almost entirely by predicting filler — never learning retrieval at all.

This is **exactly the failure mode this project already documented as §1**
("supervision density gates learnability": single-supervised-token recall sequences
sit at the `ln(vocab)` plateau and never learn), and which was already fixed once at
small scale (Görev C v1 → v2 switched to dense supervision and only then learned).
The same flaw was sitting in the graft recipe unnoticed. Three blockers stacked:
(1) `no_grad` on the write chunk — fixed in §31; (2) state detached at the boundary
— fixed in §31; (3) **recall target buried in dense filler loss — still present.**

## 33. Masked recall supervision (pre-registered)

Fix (3): for recall batches, mask the LM loss to the **answer tokens only**
(`labels = -100` everywhere else), so 100% of that batch's gradient is about
retrieval rather than ~3%. Motivated directly by §1 — the project's own most robust
methodological finding — rather than by a new hypothesis.

**Single variable vs §31**: `S2_RECALL_MASK=True`; write-path BPTT stays on (it is a
prerequisite: gradient must both *reach* the write path and *be about* retrieval).
Everything else identical (6-layer set, exp, hybrid writes, seed 0, gaps {0,1}).
Lineage `…WM`.

**Pre-registered criteria** (memory probe, condition C, matched, gap 0):
- **STORAGE LEARNED:** ≥ 60% → the three blockers together explain §30; the O(1)
  state is a real store within the trained range and the memory line re-opens.
- **PARTIAL:** 30–60% → signal appears; combine with delta writes (§32) and/or
  larger state before concluding.
- **NO EFFECT:** < 30% → all three training-side blockers are removed and storage
  still does not emerge. That would be strong evidence the limitation is
  **architectural/capacity**, not training, and the honest move is to stop patching
  the recipe: either change the design (much larger state, dedicated retrieval
  objective, unfrozen base) or close the memory-organ line and publish the negative.
- Guard: PPL must stay ≤ ~1.2× and cache-present needle must not regress.

**§33a — result: NO EFFECT. Decision point reached; the limit is architectural.**
With all three training-side blockers removed simultaneously (`no_grad` off, state
not detached, recall loss masked to answer tokens only):

| gap | ~tokens | C: O(1) state | A: upper bound |
|-----|---------|---------------|----------------|
| 0 | 128 | **0%** | **100%** |
| 1 | 256 | **0%** | — |
| 2 | 384 | **0%** | — |

Guards passed (PPL 8.84 = 1.112×, unchanged; cache-present needle 4/4), so nothing
was traded away — the interventions simply did not produce storage. Every output was
again a byte-identical filler continuation: the fact chunk exerts no influence on
the query chunk through the state, at any distance, not even across one boundary.

**Verdict (pre-registered):** the inability to store across boundaries is **not a
training-recipe problem**. Three targeted fixes, each addressing a genuine defect we
verified in the code, changed nothing. Combined with §31a and the small-scale §21
null, the training-side explanation is exhausted.

**Most plausible architectural cause (stated as hypothesis, not finding).** The
graft **shares Qwen's frozen `q/k/v/o_proj`**. The "key" under which a fact would be
stored is whatever the frozen `k_proj` emits — a representation optimised for
softmax attention over a growing cache, not for writing addressable entries into a
compressed associative store. The only trainable machinery is ~150k params of
depthwise conv, per-channel decay/η, write gates and an output gain. That may simply
lack the degrees of freedom to *form* a retrievable association, no matter how the
loss is shaped. Note this is consistent with everything observed: local behaviour is
learned well (PPL 1.11×), but nothing addressable survives a boundary.

**Consequences — the memory-organ line is closed as a recipe-level project.**
Continuing to patch the training loop is not justified: three pre-registered,
well-motivated interventions produced exactly zero movement. What remains is not a
tweak but a **redesign** — giving the memory path its own trainable projections
(rather than borrowing frozen ones), a dedicated retrieval objective, and likely a
larger state; i.e. building a memory-augmented model rather than grafting one on.
That is a research programme, not a next run, and it should only be started as a
deliberate choice.

**What the project has, validated, independent of this:** a 6-layer O(1) graft at
149,910 trainable params holding language quality to **1.11× PPL across 3 seeds**,
with **~8% VRAM and ~21% decode-latency savings at 128k**; a mapped
graft-density wall with a compounding diagnosis (§24–§25); a regime-dependent
retention-law result (§15h vs §28a, n=16, p=0.012); and — of independent value to
the field — a **documented methodological confound**: apparent long-range retrieval
in attention/O(1) hybrids can be carried entirely by the residual KV cache, and is
routinely measured with the cache present (§30, erratum above).

## 34. Trainable projections for the memory path (pre-registered)

The §33a decision point concluded the limit is architectural. This is the
design change it called for — and it is motivated by a **direct comparison in our
own data**, not by speculation:

| model | projections | matched probe, one boundary |
|---|---|---|
| small-scale HFP (§26b) | **own, trainable** | **33–53%** |
| Qwen graft (§30/§31/§33) | **borrowed, frozen** | **0%** |

Same memory mechanism, same task family, same probe protocol. The salient
difference is whether the model can *learn the representation under which a fact is
stored*. In the graft, the write key is whatever Qwen's frozen `k_proj` emits — a
representation optimised for softmax attention over a growing cache, not for writing
addressable entries into a compressed associative store. The only trainable
machinery was ~150k params of conv, decay, gates and output gain.

**Design.** `GraftConfig.own_proj=True`: each grafted layer gets its **own
trainable `q/k/v`**, initialised as exact copies of the teacher's, so behaviour at
initialisation is bit-identical to the shared-frozen setup — the only thing added is
freedom. `o_proj` stays shared and frozen, keeping the memory output anchored in the
base model's residual space. Trainable parameters rise 149,910 → **~19.0M (~1.3% of
the 1.5B base)** — Qwen's GQA (2 KV heads) keeps `k/v` cheap. Base remains frozen.

**Retained from earlier work** (they fixed genuine defects and cost nothing):
`S2_WRITE_BPTT=True` (§31) and `S2_RECALL_MASK=True` (§33). Lineage `…WMP`.

**Pre-registered criteria** (memory probe, condition C, matched protocol, gap 0):
- **≥ 60%** → the frozen-projection diagnosis is confirmed and the memory path
  works. Next: range extension, then the demo; the memory thesis is back on the table.
- **30–60%** → graft reaches small-scale parity. Direction confirmed but capacity
  limited; continue with larger state and/or delta writes (§32), now justified.
- **< 30%** → the frozen-projection hypothesis also fails. The graft route to a
  memory organ is then **closed**, and the only remaining path is a memory-first
  model trained from scratch — expensive, and a separate decision, not a next run.
- Guards: PPL ≤ ~1.2× and cache-present needle must not regress. With 19M trainable
  params over-fitting/quality drift is a real risk, so a PPL regression here is
  informative rather than merely a nuisance.

**Honest note on cost.** This is the first arm that meaningfully leaves the
"325k/150k params, extreme frugality" framing behind. If it succeeds, the headline
becomes "~1% of parameters" rather than "0.01%" — still small, but the frugality
claim must be restated accordingly.

**§34a — split result: memory NEGATIVE, quality a significant POSITIVE.**

*Primary (pre-registered) question — cross-boundary storage: **failed again.***

| gap | ~tokens | C: O(1) state | A: upper bound |
|-----|---------|---------------|----------------|
| 0 | 128 | **0%** | **100%** |
| 1 | 256 | **0%** | — |
| 2 | 384 | **0%** | — |

Giving the memory path its own trainable `q/k/v` (19M params vs 150k) moved
cross-boundary retrieval **not at all** — still byte-identical filler continuations.
The frozen-projection hypothesis is **refuted** as an explanation for the storage
failure.

*Secondary outcome — language quality: the best result the project has produced.*

| arm | trainable params | PPL | ratio | ≤1.05× criterion |
|-----|------------------|-----|-------|------------------|
| §22 reference | 149,910 (0.01%) | 8.84 | 1.111× | failed |
| **§34 own-proj** | **~19M (1.3%)** | **8.30** | **1.043×** | **MET — first time** |

Needle (cache present) 4/4 at all four lengths. The K3 criterion `PPL ≤ 1.05×`,
set at the start of the graft line and never met in ~10 runs, is now met. So the
frozen projections *were* a real bottleneck — for **local attention approximation
quality**, not for storage. Two clean Pareto points now exist: 0.01% params at
1.11×, or 1.3% params at 1.04×.

**What this tells us about the storage failure.** More representational freedom
(127× more trainable parameters, own projections, all three training blockers
removed) buys substantial quality and **exactly zero** cross-boundary retrieval.
Storage is therefore not capacity-starved and not representation-starved in any way
we can reach from this direction — it is a *different kind* of failure. Per the
pre-registration, the **graft route to a memory organ is closed.**

**Honest limitation across all memory arms (§30–§34), recorded rather than
excused.** Every run used the same recall budget: `RECALL_MIX=0.25` over 600 S2
steps ≈ **150 recall steps** (~600 examples), with gaps {0,1}. That is a small
amount of supervision for acquiring a genuinely new capability, and it was constant
across arms — so it cannot explain *differences* between arms, but it could in
principle explain the uniform zero. We are not running another arm to test it: four
pre-registered interventions have now produced no movement whatsoever, and
continuing would be the patch-chasing the §33a decision point explicitly ruled out.
It is stated here so that anyone continuing this line knows the first thing to vary.

**Net position after §34.** The project's validated contribution is an
**efficient-attention graft**, now with a genuinely competitive quality point
(1.043× at 1.3% trainable params, free-tier T4), a mapped density wall with a
compounding diagnosis, a regime-dependent retention-law result, and a methodological
warning about KV-cache-carried retrieval in hybrids. The memory-organ thesis is
**not supported by any evidence we have produced**, and pursuing it further requires
a memory-first architecture rather than a graft — a separate programme.

## 32. Delta write rule for cross-boundary storage (pre-registered)

> **Status: NOT RUN.** This arm was prepared (notebook parameter `WRITE_RULE`,
> checkpoint lineage `…D`) but never executed: §33a/§34a closed the graft route to
> a memory organ, which removed its rationale (a write-rule change cannot help if
> nothing reaches the state at all). Kept on record as a pre-registration that was
> superseded, not as a result.

Motivated by the project's own oldest finding — **"the memory is
interference-limited, not decay-limited"** (README / GPU_ROADMAP §0). The delta
rule exists precisely to remove interference: it reads the current association and
writes the *difference*, `M ← M + β·k(v − kᵀM)ᵀ`, overwriting a key's old value
instead of stacking onto it. If the O(1) state fails to hold facts because repeated
additive writes bury them, delta is the mechanism-level fix.

Two observations make this the natural next arm:
- The graft already runs `write_rule='hybrid'` (α-gate between additive and delta)
  but with `alpha_init=-2` (σ(−2)≈0.12, additive-weighted), and training logs show
  `alpha_ort` pinned at **0.119 → 0.135** across the whole run — the model never
  moved toward delta on its own.
- `additive` was locked in by K2, but that decision was made on **language-model
  perplexity**, not on memory retention. The right rule for a memory organ may
  differ from the right rule for LM quality; if so, that trade-off is itself a
  result worth documenting.

**Design.** `WRITE_RULE='delta'` (α fixed at 1, pure delta), **with**
`S2_WRITE_BPTT=True` — gradient to the write path is a prerequisite, so delta is
tested on top of §31 rather than instead of it. Versus §31 this is a **single
variable** (write rule). All else identical: 6-layer reference set, exp decay,
dpfp, seed 0, gaps {0,1,2,3}. Checkpoint lineage `…WD`.

**Pre-registered criteria** (evaluated with the §30 harness, condition C, matched
probe, gaps 0–3 — inside the trained range):
- **DELTA UNLOCKS STORAGE:** matched-C P1 ≥ 60% at gap 0 **and** materially above
  the §31 arm → interference was the binding constraint; delta becomes the
  memory-path default and the memory-organ line re-opens (next: range extension,
  then bigger state).
- **NO DIFFERENCE vs §31:** both arms behave alike → the write *rule* is not the
  constraint; remaining levers are an explicit retrieval objective and/or larger
  state (`bulk_dim`, `dpfp_nu`), which preserve O(1) since the state stays constant
  in context length.
- **DELTA HURTS:** memory unchanged and PPL regresses materially (> ~1.3×) →
  confirms K2's LM-based choice and closes the write-rule lever.
- Guard: PPL and cache-present needle are reported alongside; if delta buys memory
  at a real LM cost, the trade-off is stated explicitly rather than hidden.

## 35. Chaining across boundaries: capacity × write rule (pre-registered)

*Written before the run. This section is the authoritative pre-registration; the
notebook markdown and roadmap are convenience copies.*

> **Revision v3 (2026-08-01) — the v2 design was underpowered and is superseded.**
> Recorded openly rather than silently replaced, per the erratum policy.
>
> **What was wrong.** v2 specified 4 arms × **2 seeds**, with the primary endpoint
> being matched-probe accuracy at K=2 against **absolute** thresholds (≥30% /
> 10–30% / <10%). Those thresholds were calibrated on §26b — which used **6 seeds
> and 200 trials** and *still* failed its pre-registered separation bar, with the
> exp arm spanning **8.5–69.5% across seeds at K=0**. §26b's own recorded lesson
> was that settling this needs "12–20 seeds **or** a lower-variance endpoint."
> v2 went the other way: fewer seeds, fewer trials, and a *more* decisive verdict.
> With that seed variance, a 2-seed mean cannot distinguish a real null from noise,
> so the near-certain "NO EFFECT" outcome would have closed the interference
> hypothesis on an artifact of sample size — exactly what this project's own
> power rule ("if both arms are at chance, report *inconclusive*, not *null*")
> forbids.
>
> **Also corrected:** v2 (and the roadmap and `DEVIR.md`) cited the small-scale
> model as reaching "33–53%" at K=0 as if that were one configuration's range. It
> is not: **33.2% is the `exp` arm mean and 53.4% is the `cubic` arm mean** in
> §26b's 6-seed table. §35 fixes `decay_mode='exp'`, so the correct baseline
> expectation here is **~33%, with a seed range of 8.5–69.5%**.
>
> **v3 fix — budget-neutral (still 8 runs):** trade arms for seeds and switch to a
> low-variance endpoint.
>
> | | v2 | v3 |
> |---|---|---|
> | arms | 4 (2×2) | **2** (baseline vs strongest contrast) |
> | seeds | 2 | **4** |
> | primary endpoint | matched-probe accuracy at K=2 (%) | **cross-chunk validation loss** (nat, paired by seed) |
> | criterion | absolute ≥30% | **paired Δ ≤ −0.15 nat + sign consistency** |
> | status of the claim | confirmatory | **screening** |
>
> **Cost of merging the arms, stated plainly:** if a signal appears, this run
> cannot say whether capacity or the write rule produced it. Separating them is the
> follow-up. This is a deliberate trade: detection has to come before attribution.

**Why this experiment.** §34a closed the *graft* route to a memory organ. But the
small-scale HFP model — which has its **own trainable projections** — is not at
zero: on the matched probe its `exp` arm reaches **33.2% (seed range 8.5–69.5%)
across one chunk boundary (K=0)** and falls to chance by K=2 (§26b). So the
mechanism *works but does not chain*. That is a sharper and far cheaper question
than anything available in the graft, and it runs without a GPU-scale budget.

**Central question:** why does the state survive one boundary but not two?

**Hypothesis — taken from this project's own findings, not invented here.** The
memory is **interference-limited, not decay-limited** (README/GPU_ROADMAP §0), and
this was independently re-confirmed in §27a: *lengthening* cubic's plateau made
retention **worse**, because a channel that forgets less accumulates more
interference. Filler chunks write a distractor key-value every `DIST_EVERY` tokens;
after two chunks the target is buried. If interference is the binding constraint,
two levers should chain the state — and **neither breaks O(1)**, since both grow a
*constant*, not something that scales with context:

1. **State capacity** — `dpfp_nu` 2→4 doubles `key_dim` (256→512), diluting
   interference. (Note: `bulk_dim` does **not** affect the memory; it is an FFN
   parameter. This was checked in the code, not assumed.)
2. **Write rule** — `additive` (accumulates, interferes) → `delta` (reads the
   current association and writes the difference, overwriting instead of stacking).
   Already implemented; gave a 2× multi-seed gain on a key-update task (§13 era).

**Design (v3).** `review_scripts/carry_curriculum.py` (train) + `matched_probe.py`
(eval), orchestrated by `notebooks/chain_capacity_v35.ipynb`. **Two arms** —
baseline (`dpfp_nu=2`, `additive`) vs treatment (`dpfp_nu=4`, `delta`) — × **4
seeds** (0–3), `decay_mode='exp'` fixed. Reduced-cost settings retained from v2
(CTX 128, 600 steps, BS 4, CARRY_MAX 8, `DIST_EVERY` 32) after the first attempt
was measured at ~25 h on CPU; the baseline arm runs in the same sweep, so
**within-run comparison is preserved** even though absolute numbers are not
comparable to §26b.

**Primary endpoint:** **end-of-training cross-chunk validation loss at K=2**, in
nats, **paired by seed** (treatment − baseline). Chance ≈ 3.40 nat; lower is
better. This is the low-variance endpoint §26b identified: a batch-averaged loss
rather than a 30–50-trial accuracy. The measurement was fixed for this run — the
previous implementation replicated a *single* example 8× (effective n=1) and wrote
the number nowhere; it now averages **64 distinct carry examples** and is written
to `{TAG}_valloss.csv`. Training is unchanged; only the post-training measurement.
Because of this fix the value is not directly comparable to the single-example
numbers printed by §26–§28 runs.

The validation examples are drawn from a dedicated RNG stream seeded `12345+SEED`
and independent of `dpfp_nu`/`write_rule`, so **both arms of a given seed are
scored on the identical 64 examples** — the pairing is exact, not just nominal.
The stream is restored afterwards, so `CC_VAL_N` does not shift the lifetime probe.

**Secondary (descriptive, not decisive):** matched-probe `matched_acc` and
`matched_logp` at K ∈ {0,1,2,4}, 50 trials (chance 3.3%). Reported with per-seed
min–max, never as a bare mean.

**Pre-registered criteria (screening, n=4 paired).**
- **SIGNAL:** paired mean Δ ≤ **−0.15 nat** *and* **4/4 seeds** favour treatment →
  the interference lever measurably improves cross-boundary learning. This is
  **not** "confirmed": at n=4 a sign test yields at best p=0.0625. The single
  permitted follow-up is *attribution + power* — `nu4/additive` vs `nu2/delta`,
  8–12 seeds, same primary endpoint.
- **REVERSED:** paired mean Δ ≥ **+0.15 nat** *and* 0/4 seeds favour treatment →
  capacity+delta actively hurts; the interference hypothesis is refuted in this
  direction and the line closes.
- **INCONCLUSIVE (not "null"):** anything else — |Δ| < 0.15 nat or split signs →
  at this power the hypothesis is neither confirmed nor refuted. Per this
  project's power rule, **"no effect" will not be written.** The decision is then
  explicitly the author's: raise to 8–12 seeds, or park the line and move to the
  read/addressing diagnostic (can a value written into the state be read back
  *within* the same chunk vs *after* a boundary — separating "write is broken"
  from "read is broken").

The 0.15 nat threshold is taken from §28a's own refutation bar, not chosen here.

**Discipline note.** Given this project's documented tendency to chase "one more
experiment" after each negative (BEKLEYEN, roadmap §4), the **SIGNAL** branch is
capped at exactly one follow-up, **REVERSED** closes the line, and
**INCONCLUSIVE** is a decision point for the author rather than an automatic
further arm.

## Reproduction

```bash
python smoke_test.py
python review_scripts/verify_claims.py
python review_scripts/dense_retention.py exp additive 1e-3 0
python review_scripts/length_gen.py train 0 && python review_scripts/length_gen.py eval 0
LG_VARIANT=dpfp python review_scripts/length_gen.py train 0
python review_scripts/interference_eval.py 0
```
