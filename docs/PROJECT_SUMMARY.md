# HFP — Project Summary

**Kayrahan Yılmaz** · [github.com/kayra-hn/HFP](https://github.com/kayra-hn/HFP) · AGPL-3.0
Independent researcher · Last updated 2026-07-28

*One-page summary for grant applications, collaboration enquiries, and as the
basis for a write-up. Everything below is backed by
[`RESULTS.md`](../RESULTS.md), which records negative results and one erratum in
full.*

---

## What this is

An empirical study of **grafting O(1)-memory recurrent attention into a pretrained
LLM**: replacing a subset of a frozen Qwen2.5-1.5B's softmax-attention layers with a
constant-size recurrent state (`M ∈ ℝ^{H×H}`, `z ∈ ℝ^H`), trained by teacher-free
two-stage distillation. The line of work sits alongside Mamba-in-Llama, LoLCATs,
MOHAWK and LAWCAT; the contributions here are empirical and methodological rather
than a new architecture.

All experiments run on **free-tier T4 GPUs** (Kaggle/Colab). Every headline is
pre-registered — criteria written before the run — and negatives are published
alongside positives.

## Main results

**1. A very small graft preserves language quality — two Pareto points.**
6 of 28 attention layers converted to O(1) memory with **149,910 trainable
parameters (~0.01% of the base)** holds perplexity at **1.11× baseline
(7.96 → 8.84), replicated across 3 seeds**; giving the memory path its own
trainable projections (**~19M, 1.3%**) reaches **1.043× (7.96 → 8.30)**, meeting the
project's pre-set `≤1.05×` bar for the first time. Measured savings at 128k context:
**~8% peak VRAM and ~21% decode latency**; the grafted layers' state is
**9.5 MB, constant in context length** versus ~3.8 GB of KV cache.

**2. A mapped density wall, with a mechanism.**
Pushing from 6 to 13 grafted layers costs 1.6–1.8× perplexity, and this is
**robust**: naive layer selection, principled selection by per-layer
reconstruction cost, LR/warmup stabilisation, and a student-forward
(exposure-bias) fix all land in the same band. Stage-1 per-layer reconstruction is
excellent (MSE ≈ 0.089) while end-to-end quality degrades — localising the limit to
**error compounding through stacked lossy layers**, not per-layer capacity.

**3. Retention law: a small-scale win that reverses at LM scale.**
A content-adaptive "cubic" decay (λ = 1/√(1+2η·z²), forgetting scaled by channel
occupancy) versus a fair multi-scale learned-λ exponential control.

*Small scale, synthetic sparse writes (§28a, §29b):* a pre-registered, paired,
16-seed win — **+0.48 nat, 13/16 seeds, p = 0.012**, replicated at **+0.41 nat,
14/16, p = 0.0005**. Decomposing that metric, however, places **+0.43 nat
in-chunk** (p = 0.0008) against only **+0.28 nat cross-chunk** (paired t
p = 0.11): the gain is mostly within-chunk retention, not carrying across a
boundary.

*LM scale, with the cache confound removed (§36):* the same two controlled graft
twins were re-evaluated with the KV cache reset at every chunk boundary, so the
O(1) state is the only cross-boundary channel. Carrying the cubic state then makes
next-token prediction **0.28 nat/token worse** than discarding it, while the
exponential arm gains a marginal +0.035 — a paired difference of
**−0.3167 nat/token, Wilcoxon p = 2.7e−09**, n = 120 chunks. The earlier null in
this regime (§15h) was measured with the cache present and was therefore largely
blind to the state.

The mechanism is the same in both directions: a decay that forgets less protects
an unsaturated channel, and for exactly that reason retains stale content longer
once the context moves on. Tuning the one free parameter improves it in neither
direction and large values are numerically unstable. **Verdict: the effect is real
at small scale, narrower in attribution than first reported, and reversed at LM
scale. `exp` is the shipped default; the cubic LM-scale line is closed.**

**4. A methodological warning for the field (self-inflicted, then corrected).**
Apparent long-range retrieval in attention/O(1) **hybrids can be carried entirely by
the residual KV cache** of the un-grafted layers — and is routinely evaluated with
that cache present. Our own needle-in-haystack results at 512–16 384 tokens looked
like memory retrieval; resetting the KV cache per chunk, leaving only the recurrent
state, gives **0% at every distance, including a single chunk boundary** (upper
bound with full attention: 100%; checkpoint load verified bit-exact). We traced the
cause into our training code — the write chunk ran under `no_grad`, the state was
detached at boundaries, and the recall target was diluted in a full-sequence LM loss
— removed all three, and storage still did not emerge, which localises the
limitation to architecture rather than recipe. The affected claims are corrected in
an erratum rather than quietly edited.

## Honest status

The graft is validated as an **efficient-attention technique**, not as a long-term
memory store — and we tested that distinction hard rather than assuming it.

Four pre-registered interventions targeted cross-boundary storage: restoring
gradient to the write path, un-detaching the recurrent state at chunk boundaries,
masking the recall loss to the answer tokens only, and finally giving the memory
path its own trainable `q/k/v` (150k → 19M parameters). The last of these improved
*language quality* markedly (1.111× → 1.043×) and produced **0%** cross-boundary
retrieval — as did all the others. Storage is therefore neither capacity-limited nor
representation-limited in any way reachable from this direction.

The graft route to a memory organ is closed. Pursuing the thesis further would
require a **memory-first architecture** trained with retrieval as a primary
objective, not a graft onto a frozen model — a separate programme, not a next run.

## Method

Pre-registration of criteria before each run; controlled twin runs with a single
variable; out-of-training-vocabulary probes to exclude memorisation; upper-bound
control conditions to separate model limits from mechanism limits; explicit power
checks so zero-sensitivity experiments are reported as inconclusive rather than
null; checkpoint fingerprinting; and published negatives. Roughly two thirds of the
recorded experiments are negative results.

## What compute would enable

Current constraint is free-tier GPU quota (T4, ~2–3 h per run). With sustained
access to a single A100-class GPU: multi-seed replication of the architectural arm,
range extension for cross-boundary storage, and a scaling check at 7B — the three
things that would turn this from a small-scale study into a transferable result.

## Licence and contact

Code AGPL-3.0 (dual-licensing possible; copyright held solely by the author).
Base model weights are Qwen2.5-1.5B under its own licence and are not redistributed.
Contact: yilmazkayrahan06@gmail.com
