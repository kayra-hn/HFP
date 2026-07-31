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

**1. An extremely small graft preserves language quality.**
6 of 28 attention layers converted to O(1) memory with **149,910 trainable
parameters (~0.01% of the base)** holds perplexity at **1.11× baseline
(7.96 → 8.84), replicated across 3 seeds**. Measured savings at 128k context:
**~8% peak VRAM and ~21% decode latency**; the grafted layers' state is
**9.5 MB, constant in context length** versus ~3.8 GB of KV cache.

**2. A mapped density wall, with a mechanism.**
Pushing from 6 to 13 grafted layers costs 1.6–1.8× perplexity, and this is
**robust**: naive layer selection, principled selection by per-layer
reconstruction cost, LR/warmup stabilisation, and a student-forward
(exposure-bias) fix all land in the same band. Stage-1 per-layer reconstruction is
excellent (MSE ≈ 0.089) while end-to-end quality degrades — localising the limit to
**error compounding through stacked lossy layers**, not per-layer capacity.

**3. Retention law matters only in sparse-write regimes.**
A content-adaptive "cubic" decay (λ = 1/√(1+2η·z²), forgetting scaled by channel
occupancy) versus a fair multi-scale learned-λ exponential control:
*no measurable difference* in the dense/saturated LLM-graft regime, but a
pre-registered, paired, 16-seed win in a sparse-write carry task
(**+0.48 nat, 13/16 seeds, p = 0.012**). Tuning its one free parameter improves it
in neither direction, and large values are numerically unstable — so the effect is
real, modest, and regime-specific.

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
memory store. The open question — whether the O(1) state can hold addressable
information across chunk boundaries given adequate architectural freedom — is under
test: our small-scale model with *trainable* projections reaches 33–53% at one
boundary where the graft with *frozen, borrowed* projections reaches 0%, so the next
pre-registered experiment gives the memory path its own trainable q/k/v
(~19M params, ~1.3% of the base).

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
