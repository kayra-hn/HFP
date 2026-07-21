---
license: agpl-3.0
library_name: transformers
tags:
- pytorch
- causal-lm
- linear-attention
- long-context
- o1-memory
- hfp
language:
- en
---

# HFP — Hyper-Flux Projection

> 🇹🇷 **Türkçe:** Projenin detaylı Türkçe açıklamaları, iç planlama notları ve deney sonuçları için [docs/tr](docs/tr) ve [docs/internal_tr](docs/internal_tr) klasörlerine, özellikle [Deney Sonuçları (Türkçe)](docs/tr/DENEY_SONUCLARI.md) belgesine bakabilirsiniz.

An experimental causal language-model architecture that pairs **windowed local
attention** with a **per-layer recurrent memory** (a decayed linear-attention
state `M ∈ ℝ^{H×H}`, `z ∈ ℝ^H`). The inference-time state is **constant in
context length** — O(1) memory instead of a growing KV-cache — and long-range
information is forced to travel through that recurrent memory.

Its distinguishing feature is the **retention law** of that memory. Alongside the
standard exponential decay used by every efficient recurrent model (RetNet, GLA,
Mamba, …), HFP implements a **cubic-plateau decay** derived from the Hyper-Flux
Projection physics papers, and lets you switch between them with a single flag.

> **Honesty note.** HFP is *inspired by* the HFP physics papers (5D→4D
> projection, moduli memory). The code is **not** a simulation or isomorphism of
> that physics — the mapping is an analogy that motivated design choices. Every
> claim here is an ML claim, to be established by the experiments below. The
> physics is not validated by the model, and the model is not evidence for the
> physics.

## The idea: retention law as the design axis

A fixed-size recurrent memory must forget. *How* it forgets is the design choice.

- **`exp` (baseline).** Geometric decay, `M_t = λ⊙M_{t-1} + k_t v_tᵀ`, with a
  learned per-channel `λ = σ(decay)`. This is the only envelope that folds into
  an exact O(1) state (`g(t−j)=G(t)/G(j)` composability), which is why the whole
  efficient-recurrent family uses it.

- **`cubic_flux` (HFP).** A direct discretization of the paper's cubic relaxation
  `dθ/dτ = −η·θ³`. The stable single-step solution gives a **state-magnitude
  dependent** decay factor:

  ```
  λ_t = 1 / sqrt(1 + 2·η·s_t²)        # s_t = current per-channel state magnitude
  ```

  When a channel is nearly empty (`s→0`) it barely decays (**plateau, no
  forgetting**); when it fills up it decays in proportion to its magnitude
  (**active, self-limiting forgetting**). The result is a *plateau then
  power-law* retention envelope rather than a geometric cliff — a mechanism that,
  to our knowledge, no mainstream recurrent model uses. `η` is a learned
  per-channel "flux" parameter.

Set the mode with `decay_mode="exp"` / `"cubic_flux"` (config) or
`--decay_mode` (CLI). Everything else is held identical, so the two modes are a
clean controlled comparison of the retention law alone. That comparison has now
been run at LLM-graft scale and found **no measurable difference** between the
laws (RESULTS §15h) — the framework's honest answer so far is that *which*
retention law matters less than how the memory is trained; the law remains a
switchable axis, with `exp` the pragmatic default for new work.

## Architecture

- **Recurrent memory (per layer).** Causal, chunkwise, causal-inclusive linear
  attention: every token reads the cumulative state (its own KV included), so the
  memory path is trained end-to-end by the LM loss. Chunked streaming and a
  single-shot forward are mathematically identical (checked by `smoke_test.py`);
  both retention modes are chunk-consistent.
- **Binding convolution.** A short depthwise **causal conv** (kernel 3) on the
  Q/K pathway mixes each token with its predecessors, so a value's key encodes
  the key that preceded it. This is what makes associative recall possible in a
  linear-attention memory (cf. Mamba/H3/Based); values are read from the clean,
  un-convolved input. It is orthogonal to the retention law and applies in both
  modes. Set `conv_kernel=1` to ablate it.
- **Windowed local attention.** Multi-head attention restricted to a sliding
  window (`local_window`); long-range information must flow through the recurrent
  memory. `local_window=None` gives full causal attention.
- **Embedding / positional balance.** Token embeddings are scaled by `√d` and the
  sinusoidal positional encoding by `pe_scale` (default 0.3) so token content is
  not drowned by position — a prerequisite for content-based recall.
- **EntangledLinear FFN.** The two FFN projections are generated from a single
  shared bulk weight (`P_A·W_bulk`, `P_B·W_bulk`) — a parameter-tying scheme
  motivated by the papers' "two shadows of one bulk vector" picture.



## Status of results

See **[RESULTS.md](RESULTS.md)** for the full, multi-seed experimental record.
Headline findings (small scale, synthetic recall; patterns are seed-robust):

- **Length generalization**: models trained at 160 tokens transfer to 1280-token
  streams (8x), with fixed-gap recall *improving* as fact density falls.
  Train-short / infer-long is the supported deployment mode of the O(1) state.
- **The memory is interference-limited**, not decay-limited, in the tested regime.
- **DPFP capacity axis** (`key_feature_map="dpfp"`) is the first mechanism with a
  clear, 3-seed advantage: ~2-6x baseline accuracy at long gaps under high
  interference, and it stabilizes training across seeds.
- **Official recipe (locked)**: `cubic_flux_chunked` decay + `additive` writes +
  `dpfp` features + `ffn_type="standard"` (WikiText-2 ablation, RESULTS §10; write
  rule locked by the pre-registered K2 experiment, RESULTS §13).
- **`cubic_flux` long-horizon win**: In sparse, long-gap regimes (gap ≥ 256), `cubic_flux_chunked` paired with DPFP outperforms the exponential baseline significantly (3x recall advantage), validating the core long-horizon hypothesis.
- **Language Modeling**: HFP showed a favorable small-scale TinyShakespeare ranking against a GPT-2-style baseline under the same historical skip-one objective (PPL labels are under revision; see RESULTS §14). This is evidence for LM viability, not a final next-token/O(1) headline claim.
- **GLA family comparison — under revision (honest note)**: a metric artifact
  (double-shifted labels; RESULTS §14) made the published HFP-vs-GLA numbers
  non-comparable. A corrected single-seed probe reaches next-token PPL 55.4
  (seq 256) vs GLA's 226.7, but with full attention enabled; the
  resource-matched O(1) comparison is being re-established before any claim.

## Usage

```python
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

config = HFPConfig(vocab_size=50257, hidden_size=256, num_hidden_layers=4,
                   num_attention_heads=4, local_window=64,
                   decay_mode="cubic_flux")   # or "exp"
model = HFPForCausalLM(config)
```

Streaming inference with constant memory:

```python
past = None
for chunk in token_chunks:                    # e.g. 256-token chunks
    out = model(chunk, past_key_values=past, use_cache=True)
    past = out.past_key_values                # fixed-size state, does not grow
```

## Experiments

```bash
python smoke_test.py            # regression tests — run first, before trusting anything

# Long-range recall A/B (the headline comparison)
python run_experiment.py --task retention --steps 1500 --context 96 \
    --max_gap 64 --local_window 16 --decay_mode exp
python run_experiment.py --task retention --steps 1500 --context 96 \
    --max_gap 64 --local_window 16 --decay_mode cubic_flux

# Language modeling
python run_experiment.py --task lm --steps 1500 --seq 128 --decay_mode cubic_flux

# MQAR recall with chunked-vs-reset ablation
python run_experiment.py --task recall --steps 1500 --context 128 --pairs 8
```

`smoke_test.py` covers: gradient flow through all memory parameters, MQAR
label alignment, chunk-consistency of both retention modes, and cached
generation. Note: `cubic_flux` uses a sequential scan (O(L)); it is slower than
the parallel `exp` path and best run on GPU.

## Grafting onto pretrained LLMs (Phase 3, experimental)

`hfp/models/grafting.py` swaps a subset of a pretrained Llama-family model's
attention layers for HFP memory modules (per-head state, GQA-aware, RoPE
bypassed, projections warm-started and frozen), then distills them
teacher-free: in `teacher_forcing` mode each grafted layer trains its HFP
branch against the original attention's output inside the same forward pass —
no second model in memory, so a 1.5B model fits on a single T4.

```python
from transformers import AutoModelForCausalLM
from hfp.models.grafting import GraftConfig, graft_llama, set_graft_mode, distill_loss

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
graft_llama(model, GraftConfig(decay_mode="cubic_flux_chunked", write_rule="hybrid"))
set_graft_mode(model, "teacher_forcing")     # Stage 1: layerwise MSE distillation
out = model(input_ids); distill_loss(model)  # -> aux loss (or use set_distill_backward)
set_graft_mode(model, "student")             # inference: O(1) grafted layers
```

`write_rule="hybrid"` is the **alpha-gate write**: a learnable per-head
interpolation between additive (archival) and delta (updating) writes,
`M += β·k(v − α·v_old)ᵀ`, solved chunkwise-parallel. Run
`review_scripts/graft_smoke.py` first; the full training/validation pipeline
(zero-shot sanity → Stage 1/2 → needle + constant-VRAM checks) is
`notebooks/colab_graft_qwen_v3_kaggle.ipynb`.

**Status after six runs (2026-07-18/19, full honest record in
[`RESULTS.md` §15-§15h](RESULTS.md)):** the decisive ingredient turned out to
be a **cross-chunk recall distillation curriculum** — recall documents split
across chunk boundaries so attention cannot see the needle and the teacher's
full-attention retrieval becomes the KL target for the memory path. With it,
the 325k-parameter graft retrieves **never-seen-in-training passphrases at
512-16384 token distances** (reliability grid over 5 lengths × 3 positions ×
3 seeds: 38/45 for cubic, 42/45 for exp), trained entirely on a free T4.
An in-window recall mix teaches nothing (ablated: Run 3 vs Run 4) — the
cross-chunk structure is causal. A controlled retention-law ablation
(identical twin runs) found **no measurable cubic-vs-exp difference** (PPL
13.04 vs 12.87; grids above): the retrieval capability belongs to the
protocol, not the decay law. Open gaps, also recorded: LM quality cost is
still ~1.6× PPL (criterion ≤1.05× not met), one weak grid cell per law, and
all of this is a single training seed per arm. Diagnostics:
`notebooks/kaggle_graft_diagnostics_v1.ipynb`.

## Repository layout

```
hfp/                              the package
  core/hfp_bulk_state.py          recurrent memory: M,z state, exp + cubic_flux decay, binding conv
  core/bulk_trigger_decoder.py    decoder layer: windowed attention + EntangledFFN
  models/modeling_hfp.py          HuggingFace-compatible model
  models/configuration_hfp.py     config
  models/grafting.py              Phase 3: HFP memory grafting onto pretrained LLMs
run_experiment.py                 retention / recall / lm experiments
smoke_test.py                     regression tests (CI gate)
train.py                          standard AdamW training loop
eval_*.py                         memory-scaling / optimizer / passkey evals
review_scripts/                   independent verification & analysis scripts (CI runs verify_claims.py)
notebooks/                        Colab and Kaggle training/diagnostic notebooks
examples/                         agent-framework integration demos (HFPAgentWrapper)
hf_upload/                        Hugging Face release package (model card, standalone modeling files)
docs/                             paper draft (.tex), OSF companion PDF, translations, internal notes
assets/figures/                   generated figures (plot_plateau.py output)
requirements.txt / pyproject.toml dependencies & package metadata
LICENSE                           GNU AGPL-3.0 (full text)
```

## Papers and Decoupling from Physics

This repository is the implementation of **Hyper-Flux Projection Model III: O(1)-Memory Language Modeling via Cubic-Plateau Retention** (draft in `docs/paper3_ml_architecture.tex`).

While the architecture is *inspired* by the theoretical frameworks in Papers I and II (5D Einstein-Dilaton geometry, Kasner metrics, and Moduli Selection hosted on OSF: <https://osf.io/xc7e4>), **the ML implementation is strictly decoupled from the physics**. The cubic decay equation (`dθ/dτ = −η·θ³`) and mechanisms like DPFP and Delta writes are standalone machine learning innovations designed for efficient text modeling. 

**This description supersedes any earlier, marketing-styled summary of HFP.**
The physics does not validate the ML architecture, and the ML results do not serve as proof of the physics. There is no active "Ryu–Takayanagi bound", "Witten propagator", "5D curvature" or "quantized-energy scheduler" in the trained path. 

The authoritative record of the ML architecture's empirical performance is in [`RESULTS.md`](RESULTS.md) (and its Turkish translation `docs/tr/DENEY_SONUCLARI.md`). Demonstrated so far: the **DPFP capacity axis** (§5), **train-short / infer-long length generalization** (§3), **initial small-scale LM viability** (§7, §10), the **K1 family-baseline gate passed** against an in-house GLA implementation (§16: −0.211 nats over 3 seeds, all GLA seeds diverged), and — the current headline — **long-range retrieval grafted onto a pretrained LLM via cross-chunk distillation** (§15f-g). On the retention law itself, honesty requires stating both directions: the small-scale long-horizon advantage of `cubic_flux` (§6) remains on record, but a controlled ablation at LLM-graft scale found **no measurable cubic-vs-exp difference** (§15h), and neither law survived an out-of-training-regime lifetime probe at small scale (§17-§20a, under active investigation — the current suspect is a training-recipe issue: learned decay calibrated to the training window plus detached chunk boundaries, not the architecture).

## License

GNU AGPL v3.0 — full text in [`LICENSE`](LICENSE). Commercial network deployment
of this architecture or derivatives requires open-sourcing modifications under
the same license. Code is AGPL-3.0; the OSF text/figures are licensed separately
(see the OSF project). Citation metadata: [`CITATION.cff`](CITATION.cff).
