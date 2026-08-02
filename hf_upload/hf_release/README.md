---
license: agpl-3.0
library_name: transformers
pipeline_tag: text-generation
tags:
- pytorch
- causal-lm
- linear-attention
- long-context
- recurrent-memory
- o1-memory
- hfp
- custom_code
language:
- en
---

# HFP — Hyper-Flux Projection (O(1)-Memory Causal LM)

> **Status: research preview — architecture only, weights are UNTRAINED.**
> This repository ships the model *code* and a randomly-initialized checkpoint so
> the architecture can be loaded, inspected and trained. It is not a usable
> language model yet. Canonical source & experiments:
> **[github.com/kayra-hn/HFP](https://github.com/kayra-hn/HFP)**

HFP is an experimental causal LM that pairs **windowed local attention** with a
**per-layer recurrent linear-attention memory** (`M ∈ ℝ^{key_dim×H}`, `z ∈ ℝ^{key_dim}`).
The inference-time state is **constant in context length** (O(1) memory instead of
a growing KV-cache); long-range information must flow through the recurrent memory.

Its distinguishing feature is a selectable **retention law** for that memory:

- `decay_mode="exp"` — standard geometric decay (the RetNet/GLA/Mamba family baseline).
- `decay_mode="cubic_flux"` — an exact discretization of the cubic relaxation
  `dθ/dτ = −η·θ³`: a **state-magnitude-dependent** decay
  `λ_t = 1/√(1+2η·z_t²)`. Empty channels barely decay (plateau); full channels
  forget actively (self-limiting).

Two further independent axes: a **binding convolution** on the Q/K path
(`conv_kernel`, ablate with 1) and a **capacity axis** via DPFP key feature maps
(`key_feature_map="dpfp"`).

## Honest status of results

Full multi-seed record: [RESULTS.md on GitHub](https://github.com/kayra-hn/HFP/blob/main/RESULTS.md).
Highlights (small scale, synthetic recall; patterns seed-robust across 3 seeds):

- **Length generalization**: trained at 160 tokens, the model transfers to
  1280-token streams (8x), with fixed-gap recall *improving* as fact density falls —
  train-short / infer-long is the supported deployment mode of the O(1) state.
- **DPFP capacity axis** (`key_feature_map="dpfp"`): first mechanism with a clear
  advantage — 2-6x baseline accuracy at long gaps under high interference, plus
  more stable training. Recommended: `exp` + additive + `dpfp` + `ffn_type="standard"`.
- `cubic_flux` currently trails the exponential baseline at this scale (parked as
  a long-horizon hypothesis; exact parallel form implemented). No LM-benchmark
  claims are made. Weights in this repo remain untrained/architecture-only.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "kayrahan35/HFP-O1-Memory-Model",
    trust_remote_code=True,          # custom architecture (HFPForCausalLM)
)

# Streaming inference with constant memory:
past = None
for chunk in token_chunks:                    # e.g. 256-token chunks
    out = model(chunk, past_key_values=past, use_cache=True)
    past = out.past_key_values                # fixed-size state, does not grow
```

Switch the retention law / capacity axis at construction:

```python
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("kayrahan35/HFP-O1-Memory-Model", trust_remote_code=True)
cfg.decay_mode = "cubic_flux"        # or "exp"
cfg.key_feature_map = "dpfp"         # or "elu"
model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
```

Note: `cubic_flux` uses a sequential scan (O(L)) and is ~2–3× slower than the
parallel `exp` path.

## Files

`modeling_hfp.py` / `configuration_hfp.py` — HF-compatible model & config;
`hfp_bulk_state.py` — the recurrent memory (retention laws, binding conv, DPFP);
`bulk_trigger_decoder.py` — decoder layer (windowed attention + shared-bulk FFN).
Training scripts, regression tests (`smoke_test.py`) and the retention/recall
experiment suite live in the [GitHub repository](https://github.com/kayra-hn/HFP).

## Links & license

Theory preprint: [OSF](https://osf.io/xc7e4) (inspiration for the retention law;
the model neither validates nor is validated by the physics).

**GNU AGPL-3.0.** Network deployment of this architecture or derivatives
requires open-sourcing modifications under the same license.
