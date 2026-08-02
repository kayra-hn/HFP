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

> **Status: Architecture and [PENDING] Pre-trained Weights.**
> This repository contains the Hugging Face-compatible architecture code for HFP. 
> The current checkpoint is a placeholder. Final pre-trained weights from our large-scale Colab LM Ablation test will be uploaded here shortly.
> Canonical source & architecture experiments: **[github.com/kayra-hn/HFP](https://github.com/kayra-hn/HFP)**

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
Highlights of the architecture (patterns seed-robust across 3 seeds):

- **Length generalization**: trained at 160 tokens, the model transfers to
  1280-token streams (8x), with fixed-gap recall *improving* as fact density falls —
  train-short / infer-long is the supported deployment mode of the O(1) state.
- **DPFP capacity axis** (`key_feature_map="dpfp"`): first mechanism with a clear
  advantage — 2-6x baseline accuracy at long gaps under high interference.
- **cubic_flux long-horizon advantage**: In sparse, long-gap regimes (gap ≥ 256), `cubic_flux` dramatically outperforms the exponential baseline (63.9% vs 20.7% recall), validating the polynomial retention hypothesis.
- **Initial LM Viability**: On TinyShakespeare (~16M params), HFP (`cubic_flux` + `dpfp`) outperforms GPT-2 (Transformer baseline).

---

## 🏆 WikiText-2 Language Modeling Ablation (16M Params, 3 Seeds)

The final architecture ablations confirm a massive performance synergy when combining the `cubic_flux` retention law with `DPFP` capacity expansion.

| Configuration | Validation Loss (Avg) | Perplexity (PPL) | Note |
| :--- | :--- | :--- | :--- |
| `exp + additive + elu` | 5.2672 | 193.9 | Baseline |
| `exp + additive + dpfp` | 5.2814 | 196.6 | Capacity interference (+2.7 PPL) |
| `exp + delta + dpfp` | 5.2660 | 193.6 | Delta fixes interference |
| `cubic_flux + delta + dpfp` | 5.2534 | 191.2 | Cubic improves delta |
| **`cubic_flux + additive + dpfp`** | **5.2127** | **183.6** | **Optimal Synergy (-10.3 PPL)** |

**Conclusion:** The `cubic_flux` retention law paired with `additive` writes and `dpfp` feature mapping yields a **10.3 PPL reduction** over the baseline, establishing it as the definitive HFP recipe for scale. Pre-trained weights for this configuration will be uploaded shortly.

---

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

This repository implements the ML architecture defined in **Hyper-Flux Projection Model III**. 

Theory preprints (Papers I and II): [OSF](https://osf.io/xc7e4). 
*Note: The theoretical physics models served as inspiration for the retention law, but this ML architecture is strictly decoupled from the physics. The model neither validates nor is validated by the 5D physics frameworks.*

**GNU AGPL-3.0.** Network deployment of this architecture or derivatives
requires open-sourcing modifications under the same license.
