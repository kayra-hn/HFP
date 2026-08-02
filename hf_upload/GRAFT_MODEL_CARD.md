---
license: other
license_name: apache-2.0-base-agpl-3.0-adapter
base_model: Qwen/Qwen2.5-1.5B
library_name: transformers
tags:
- pytorch
- causal-lm
- linear-attention
- long-context
- o1-memory
- hfp
- grafting
language:
- en
---

<!--
  TASLAK — yayin oncesi doldurulacak alanlar [SONUC:...] ile isaretli.
  Yayin akisi: SONRAKI_ADIMLAR_PLANI.md Bolum 4.6 / 5.
-->

# Qwen2.5-1.5B-HFP-O1 — O(1)-Memory Grafted Hybrid (experimental)

**Qwen2.5-1.5B with half of its attention layers surgically replaced by HFP
recurrent memory modules** — constant-size per-layer state instead of a
growing KV-cache on those layers. Only the HFP parameters (~325K, <0.03% of
the model) were trained; everything else is the frozen original.

> **Honesty note.** This is an experimental research artifact, not a
> production model. All claims below are backed by the linked experiment
> records; pre-registered criteria and outcomes are in the
> [HFP repository](https://github.com/kayra-hn/HFP).

## What was done

- **Grafting:** layers [1,3,...,25] (13/28) of Qwen2.5-1.5B swapped for
  `HFPGraftAttention` (per-head recurrent memory: M ∈ ℝ^{key_dim×head_dim}
  per head; DPFP feature map; GQA-aware K/V expansion; RoPE bypassed on
  grafted layers; q/k/v/o projections warm-started from the base model and
  frozen).
- **Write rule:** alpha-gate hybrid `M += β·k(v − α·v_old)ᵀ` with learnable
  per-head α ∈ (0,1) interpolating additive (archival) ↔ delta (updating)
  writes.
- **Distillation:** Stage 1 — teacher-free layerwise MSE against the original
  attention outputs (teacher-forcing inside the same forward). Stage 2 —
  logit KL + LM loss. Data: [SONUC: veri + token sayisi].

## Results (pre-registered criteria)

| Check | Criterion | Result |
|---|---|---|
| Zero-shot sanity after graft | PPL < 1000 before any training | [SONUC] |
| Short-context degradation | ≤ 1.05× original WikiText PPL | [SONUC] |
| Needle-in-haystack (streaming, O(1) state) | found at 2K/8K/16K | [SONUC] |
| Constant memory | grafted state size independent of context length | [SONUC: MB] |
| Alpha-gate distribution | (exploratory) archive vs working-memory heads | [SONUC: dagilim] |

## Usage

```python
# 1) base model + HFP kodu
from transformers import AutoModelForCausalLM, AutoTokenizer
from hfp.models.grafting import GraftConfig, graft_llama, set_graft_mode

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
graft_llama(model, GraftConfig(decay_mode="cubic_flux_chunked", write_rule="hybrid"))

# 2) bu repodaki HFP parametrelerini yukle (yalnizca ~325K param)
import torch
sd = torch.load("hfp_graft_final.pt", map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
set_graft_mode(model, "student")
```

Streaming (O(1) grafted layers) — see `enable_streaming` in
`hfp/models/grafting.py` and the needle test in
`notebooks/colab_graft_qwen_v3_kaggle.ipynb`.

## Licensing

- **Base model** (Qwen2.5-1.5B weights, frozen): Apache 2.0 (© Alibaba Cloud).
- **HFP adapter parameters + grafting code** (this repo's additions):
  **AGPL-3.0** (© 2026 Kayrahan Yılmaz). Network deployment of derivatives
  requires open-sourcing modifications under AGPL-3.0.
- This repository distributes **only the HFP adapter parameters**; the base
  model is referenced, not redistributed.

## Links

- Code / experiments: https://github.com/kayra-hn/HFP
- Architecture results record: `RESULTS.md` (EN), `docs/tr/DENEY_SONUCLARI.md` (TR)
- Small-scale from-scratch model: https://huggingface.co/kayrahan35/HFP-O1-Memory-Model
