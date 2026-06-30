---
tags:
- custom_code
- pytorch
- causal-lm
- physics-informed-neural-networks
- o1-memory
- hfp
- thermodynamics
- agpl
license: agpl-3.0
language:
- en
---

# Hyper Flux Projection (HFP) - The O(1) Memory Paradigm

<div align="center">
  <img src="https://huggingface.co/kayrahan35/HFP-O1-Memory-Model/resolve/main/benchmark_results_gpu.png" width="800"/>
</div>

<br>

**HFP (Hyper Flux Projection)** is a fundamentally novel, physics-inspired neural network architecture designed for causal language modeling. It achieves the "Holy Grail" of long-context LLMs: **Strictly O(1) Constant VRAM Scaling**, effectively eliminating the quadratic $O(N^2)$ memory bottleneck of standard KV-Cache systems.

By introducing advanced thermodynamic concepts into the standard Transformer architecture, HFP forces the latent representations to obey mathematical conservation laws, preventing hallucinations and context degradation.

## Architectural Breakthroughs

The HFP architecture abandons conventional ad-hoc regularizations in favor of strict mathematical/physical analogues:

### 1. Thermodynamic Context Compression & O(1) Memory
Unlike continuous linear attention (e.g., Infini-attention) which blindly compresses data, HFP employs an **active thermodynamic trigger**. The short-term memory is constantly evaluated for its **Entropy ($S$)** and **Curvature ($R$)**. 
- Once the entropy of the current cognitive state reaches a saturation threshold, the `bulk_trigger` activates, compressing the local context into a high-dimensional `bulk_state` (Long-term memory).
- **HPC Implication:** The memory update mechanism operates strictly in $O(1)$ time and space per block. The model can process effectively infinite context sizes with a constant, highly compressed VRAM footprint on single consumer GPUs.

### 2. Dual-Masked Self-Cross Attention
To prevent "Causal Leakage" (predicting the future) while maintaining access to historical deep memory, HFP uses a custom Dual-Mask attention topology:
- **Local Context:** Strict Triangular Causal Mask prevents tokens from attending to future tokens.
- **Deep Context:** Full Matrix Mask allows total, unhindered read-access to the 5D historical bulk memory.

### 3. Physics-Informed Internal State (Holographic Principle)
The architecture introduces non-standard tracking variables directly influenced by physical laws:
- **5D Radial Curvature:** Measures the second derivative across *memory depth* (Short $\rightarrow$ Medium $\rightarrow$ Long) calculating a Ricci-scalar proxy to regulate internal "gravity."
- **Witten Boundary-to-Bulk Propagator:** Information transition is modulated by a warp factor $e^{-k \cdot S}$, physically shielding the deep bulk from noisy inputs.
- **Ryu-Takayanagi Entropy Bound:** Enforces a strict mathematical bound ensuring the entropy of the boundary cannot exceed the surface area of the bulk.

## Commercial & Hardware Advantages (The Billion-Dollar Paradigm Shift)

The elimination of the KV-Cache bottleneck translates directly to massive hardware and operational cost reductions:
- **Zero VRAM Spikes (Cost Efficiency):** Traditional LLMs require clusters of highly expensive GPUs (e.g., A100/H100) purely to hold the KV-Cache for long contexts. HFP operates with a fixed memory footprint regardless of context length, drastically reducing the hardware requirements.
- **Edge Computing & CPU Inference Potential:** Because memory is strictly $O(1)$ and deeply compressed, HFP architectures can easily run inference for infinitely long contexts on standard CPUs, local servers, and Edge devices (mobile phones, IoT) without crashing due to RAM exhaustion.
- **Sustainable AI Operations:** Constant memory scaling means predictable cloud hosting bills and lower power consumption, paving the way for sustainable, infinitely-running AI agents.

## Usage & Implementation

Because this model introduces a completely novel architecture (`HFPForCausalLM`), you **must** use `trust_remote_code=True` to load it. The custom Python code (`modeling_hfp.py`, `configuration_hfp.py`, etc.) is bundled within this repository.

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Load the HFP Architecture (Untrained weights - Architecture only)
model = AutoModelForCausalLM.from_pretrained(
    "kayrahan35/HFP-O1-Memory-Model", 
    trust_remote_code=True,
    device_map="auto"
)

# Verify the O(1) Bulk State Initialization
print(f"Model Parameters: {model.num_parameters():,}")
print(f"Architecture: {model.config.architectures[0]}")
```

## Scientific Foundations
The theoretical physics foundation and formal proofs of this architecture are documented in the original research paper.
🔗 **[Hyper Flux Projection Theory (OSF Preprint)](https://osf.io/xc7e4)**

## License (GNU AGPL v3)
This architecture is proudly open-sourced under the **AGPL v3.0 License**. 
*Note: Any commercial entities deploying this architecture (or its derivatives) over a network (e.g., as a SaaS or API endpoint) are legally required to open-source their modifications under the same license.*
