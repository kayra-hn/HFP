# Hyper Flux Projection (HFP) Causal Language Model

HFP (Hyper Flux Projection) is a novel, physics-inspired neural network architecture designed for causal language modeling. It introduces advanced physics concepts into the standard Transformer architecture and optimization process, aiming to achieve more stable, coherent, and physically-grounded learning trajectories.

## Key Innovations

The HFP architecture replaces conventional ad-hoc regularizations with strict mathematical/physical analogues:

- **Hyper Flux Core**: A redesigned state-space that captures long-range dependencies efficiently.
- **Quantized Energy Levels (QuantizedLR)**: A learning rate scheduler inspired by quantum mechanics. Instead of arbitrary continuous decay, the learning rate transitions between discrete, stable energy levels based on loss plateaus.
- **Stiff Transient Scheduler**: An inverse-time decay mechanism that introduces "stiffness" to the learning rate, allowing aggressive initial exploration followed by highly stabilized fine-tuning.
- **Uncertainty Regularizer**: A dynamic regularizer that penalizes chaotic states, enforcing thermodynamic coherence within the hidden states.
- **Curvature & Entropy Maps**: The architecture inherently tracks geometric curvature and gate entropy, emitting warnings when the representation space loses coherence (e.g., `Low coherence detected`).

## Architecture Details

The core model, `HFPForCausalLM` (~124M parameters), structurally maps to a standard Causal LM but intercepts and overrides the hidden state propagation using physics-informed modules.

### 1. Thermodynamic Context Compression (`bulk_trigger_decoder`)
Unlike continuous linear attention (e.g., Google's Infini-attention) which blindly compresses data, HFP employs an **active thermodynamic trigger**. The short-term memory is constantly evaluated for its **Entropy** and **Curvature**. Once the entropy of the current cognitive state reaches a saturation threshold, the `bulk_trigger` activates, compressing the local context into a high-dimensional `bulk_state` (Long-term memory). This prevents context dilution and catastrophic forgetting while drastically reducing the $O(N^2)$ attention bottlenecks.

### 2. Physics-Informed Internal State (`hfp_bulk_state` & `hfp_utils`)
The architecture introduces several non-standard tracking variables directly influenced by physical laws:
- **5D Radial Curvature:** Unlike standard models that only measure temporal change, HFP measures the second derivative across its *memory depth* (Short -> Medium -> Long). It calculates a Ricci-scalar proxy to regulate the internal "gravity" of the context window.
- **Witten Boundary-to-Bulk Propagator:** The transition of information from short-term memory (Boundary) to long-term memory (Bulk) is not linear. It is modulated by a warp factor $e^{-k \cdot S}$ based on the entropy (chaos) of the boundary, physically shielding the deep bulk from noisy inputs.
- **Ryu-Takayanagi Entropy Bound:** Inspired by the holographic entanglement entropy formula, the model enforces a strict mathematical bound: the entropy of the boundary cannot exceed the surface area of the bulk. If the model approaches hallucination, a ReLU penalty restricts the gradients.
- **Conservation Checks:** Enforces mathematical conservation laws across hidden states to ensure the model doesn't hallucinate context shifts out of thin air.

### 3. Quantum-Inspired Schedulers (`physics_optimizers.py`)
To solve the instability of LLM training (loss spikes):
- **QuantizedLR:** Instead of continuous cosine decay, the learning rate transitions through discrete "energy levels" (quanta) based on mathematical plateaus.
- **Stiff Transient Scheduler:** Applies "stiffness" (borrowed from stiff ODE systems) to the optimizer. It allows aggressive early exploration but applies immense thermodynamic braking during fine-tuning, preventing the model from collapsing.

## License
This project is open-sourced under the **AGPL v3 License**. 
*Commercial entities utilizing this architecture over a network are required to open-source their modifications under the same license.* See the [LICENSE](LICENSE) file for more details.
