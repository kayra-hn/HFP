# Infinity Artificial Consciousness (IAC)
### Powered by Hyper Flux Projection (HFP) Causal Language Model

HFP (Hyper Flux Projection) is a novel, physics-inspired neural network architecture designed for causal language modeling. It introduces advanced physics concepts into the standard Transformer architecture and optimization process, aiming to achieve more stable, coherent, and physically-grounded learning trajectories.

## Key Innovations

The HFP architecture replaces conventional ad-hoc regularizations with strict mathematical/physical analogues:

- **Hyper Flux Core**: A redesigned state-space that captures long-range dependencies efficiently.
- **Quantized Energy Levels (QuantizedLR)**: A learning rate scheduler inspired by quantum mechanics. Instead of arbitrary continuous decay, the learning rate transitions between discrete, stable energy levels based on loss plateaus.
- **Stiff Transient Scheduler**: An inverse-time decay mechanism that introduces "stiffness" to the learning rate, allowing aggressive initial exploration followed by highly stabilized fine-tuning.
- **Uncertainty Regularizer**: A dynamic regularizer that penalizes chaotic states, enforcing thermodynamic coherence within the hidden states.
- **Curvature & Entropy Maps**: The architecture inherently tracks geometric curvature and gate entropy, emitting warnings when the representation space loses coherence (e.g., `Low coherence detected`).

## Architecture Details
The model (`HFPForCausalLM`) is parameter-equivalent to standard scaled models (e.g., GPT-2 Small, ~124M parameters) but operates with a fundamentally different internal dynamic state (`hfp_bulk_state`), tracking variables such as:
- Short memory boundaries (`short_len`)
- Bulk state dimensions (`bulk_dim`)
- Defect Flags and Conservation Checks

## License
This project is open-sourced under the **AGPL v3 License**. 
*Commercial entities utilizing this architecture over a network are required to open-source their modifications under the same license.* See the [LICENSE](LICENSE) file for more details.
