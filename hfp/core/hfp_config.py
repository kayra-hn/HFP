import dataclasses

@dataclasses.dataclass
class HFPConfig:
    """Feature flags and hyper‑parameters for optional physics‑based analogues.

    All flags are *disabled* by default to keep the baseline model unchanged.
    Toggle a flag to ``True`` to activate the corresponding functionality.
    """
    # Feature toggles (low‑risk first)
    ENABLE_CURVATURE: bool = True
    ENABLE_ENTROPY_MAP: bool = True
    ENABLE_DEFECT_FLAG: bool = True
    ENABLE_COHERENCE: bool = True
    ENABLE_CONSERVATION: bool = True

    # Hyper‑parameters (used when the feature is enabled)
    REG_WEIGHT: float = 0.01               # gate‑entropy regularisation weight
    LANDMARK_MAX: int = 49                 # max entries in landmark buffer
    ENTROPY_THRESH: float = 0.25            # threshold for dynamic short‑memory expansion (more aggressive)
    MAX_SHORT_LEN: int = 32                # maximum short‑memory length (tokens)
    GRAD_CLIP_VAL: float = 0.5             # gradient‑clipping value per memory block (tighter)
    MIXED_PRECISION: bool = True          # use torch.float16 for gate logits only

# Global singleton configuration used throughout the package
config = HFPConfig()
