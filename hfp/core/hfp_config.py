# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import dataclasses

@dataclasses.dataclass
class HFPConfig:
    """Feature flags and hyper-parameters for optional physics-inspired analogues.

    All flags are *disabled* by default to keep the baseline model clean, fast
    and honest: these physics-inspired aux terms are experimental hooks, NOT
    load-bearing parts of the trained model. Enable one only to *test* whether
    it adds value; when off, no wasted compute and no false "physics" claim.
    (Onceki surumde hepsi True idi ama modeling bunlari loss'a hic baglamiyordu
    -> olu hesap. Durust baseline icin kapatildi; ilham olarak deneye acik kalir.)
    """
    # Feature toggles - deneysel, default kapali (opt-in)
    ENABLE_CURVATURE: bool = False
    ENABLE_ENTROPY_MAP: bool = False
    ENABLE_DEFECT_FLAG: bool = False
    ENABLE_COHERENCE: bool = False
    ENABLE_CONSERVATION: bool = False
    ENABLE_RYU_TAKAYANAGI: bool = False
    ENABLE_5D_CURVATURE: bool = False

    # Hyper-parameters (used when the feature is enabled)
    REG_WEIGHT: float = 0.01               # gate-entropy regularisation weight
    LANDMARK_MAX: int = 49                 # max entries in landmark buffer
    ENTROPY_THRESH: float = 0.25           # threshold for dynamic short-memory expansion
    MAX_SHORT_LEN: int = 32                # maximum short-memory length (tokens)
    GRAD_CLIP_VAL: float = 0.5             # gradient-clipping value per memory block
    MIXED_PRECISION: bool = True           # use torch.float16 for gate logits only
    WARP_K: float = 0.5                    # Witten propagator warp factor

# Global singleton configuration used throughout the package
config = HFPConfig()
