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

import heapq
import torch

def compute_gate_entropy(gate_tensor):
    """Compute entropy of gate probabilities.
    gate_tensor is expected to be in [0,1] range (after sigmoid).
    Returns scalar tensor (float). DIFFERENTIABLE.
    """
    eps = 1e-8
    p = torch.clamp(gate_tensor, min=eps, max=1 - eps)
    entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy.mean()

class LandmarkBuffer:
    """Priority buffer that keeps the top-k token summaries based on gate strength.
    Uses a min-heap of size max_size; lower strengths are popped.
    """
    def __init__(self, max_size=49):
        self.max_size = max_size
        self.heap = []  # each entry: (strength, counter, tensor)
        self.counter = 0

    def clear(self):
        self.heap.clear()
        self.counter = 0

    def push(self, strength, token_summary):
        entry = (strength, self.counter, token_summary.clone().detach())
        self.counter += 1
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, entry)
        else:
            if strength > self.heap[0][0]:
                heapq.heapreplace(self.heap, entry)

    def get_buffer(self):
        """Return a tensor of stacked token summaries sorted by strength descending."""
        if not self.heap:
            return None
        sorted_entries = sorted(self.heap, key=lambda e: e[0], reverse=True)
        tensors = [e[2] for e in sorted_entries]
        return torch.stack(tensors, dim=1)  # shape: (batch, slots, hidden)

def compute_curvature(vector: torch.Tensor) -> torch.Tensor:
    """Discrete geometric curvature via second-order finite differences across time.
    vector shape: (batch, seq_len, hidden_dim).
    NOT: kaynak tensor detach edilmisse gradyan tasimaz; regularizer olarak kullanilacaksa
    gradyanli bir tensore (or. katman girisi) uygulanmalidir.
    """
    if vector.size(1) < 3:
        return torch.tensor(0.0, device=vector.device)
    second_deriv = vector[:, 2:, :] - 2 * vector[:, 1:-1, :] + vector[:, :-2, :]
    return torch.norm(second_deriv, dim=-1).mean()

def compute_entropy_map(gates: torch.Tensor) -> torch.Tensor:
    """Per-gate entropy map, shape (batch, seq_len)."""
    eps = 1e-8
    p = torch.clamp(gates, min=eps, max=1 - eps)
    entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy.mean(dim=-1)

def magnitude_defect_flag(vector: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """[DIAGNOSTIC ONLY - NON-DIFFERENTIABLE] norm(vector) > threshold.
    Bir '>' karsilastirmasi -> gradyan TASIMAZ. Loss olarak kullanmayin; teshis metrigidir.
    """
    norm = torch.norm(vector, dim=-1)
    return (norm > threshold).float()

def coherence_score(memory_states: torch.Tensor) -> torch.Tensor:
    """Average cosine similarity between consecutive memory states along the sequence dim.
    Returns a scalar tensor.
    """
    if memory_states.size(1) < 2:
        return torch.tensor(0.0, device=memory_states.device)
    sims = torch.nn.functional.cosine_similarity(
        memory_states[:, :-1, :], memory_states[:, 1:, :], dim=-1
    )
    return sims.mean()

def conservation_check(state: torch.Tensor) -> bool:
    """[DIAGNOSTIC ONLY - NON-DIFFERENTIABLE] Python bool dondurur -> gradyan TASIMAZ.
    Temporal 'korunum' teshisi: gizli boyuttaki toplam zaman icinde suruklenmiyor mu?
    Loss olarak kullanmayin; yalnizca izleme metrigidir. state shape: (batch, seq_len, hidden)
    """
    if state.size(1) < 2:
        return True
    eps = 1e-2
    temporal_sum = state.sum(dim=-1)
    drift = torch.abs(temporal_sum[:, 1:] - temporal_sum[:, :-1])
    return torch.all(drift < eps).item()

def holographic_information_bound(entropy_val: torch.Tensor, memory_matrix: torch.Tensor) -> torch.Tensor:
    """Holographic Information Bound (V2.1): soft penalty ensuring current attention entropy
    does not exceed the Frobenius-norm capacity of the [hidden, hidden] memory matrix.
    DIFFERENTIABLE (softplus).
    """
    matrix_capacity = torch.linalg.matrix_norm(memory_matrix, ord='fro', dim=(-2, -1)).mean()
    ratio = entropy_val / (matrix_capacity + 1e-8)
    bound_violation = torch.nn.functional.softplus(ratio - 1.0)
    return bound_violation
