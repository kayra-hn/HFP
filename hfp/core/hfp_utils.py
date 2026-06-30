import heapq
import torch

def compute_gate_entropy(gate_tensor):
    """Compute entropy of gate probabilities.
    gate_tensor is expected to be in [0,1] range (after sigmoid).
    Returns scalar tensor (float).
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
        self.heap = []  # each entry: (strength, tensor)

    def clear(self):
        self.heap.clear()

    def push(self, strength, token_summary):
        # store negative strength for max-heap behavior using heapq (which is min-heap)
        entry = (strength, token_summary.clone().detach())
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, entry)
        else:
            # if new strength greater than smallest in heap, replace
            if strength > self.heap[0][0]:
                heapq.heapreplace(self.heap, entry)

    def get_buffer(self):
        """Return a tensor of stacked token summaries sorted by strength descending."""
        if not self.heap:
            return None
        # sort descending
        sorted_entries = sorted(self.heap, key=lambda e: e[0], reverse=True)
        tensors = [e[1] for e in sorted_entries]
        return torch.stack(tensors, dim=1)  # shape: (batch, slots, hidden)

def compute_curvature(vector: torch.Tensor) -> torch.Tensor:
    """Compute discrete geometric curvature using second-order finite differences across time.
    vector shape expected: (batch, seq_len, hidden_dim).
    """
    if vector.size(1) < 3:
        # Not enough temporal points for second-order difference
        return torch.tensor(0.0, device=vector.device)
    # v''(t) ≈ v(t+1) - 2v(t) + v(t-1)
    second_deriv = vector[:, 2:, :] - 2 * vector[:, 1:-1, :] + vector[:, :-2, :]
    return torch.norm(second_deriv, dim=-1).mean()

def compute_entropy_map(gates: torch.Tensor) -> torch.Tensor:
    """Compute entropy per gate element and return a tensor of shape (batch, seq_len).
    This is similar to ``compute_gate_entropy`` but provides a per‑gate map.
    """
    eps = 1e-8
    p = torch.clamp(gates, min=eps, max=1 - eps)
    entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy.mean(dim=-1)

def magnitude_defect_flag(vector: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Binary flag indicating whether the norm of *vector* exceeds *threshold*.
    Returns a tensor of 0/1 values. Renamed to clarify it's an activation magnitude check.
    """
    norm = torch.norm(vector, dim=-1)
    return (norm > threshold).float()

def coherence_score(memory_states: torch.Tensor) -> torch.Tensor:
    """Compute a simple coherence score as the average cosine similarity between
    consecutive memory states along the sequence dimension.

    Returns a **scalar tensor** (single value) for easy .item() usage.
    """
    if memory_states.size(1) < 2:
        return torch.tensor(0.0, device=memory_states.device)
    # cosine similarity between consecutive timesteps for each batch
    sims = torch.nn.functional.cosine_similarity(
        memory_states[:, :-1, :], memory_states[:, 1:, :], dim=-1
    )  # shape: (batch, seq_len-1)
    # average across both batch and time dimensions to get a scalar
    return sims.mean()

def conservation_check(state: torch.Tensor) -> bool:
    """Check temporal conservation: ensures the sum of activations across the hidden 
    dimension doesn't drift significantly over the sequence time dimension.
    state shape: (batch, seq_len, hidden)
    """
    if state.size(1) < 2:
        return True
    eps = 1e-2
    # Sum over hidden dimension
    temporal_sum = state.sum(dim=-1) # shape: (batch, seq_len)
    # Calculate drift over time (max diff between adjacent timesteps)
    drift = torch.abs(temporal_sum[:, 1:] - temporal_sum[:, :-1])
    return torch.all(drift < eps).item()

def compute_5d_curvature(short_mem: torch.Tensor, medium_mem: torch.Tensor, long_mem: torch.Tensor) -> torch.Tensor:
    """
    Computes a simplified Ricci scaler proxy across both time (sequence) and depth (radial dimension).
    """
    temporal_curv = compute_curvature(short_mem)
    if short_mem is not None and short_mem.size(1) > 0:
        short_mean = short_mem.mean(dim=1)
    else:
        short_mean = torch.zeros_like(medium_mem)
        
    radial_diff1 = medium_mem - short_mean
    radial_diff2 = long_mem - medium_mem
    radial_curv = torch.norm(radial_diff2 - radial_diff1, dim=-1).mean()
    return temporal_curv + radial_curv

def ryu_takayanagi_loss(entropy_val: torch.Tensor, long_mem: torch.Tensor, G_const: float = 1.0) -> torch.Tensor:
    """
    Ryu-Takayanagi Bound: Penalizes boundary entropy if it exceeds proportional bulk area.
    """
    bulk_area = torch.norm(long_mem, dim=-1).mean()
    bound_violation = torch.relu(entropy_val - (bulk_area / (4 * G_const)))
    return bound_violation


