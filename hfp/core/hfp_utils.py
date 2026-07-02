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
        self.heap = []  # each entry: (strength, counter, tensor)
        self.counter = 0

    def clear(self):
        self.heap.clear()
        self.counter = 0

    def push(self, strength, token_summary):
        # store negative strength for max-heap behavior using heapq (which is min-heap)
        entry = (strength, self.counter, token_summary.clone().detach())
        self.counter += 1
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
        tensors = [e[2] for e in sorted_entries]
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

def holographic_information_bound(entropy_val: torch.Tensor, memory_matrix: torch.Tensor) -> torch.Tensor:
    """
    Holographic Information Bound (V2.1):
    Ensures that the entropy of the current attention distribution does not exceed 
    the theoretical capacity (Frobenius norm) of the [hidden, hidden] Associative Memory Matrix.
    """
    # Calculate Frobenius norm over the matrix dimensions ONLY (dim -2 and -1), not batch.
    # memory_matrix is [batch, hidden, hidden]
    matrix_capacity = torch.linalg.matrix_norm(memory_matrix, ord='fro', dim=(-2, -1)).mean()
    
    # Soft Penalty (Log-Barrier / Softplus approach) instead of dead ReLU
    # capacity ratio
    ratio = entropy_val / (matrix_capacity + 1e-8)
    
    # Softplus ensures gradients always flow smoothly (no dead ReLU).
    # It penalizes heavily if entropy approaches or exceeds capacity.
    bound_violation = torch.nn.functional.softplus(ratio - 1.0)
    return bound_violation
