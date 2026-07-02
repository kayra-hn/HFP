import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
from .hfp_utils import LandmarkBuffer, compute_gate_entropy, coherence_score
from .hfp_config import config as hfp_config

class HFPBulkState(nn.Module):
    """
    HFPBulkState V2.0: Holographic Associative Memory Matrix System.
    Fixed Linear Attention Crosstalk by separating Q, K, V projections
    and introducing Thermodynamic Matrix Decay.
    """

    def __init__(self, hidden_size, short_len=8, medium_freq=32, long_freq=128,
                 medium_momentum=0.1, use_mixed_precision=False, clip_value=1.0):
        super(HFPBulkState, self).__init__()
        self.hidden_size = hidden_size
        self.base_short_len = short_len
        self.short_len_dynamic = short_len
        self.landmark_max = hfp_config.LANDMARK_MAX
        self.gate_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.use_mixed_precision = use_mixed_precision
        self.clip_value = clip_value
        self.dynamic_short_thresh = hfp_config.ENTROPY_THRESH
        self.max_short_len = hfp_config.MAX_SHORT_LEN

        # Independent Projections (Expressivity Fix)
        # Gating mechanism for holographics
        self.importance_gate = nn.Linear(hidden_size, hidden_size)
        self.gate_dropout = nn.Dropout(0.1)

        # Linear Attention Projections
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        # Learnable decay factor for forget gate
        self.decay = nn.Parameter(torch.zeros(hidden_size))

        self.landmark_buffer = LandmarkBuffer(max_size=hfp_config.LANDMARK_MAX)

    def get_initial_state(self, batch_size, device, dtype):
        # memory_matrix: [batch, hidden_size, hidden_size]
        M = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        # normalization vector: [batch, hidden_size]
        z = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return (None, M, z, 0, self.base_short_len)

    def reset_state(self):
        self.landmark_buffer.clear()
        if hasattr(self, "_last_gate"):
            del self._last_gate

    def _maybe_expand_short_memory(self, gate_entropy, short_len_dynamic, short_memory):
        if gate_entropy < self.dynamic_short_thresh and short_len_dynamic < self.max_short_len:
            new_len = min(short_len_dynamic + 4, self.max_short_len)
            if short_memory is not None:
                batch, _, hidden = short_memory.shape
                new_buf = torch.zeros(batch, new_len, hidden,
                                      device=short_memory.device,
                                      dtype=short_memory.dtype)
                new_buf[:, :short_memory.size(1), :] = short_memory
                return new_len, new_buf
            return new_len, short_memory
        return short_len_dynamic, short_memory

    def gate_entropy_loss(self):
        if not hasattr(self, "_last_gate"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if hfp_config.ENABLE_ENTROPY_MAP:
            return compute_gate_entropy(self._last_gate) * hfp_config.REG_WEIGHT
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def update(self, x, past_state=None, detach_state=True):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        if past_state is not None:
            (short_memory, M, z, token_count, short_len_dynamic) = past_state
            if short_memory is not None and short_memory.size(0) != batch_size:
                (short_memory, M, z, token_count, short_len_dynamic) = self.get_initial_state(batch_size, device, dtype)
        else:
            (short_memory, M, z, token_count, short_len_dynamic) = self.get_initial_state(batch_size, device, dtype)

        if detach_state:
            if short_memory is not None: short_memory = short_memory.detach()
            if M is not None: M = M.detach()
            if z is not None: z = z.detach()

        # 1. Update Short Memory
        if short_memory is None:
            short_memory = x
        else:
            short_memory = torch.cat([short_memory, x], dim=1)
            
        if short_memory.size(1) > short_len_dynamic:
            short_memory = short_memory[:, -short_len_dynamic:, :]
            
        token_count += seq_len
        
        # 2. Extract Q, K, V Projections
        Q = F.elu(self.W_q(x)) + 1.0  # Must be strictly positive
        K = F.elu(self.W_k(x)) + 1.0
        V_raw = self.W_v(x)
        
        # retrieved: [batch, seq, hidden]
        # Q: [batch, seq, hidden], M: [batch, hidden, hidden]
        num = torch.bmm(Q, M)
        den = (Q * z.unsqueeze(1)).sum(dim=-1, keepdim=True) + 1e-6
        retrieved_memory = num / den

        # 3. Gating Logic & Physics Warp
        gate_logits = self.importance_gate(x) / self.gate_temperature
        use_mixed_precision = hfp_config.MIXED_PRECISION and batch_size > 16
        if use_mixed_precision:
            gate_logits = gate_logits.half()
            
        gate = torch.sigmoid(self.gate_dropout(gate_logits))
        gate = gate.to(M.dtype)
        self._last_gate = gate.clone().detach()
        
        gate_entropy = None
        if hfp_config.ENABLE_ENTROPY_MAP or hfp_config.ENABLE_DEFECT_FLAG or hfp_config.ENABLE_RYU_TAKAYANAGI:
            gate_entropy = compute_gate_entropy(gate)
            
        if gate_entropy is not None:
            warp_factor = torch.exp(-hfp_config.WARP_K * gate_entropy)
            gate_eff = gate * warp_factor
        else:
            gate_eff = gate
            
        # 4. Update M and z with current step (Outer Product with Decay)
        # V is modulated by gate_eff
        V = V_raw * gate_eff
        M_update = torch.bmm(K.transpose(1, 2), V)
        
        # Apply learned decay to prevent catastrophic interference
        # decay_factor shape: [hidden], we decay across the key dimension
        decay_factor = torch.sigmoid(self.decay)
        M_new = M * decay_factor.unsqueeze(0).unsqueeze(2) + M_update
        z_new = z * decay_factor.unsqueeze(0) + K.sum(dim=1)

        # 6. Dynamic short-memory & Landmarks
        if gate_entropy is not None:
            short_len_dynamic, short_memory = self._maybe_expand_short_memory(gate_entropy, short_len_dynamic, short_memory)

        if hfp_config.ENABLE_DEFECT_FLAG:
            coherence = None
            if hfp_config.ENABLE_COHERENCE:
                coherence = coherence_score(short_memory)
            
            if gate_entropy is not None and coherence is not None:
                priority = coherence.item() * gate_entropy.item()
            else:
                priority = gate.mean().item()
                
            self.landmark_buffer.push(priority, short_memory.mean(dim=1))

        new_past_state = (short_memory, M_new, z_new, token_count, short_len_dynamic)
        return short_memory, retrieved_memory, new_past_state

if __name__ == "__main__":
    batch_size = 2
    hidden_size = 512
    seq_length = 200
    memory_system = HFPBulkState(hidden_size=hidden_size)
    dummy_input = torch.randn(batch_size, seq_length, hidden_size)
    short_mem, retrieved_mem, past_state = memory_system.update(dummy_input)
    print(f"Total tokens processed: {past_state[3]}")
    print(f"Short Memory shape: {short_mem.shape}")
    print(f"Retrieved Memory shape: {retrieved_mem.shape}")
    print(f"M Matrix shape: {past_state[1].shape}")
