import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
from .hfp_utils import LandmarkBuffer, compute_gate_entropy, coherence_score
from .hfp_config import config as hfp_config


class HFPBulkState(nn.Module):
    """
    HFPBulkState: Hierarchical memory system for large language models.
    Enhancements:
    - Gate temperature scaling and entropy regularization.
    - Dynamic short‑memory length.
    - Landmark priority buffer (capacity 49).
    - Gradient clipping per update.
    - Optional mixed‑precision for gate logits.
    """

    def __init__(self, hidden_size, short_len=8, medium_freq=32, long_freq=128,
                 medium_momentum=0.1, use_mixed_precision=False, clip_value=1.0):
        super(HFPBulkState, self).__init__()
        self.hidden_size = hidden_size
        self.base_short_len = short_len
        self.short_len_dynamic = short_len  # can grow during training
        self.medium_freq = medium_freq
        self.long_freq = long_freq
        self.medium_momentum = medium_momentum
        self.landmark_max = hfp_config.LANDMARK_MAX
        self.gate_temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.use_mixed_precision = use_mixed_precision
        self.clip_value = clip_value
        self.dynamic_short_thresh = hfp_config.ENTROPY_THRESH
        self.max_short_len = hfp_config.MAX_SHORT_LEN

        # Gating mechanism for long‑term memory
        self.importance_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_dropout = nn.Dropout(0.1)

        # Landmark buffer stores (gate_strength, token_summary) pairs
        self.landmark_buffer = LandmarkBuffer(max_size=hfp_config.LANDMARK_MAX)

    def get_initial_state(self):
        """Returns the initial empty state tuple."""
        return (None, None, None, 0, 0, 0, self.base_short_len)

    def _get_short_view(self, short_memory, fill_count, short_len_dynamic):
        """Return the filled portion of the short‑memory ring buffer."""
        if short_memory is None:
            return None
        if self.training:
            return short_memory
        if fill_count < short_len_dynamic:
            return short_memory[:, :fill_count, :]
        return short_memory

    def _get_short_mean(self, short_memory, fill_count, short_len_dynamic):
        """Mean over the short‑memory view (used as context summary)."""
        view = self._get_short_view(short_memory, fill_count, short_len_dynamic)
        if view is None:
            return None
        return view.mean(dim=1)

    def _maybe_expand_short_memory(self, gate_entropy, short_len_dynamic, short_memory):
        """Increase short‑memory length when gate entropy is low.
        Returns the new short_len_dynamic and possibly re-allocated short_memory.
        """
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
        """Return entropy regularization term for the last gate values."""
        if not hasattr(self, "_last_gate"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if hfp_config.ENABLE_ENTROPY_MAP:
            return compute_gate_entropy(self._last_gate) * hfp_config.REG_WEIGHT
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def update(self, x, past_state=None, detach_state=True):
        """Update memories with a new token sequence ``x``.
        Returns short view, medium memory, long memory and a tuple representing the new state.
        """
        if past_state is not None:
            (short_memory, medium_memory, long_memory,
             token_count, write_idx, fill_count, short_len_dynamic) = past_state
        else:
            (short_memory, medium_memory, long_memory,
             token_count, write_idx, fill_count, short_len_dynamic) = self.get_initial_state()

        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        # ensure state matches batch size
        if short_memory is not None and short_memory.size(0) != batch_size:
            (short_memory, medium_memory, long_memory,
             token_count, write_idx, fill_count, short_len_dynamic) = self.get_initial_state()

        # detach previous graphs (Truncated BPTT)
        if detach_state:
            if short_memory is not None:
                short_memory = short_memory.detach()
            if medium_memory is not None:
                medium_memory = medium_memory.detach()
            if long_memory is not None:
                long_memory = long_memory.detach()

        if medium_memory is None:
            medium_memory = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if long_memory is None:
            long_memory = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

        # Clear landmark buffer at the start of a sequence update if we want fresh landmarks,
        # but typically it's a running buffer. We'll leave it as a class attribute since it's just monitoring.
        
        for i in range(seq_len):
            token = x[:, i:i+1, :]
            if self.training:
                # append to short memory and truncate if needed
                if short_memory is None:
                    short_memory = token
                else:
                    short_memory = torch.cat([short_memory, token], dim=1)
                    if short_memory.size(1) > short_len_dynamic:
                        short_memory = short_memory[:, -short_len_dynamic:, :]
            else:
                # ring buffer update
                if short_memory is None:
                    short_memory = torch.zeros(batch_size, short_len_dynamic,
                                                    self.hidden_size, device=device, dtype=dtype)
                short_memory[:, write_idx, :] = token.squeeze(1)
                write_idx = (write_idx + 1) % short_len_dynamic
                fill_count = min(fill_count + 1, short_len_dynamic)

            token_count += 1
            context_summary = self._get_short_mean(short_memory, fill_count, short_len_dynamic)

            # medium memory update
            if token_count % self.medium_freq == 0:
                medium_memory = (1.0 - self.medium_momentum) * medium_memory + \
                                    self.medium_momentum * context_summary

            # combine medium and short summary for gating
            combined_features = torch.cat([medium_memory, context_summary], dim=-1)
            gate_logits = self.importance_gate(combined_features) / self.gate_temperature
            
            # Activate mixed precision only for large batches
            use_mixed_precision = hfp_config.MIXED_PRECISION and batch_size > 16
            if use_mixed_precision:
                gate_logits = gate_logits.half()
                
            gate = torch.sigmoid(self.gate_dropout(gate_logits))
            
            # [CRITICAL BUG FIX]: Ensure gate is same dtype as long_memory (e.g. back to float32 if mixed precision casted it)
            gate = gate.to(long_memory.dtype)
            
            self._last_gate = gate.clone().detach()
            
            # Conditional computations
            gate_entropy = None
            if hfp_config.ENABLE_ENTROPY_MAP or hfp_config.ENABLE_DEFECT_FLAG or hfp_config.ENABLE_RYU_TAKAYANAGI:
                gate_entropy = compute_gate_entropy(gate)
                
            # [5D INTEGRATION]: Witten Propagator Warp Factor
            if gate_entropy is not None:
                warp_factor = torch.exp(-hfp_config.WARP_K * gate_entropy)
                gate_eff = gate * warp_factor
            else:
                gate_eff = gate
                
            long_memory = (1.0 - gate_eff) * long_memory + gate_eff * context_summary

            # Dynamic short‑memory expansion based on entropy
            if gate_entropy is not None:
                short_len_dynamic, short_memory = self._maybe_expand_short_memory(gate_entropy, short_len_dynamic, short_memory)

            # Update landmark buffer based on gate strength and priority
            if hfp_config.ENABLE_DEFECT_FLAG:
                coherence = None
                if hfp_config.ENABLE_COHERENCE:
                    short_view = self._get_short_view(short_memory, fill_count, short_len_dynamic)
                    coherence = coherence_score(short_view)
                
                if gate_entropy is not None and coherence is not None:
                    priority = coherence.item() * gate_entropy.item()
                else:
                    priority = gate.mean().item()
                    
                self.landmark_buffer.push(priority, context_summary)
                
                if hfp_config.ENABLE_COHERENCE and coherence is not None and coherence.item() < 0.2:
                    logging.warning(f"Low coherence detected (score={coherence.item():.4f}).")

        # [CRITICAL BUG FIX]: Removed torch.nn.utils.clip_grad_norm_ from here. It belongs in train.py

        new_past_state = (short_memory, medium_memory, long_memory,
                          token_count, write_idx, fill_count, short_len_dynamic)
                          
        final_short_view = self._get_short_view(short_memory, fill_count, short_len_dynamic)
        return final_short_view, medium_memory, long_memory, new_past_state

# Example usage
if __name__ == "__main__":
    batch_size = 2
    hidden_size = 512
    seq_length = 200
    memory_system = HFPBulkState(hidden_size=hidden_size)
    dummy_input = torch.randn(batch_size, seq_length, hidden_size)
    short_mem, medium_mem, long_mem, past_state = memory_system.update(dummy_input)
    print(f"Total tokens processed: {past_state[3]}")
    print(f"Short Memory shape: {short_mem.shape}")
    print(f"Medium Memory shape: {medium_mem.shape}")
    print(f"Long Memory shape: {long_mem.shape}")
