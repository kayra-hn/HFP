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
        self.landmark_buffer = LandmarkBuffer(max_size=self.landmark_max)

        self.reset_state()

    def reset_state(self):
        """Reset all memory buffers."""
        self.token_count = 0
        self.short_memory = None
        self.medium_memory = None
        self.long_memory = None
        self.write_idx = 0
        self.fill_count = 0
        # reset dynamic short length to base value
        self.short_len_dynamic = self.base_short_len
        self.landmark_buffer.clear()

    def _get_short_view(self):
        """Return the filled portion of the short‑memory ring buffer."""
        if self.training:
            return self.short_memory
        if self.fill_count < self.short_len_dynamic:
            return self.short_memory[:, :self.fill_count, :]
        return self.short_memory

    def _get_short_mean(self):
        """Mean over the short‑memory view (used as context summary)."""
        view = self._get_short_view()
        return view.mean(dim=1)

    def _maybe_expand_short_memory(self, gate_entropy):
        """Increase short‑memory length when gate entropy is low.
        This helps capture long‑range dependencies.
        """
        if gate_entropy < self.dynamic_short_thresh and self.short_len_dynamic < self.max_short_len:
            self.short_len_dynamic = min(self.short_len_dynamic + 4, self.max_short_len)
            # enlarge ring buffer if already allocated
            if self.short_memory is not None:
                batch, _, hidden = self.short_memory.shape
                new_buf = torch.zeros(batch, self.short_len_dynamic, hidden,
                                      device=self.short_memory.device,
                                      dtype=self.short_memory.dtype)
                new_buf[:, :self.short_memory.size(1), :] = self.short_memory
                self.short_memory = new_buf

    def gate_entropy_loss(self):
        """Return entropy regularization term for the last gate values."""
        if not hasattr(self, "_last_gate"):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if hfp_config.ENABLE_ENTROPY_MAP:
            return compute_gate_entropy(self._last_gate) * hfp_config.REG_WEIGHT
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def update(self, x, past_state=None):
        """Update memories with a new token sequence ``x``.
        Returns short view, medium memory, long memory and a tuple representing the new state.
        """
        if past_state is not None:
            (self.short_memory, self.medium_memory, self.long_memory,
             self.token_count, self.write_idx, self.fill_count) = past_state
        elif not self.training and self.token_count == 0:
            self.reset_state()

        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype

        # ensure state matches batch size
        if self.short_memory is not None and self.short_memory.size(0) != batch_size:
            self.reset_state()

        # detach previous graphs (Truncated BPTT)
        if self.short_memory is not None:
            self.short_memory = self.short_memory.detach()
        if self.medium_memory is not None:
            self.medium_memory = self.medium_memory.detach()
        if self.long_memory is not None:
            self.long_memory = self.long_memory.detach()

        if self.medium_memory is None:
            self.medium_memory = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.long_memory is None:
            self.long_memory = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

        for i in range(seq_len):
            token = x[:, i:i+1, :]
            if self.training:
                # append to short memory and truncate if needed
                if self.short_memory is None:
                    self.short_memory = token
                else:
                    self.short_memory = torch.cat([self.short_memory, token], dim=1)
                    if self.short_memory.size(1) > self.short_len_dynamic:
                        self.short_memory = self.short_memory[:, -self.short_len_dynamic:, :]
            else:
                # ring buffer update
                if self.short_memory is None:
                    self.short_memory = torch.zeros(batch_size, self.short_len_dynamic,
                                                    self.hidden_size, device=device, dtype=dtype)
                self.short_memory[:, self.write_idx, :] = token.squeeze(1)
                self.write_idx = (self.write_idx + 1) % self.short_len_dynamic
                self.fill_count = min(self.fill_count + 1, self.short_len_dynamic)

            self.token_count += 1
            context_summary = self._get_short_mean()

            # medium memory update
            if self.token_count % self.medium_freq == 0:
                self.medium_memory = (1.0 - self.medium_momentum) * self.medium_memory + \
                                    self.medium_momentum * context_summary

            # combine medium and short summary for gating
            combined_features = torch.cat([self.medium_memory, context_summary], dim=-1)
            gate_logits = self.importance_gate(combined_features) / self.gate_temperature
            # Activate mixed precision only for large batches
            self.use_mixed_precision = hfp_config.MIXED_PRECISION and batch_size > 16
            if self.use_mixed_precision:
                gate_logits = gate_logits.half()
            gate = torch.sigmoid(self.gate_dropout(gate_logits))
            self._last_gate = gate.clone().detach()
            self.long_memory = (1.0 - gate) * self.long_memory + gate * context_summary

            # Conditional computations
            gate_entropy = None
            if hfp_config.ENABLE_ENTROPY_MAP or hfp_config.ENABLE_DEFECT_FLAG:
                gate_entropy = compute_gate_entropy(gate)
            # Dynamic short‑memory expansion based on entropy
            if gate_entropy is not None:
                self._maybe_expand_short_memory(gate_entropy)

            # Update landmark buffer based on gate strength and priority
            if hfp_config.ENABLE_DEFECT_FLAG:
                # Compute priority using coherence * entropy as per user spec
                # Compute coherence if needed
                coherence = None
                if hfp_config.ENABLE_COHERENCE:
                    coherence = coherence_score(self._get_short_view())
                # Compute priority
                if gate_entropy is not None and coherence is not None:
                    priority = coherence.item() * gate_entropy.item()
                else:
                    # fallback to gate strength only
                    priority = gate.mean().item()
                self.landmark_buffer.push(priority, context_summary)
                # Emit warning if coherence is low when monitoring enabled
                if hfp_config.ENABLE_COHERENCE and coherence is not None and coherence.item() < 0.2:
                    logging.warning(f"Low coherence detected (score={coherence.item():.4f}).")




        # gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=hfp_config.GRAD_CLIP_VAL)

        new_past_state = (self.short_memory, self.medium_memory, self.long_memory,
                          self.token_count, self.write_idx, self.fill_count)
        return self._get_short_view(), self.medium_memory, self.long_memory, new_past_state

# Example usage (unchanged from original)
if __name__ == "__main__":
    batch_size = 2
    hidden_size = 512
    seq_length = 200
    memory_system = HFPBulkState(hidden_size=hidden_size)
    dummy_input = torch.randn(batch_size, seq_length, hidden_size)
    short_mem, medium_mem, long_mem = memory_system.update(dummy_input)
    print(f"Total tokens processed: {memory_system.token_count}")
    print(f"Short Memory shape: {short_mem.shape}")
    print(f"Medium Memory shape: {medium_mem.shape}")
    print(f"Long Memory shape: {long_mem.shape}")
