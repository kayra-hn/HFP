import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .hfp_config import config as hfp_config
from .hfp_utils import compute_curvature, compute_entropy_map, magnitude_defect_flag, coherence_score, conservation_check, holographic_information_bound
from .hfp_bulk_state import HFPBulkState

class HFPLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(HFPLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class TunnelingDropout(nn.Module):
    def __init__(self, p=0.5, tunnel_depth=3, decay_factor=0.8):
        super(TunnelingDropout, self).__init__()
        self.p = p
        self.tunnel_depth = tunnel_depth
        self.decay_factor = decay_factor
        from collections import deque
        self.buffer = deque(maxlen=tunnel_depth)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        mask = (torch.rand_like(x) > self.p).float()
        kept_x = x * mask
        dropped_x = x * (1.0 - mask)
        
        self.buffer.append(dropped_x.detach())
        
        output = kept_x
        if len(self.buffer) == self.tunnel_depth:
            tunneled_x = self.buffer.popleft().to(output.device)
            if tunneled_x.shape == output.shape:
                output = output + tunneled_x * self.decay_factor
            elif tunneled_x.size(-1) == output.size(-1):
                self.buffer.clear()
            else:
                self.buffer.clear()
                
        return output

class EntangledLinear(nn.Module):
    def __init__(self, in_features_A, out_features_A, in_features_B, out_features_B, bulk_dim=128):
        super(EntangledLinear, self).__init__()
        self.max_in = max(in_features_A, in_features_B)
        self.W_bulk = nn.Parameter(torch.randn(bulk_dim, self.max_in) / math.sqrt(self.max_in))
        
        self.P_A = nn.Parameter(torch.randn(out_features_A, bulk_dim) / math.sqrt(bulk_dim))
        self.P_B = nn.Parameter(torch.randn(out_features_B, bulk_dim) / math.sqrt(bulk_dim))
        
        self.bias_A = nn.Parameter(torch.zeros(out_features_A))
        self.bias_B = nn.Parameter(torch.zeros(out_features_B))

    def get_orthogonality_loss(self):
        dot = self.P_A @ self.P_B.t()
        return torch.norm(dot, p='fro')

    def forward_A(self, x):
        if not self.training:
            if not hasattr(self, 'W_A_cache'):
                self.W_A_cache = self.P_A @ self.W_bulk[:, :x.size(-1)]
            W_A = self.W_A_cache
        else:
            if hasattr(self, 'W_A_cache'):
                del self.W_A_cache
            W_A = self.P_A @ self.W_bulk[:, :x.size(-1)]
        return F.linear(x, W_A, self.bias_A)
        
    def forward_B(self, x):
        if not self.training:
            if not hasattr(self, 'W_B_cache'):
                self.W_B_cache = self.P_B @ self.W_bulk[:, :x.size(-1)]
            W_B = self.W_B_cache
        else:
            if hasattr(self, 'W_B_cache'):
                del self.W_B_cache
            W_B = self.P_B @ self.W_bulk[:, :x.size(-1)]
        return F.linear(x, W_B, self.bias_B)

class EntangledFFN(nn.Module):
    def __init__(self, hidden_size, feedforward_dim, bulk_dim=128):
        super(EntangledFFN, self).__init__()
        self.entangled = EntangledLinear(hidden_size, feedforward_dim, feedforward_dim, hidden_size, bulk_dim)
        self.gelu = nn.GELU()
        self.dropout = TunnelingDropout(p=0.2, tunnel_depth=3, decay_factor=0.8)

    def forward(self, x):
        mid = self.entangled.forward_A(x)
        mid = self.gelu(mid)
        mid = self.dropout(mid)
        out = self.entangled.forward_B(mid)
        return out
        
    def get_orthogonality_loss(self):
        return self.entangled.get_orthogonality_loss()

class BulkTriggerDecoderLayer(nn.Module):
    """
    BulkTriggerDecoderLayer V2.0: Holographic Matrix Integration
    """
    def __init__(self, hidden_size, num_heads, feedforward_dim, bulk_dim=128, vocab_size=None, return_aux=False):
        super(BulkTriggerDecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.return_aux = return_aux
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.ffn = EntangledFFN(hidden_size, feedforward_dim, bulk_dim=bulk_dim)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.vocab_size = vocab_size
        if vocab_size is not None:
            self.lm_head = HFPLinear(hidden_size, vocab_size)
        else:
            self.lm_head = None

    def forward(self, x, bulk_state, past_state=None, return_past_state=False, return_aux=None):
        if return_aux is None:
            return_aux = getattr(self, 'return_aux', False)
            
        # 1. Update HFPBulkState to get retrieved holographic memory
        short_mem, retrieved_memory, new_past_state = bulk_state.update(x, past_state=past_state)
        
        aux_losses = []
        
        # [5D INTEGRATION]: Auxiliary physics losses
        if hfp_config.ENABLE_RYU_TAKAYANAGI:
            gate_entropy_tensor = bulk_state.gate_entropy_loss() / hfp_config.REG_WEIGHT if hfp_config.ENABLE_ENTROPY_MAP else torch.tensor(0.0, device=x.device)
            # Use the M matrix from new_past_state (index 1) to calculate Holographic Information Bound
            M_matrix = new_past_state[1]
            rt_loss = holographic_information_bound(gate_entropy_tensor, M_matrix)
            aux_losses.append(rt_loss.mean().unsqueeze(0))
            
        if hfp_config.ENABLE_ENTROPY_MAP:
            aux_losses.append(bulk_state.gate_entropy_loss())
            
        if hfp_config.ENABLE_5D_CURVATURE or hfp_config.ENABLE_CURVATURE:
            # Replaced 5D curvature with standard temporal curvature
            aux_losses.append(compute_curvature(short_mem).unsqueeze(0))
            
        if hfp_config.ENABLE_DEFECT_FLAG:
            aux_losses.append(magnitude_defect_flag(short_mem).mean().unsqueeze(0))
        if hfp_config.ENABLE_COHERENCE:
            aux_losses.append(coherence_score(short_mem).unsqueeze(0))
        if hfp_config.ENABLE_CONSERVATION:
            aux_losses.append(torch.tensor(1.0 if conservation_check(short_mem) else 0.0, device=short_mem.device))
        
        # 2. Local Attention over Brane (Short Memory) ONLY
        if past_state is not None and past_state[0] is not None:
            past_short_mem = past_state[0]
        else:
            past_short_mem = torch.zeros(x.size(0), 1, self.hidden_size, device=x.device, dtype=x.dtype)
            
        memory_bank = torch.cat([x, past_short_mem], dim=1)
        
        seq_len = x.size(1)
        mem_len = memory_bank.size(1)
        
        past_mem_len = mem_len - seq_len
        causal_part = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        past_part = torch.zeros(seq_len, past_mem_len, device=x.device).bool()
        dual_mask = torch.cat([causal_part, past_part], dim=1)
        
        attn_out, _ = self.cross_attention(query=x, key=memory_bank, value=memory_bank, attn_mask=dual_mask)
        
        # [V2.0 HOLOGRAPHIC INTEGRATION]: Add retrieved memory from the 5D Matrix
        attn_out = attn_out + retrieved_memory
        
        x = self.norm1(x + attn_out)
        
        # 4. FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if return_aux:
            aux_losses.append(self.ffn.get_orthogonality_loss().unsqueeze(0))
        
        # 5. Logits
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x
        
        if return_aux:
            return logits, bulk_state, new_past_state, aux_losses
        if return_past_state:
            return logits, bulk_state, new_past_state
        return logits, bulk_state

if __name__ == "__main__":
    batch_size = 2
    hidden_size = 256
    num_heads = 8
    feedforward_dim = 1024
    vocab_size = 50000
    
    layer = BulkTriggerDecoderLayer(
        hidden_size=hidden_size, 
        num_heads=num_heads, 
        feedforward_dim=feedforward_dim, 
        vocab_size=vocab_size
    )
    memory_system = HFPBulkState(hidden_size=hidden_size)
    current_token = torch.randn(batch_size, 1, hidden_size)
    logits, updated_memory = layer(current_token, memory_system)
    
    print(f"Girdi Boyutu: {current_token.shape}")
    print(f"Logits Çıktı Boyutu: {logits.shape}")
