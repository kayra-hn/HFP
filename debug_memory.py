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

import torch
import gc
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

def debug():
    device = torch.device('cuda')
    config = HFPConfig(vocab_size=50257, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=8192)
    model = HFPForCausalLM(config).eval().to(device)
    
    length = 8192
    chunk_size = 256
    dummy_input = torch.randint(0, 50000, (1, length), device=device)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Initial allocated:", torch.cuda.memory_allocated() / (1024*1024), "MB")
    
    hfp_state = None
    with torch.no_grad():
        for i in range(0, length, chunk_size):
            chunk = dummy_input[:, i:i+chunk_size]
            outputs = model(chunk, past_key_values=hfp_state, use_cache=True)
            hfp_state = outputs.past_key_values
            print(f"Step {i}, Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB, Peak: {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB")

if __name__ == "__main__":
    debug()
