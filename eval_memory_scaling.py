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
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

def memory_worker(model_type, length, batch_size, precision, queue):
    try:
        # Import inside worker to ensure fresh CUDA context
        import torch
        from transformers import GPT2LMHeadModel, GPT2Config
        from hfp.models.configuration_hfp import HFPConfig
        from hfp.models.modeling_hfp import HFPForCausalLM
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if precision == 'FP16' else torch.float32
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Determine hidden size and layers for roughly equal parameter counts (~124M)
        hidden_size = 768
        num_layers = 12
        num_heads = 12
        vocab_size = 50257
        
        dummy_input = torch.randint(0, vocab_size, (batch_size, length), device=device)
        
        if model_type == 'GPT2':
            config = GPT2Config(
                vocab_size=vocab_size, n_embd=hidden_size, n_layer=num_layers,
                n_head=num_heads, n_positions=32768
            )
            model = GPT2LMHeadModel(config).to(dtype).to(device).eval()
            
            with torch.no_grad():
                outputs = model(dummy_input, use_cache=True)
                del outputs
                
        elif model_type == 'HFP':
            config = HFPConfig(
                vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_layers,
                num_attention_heads=num_heads, max_position_embeddings=32768,
                short_len=8, bulk_dim=32, intermediate_size=hidden_size * 4,
                MIXED_PRECISION=(precision == 'FP16')
            )
            model = HFPForCausalLM(config).to(dtype).to(device).eval()
            
            chunk_size = 256
            hfp_state = None
            
            with torch.no_grad():
                for i in range(0, length, chunk_size):
                    chunk = dummy_input[:, i:i+chunk_size]
                    outputs = model(chunk, past_key_values=hfp_state, use_cache=True)
                    hfp_state = outputs.past_key_values
                    del outputs
                    
        # Capture metrics
        # Cross validate with memory stats dictionary
        peak_bytes = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        peak_mb = peak_bytes / (1024 * 1024)
        
        # Clean up explicitly before returning
        del model
        del dummy_input
        torch.cuda.empty_cache()
        
        queue.put(('SUCCESS', peak_mb))
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            queue.put(('OOM', 0.0))
        else:
            queue.put(('ERROR', str(e)))
    except Exception as e:
        queue.put(('ERROR', str(e)))

def run_isolated(model_type, length, batch_size, precision):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=memory_worker, args=(model_type, length, batch_size, precision, q))
    p.start()
    p.join(timeout=300) # 5 minutes max
    
    if p.is_alive():
        p.terminate()
        p.join()
        return 'TIMEOUT', 0.0
        
    if not q.empty():
        return q.get()
    else:
        return 'CRASH', 0.0

def plot_memory_results(results_dict, lengths, num_runs, batch_sizes):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(6 * len(batch_sizes), 6))
    if len(batch_sizes) == 1:
        axes = [axes]
        
    for ax, bs in zip(axes, batch_sizes):
        ax.set_title(f"Memory Scaling (Batch Size={bs}, FP16)\n{num_runs}-Run Variance", fontsize=12)
        ax.set_xlabel("Context Length (Tokens)")
        ax.set_ylabel("Peak VRAM Allocated (MB)")
        ax.grid(True, alpha=0.2)
        
        for model in ['GPT2', 'HFP']:
            means = []
            stds = []
            valid_lengths = []
            
            for l in lengths:
                vals = results_dict[bs][model][l]
                valid_vals = [v for v in vals if v is not None]
                if len(valid_vals) > 0:
                    means.append(np.mean(valid_vals))
                    stds.append(np.std(valid_vals))
                    valid_lengths.append(l)
                else:
                    # OOM reached
                    if len(valid_lengths) > 0:
                        last_x = valid_lengths[-1]
                        last_y = means[-1]
                        ax.annotate('OOM', xy=(last_x, last_y), xytext=(last_x, last_y + 1000),
                                    arrowprops=dict(facecolor='red', shrink=0.05), color='red',
                                    fontsize=10, ha='center')
                    break
                    
            if len(valid_lengths) > 0:
                color = '#ff4757' if model == 'GPT2' else '#2ed573'
                ax.errorbar(valid_lengths, means, yerr=stds, label=f"{model}",
                            color=color, marker='o', linewidth=2, capsize=5)
                            
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('benchmark_results_gpu_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive plot to benchmark_results_gpu_comprehensive.png")

if __name__ == '__main__':
    # Modified length sequence as requested: up to 16K
    lengths = [1, 256, 512, 1024, 2048, 4096, 8192, 16384]
    batch_sizes = [1, 4, 8]
    precision = 'FP16'
    num_runs = 5
    
    results = {bs: {'GPT2': {l: [] for l in lengths}, 'HFP': {l: [] for l in lengths}} for bs in batch_sizes}
    
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        for length in lengths:
            for model in ['GPT2', 'HFP']:
                # Skip if already OOM'd in previous length step
                idx = lengths.index(length)
                if idx > 0:
                    prev_l = lengths[idx-1]
                    prev_vals = results[bs][model][prev_l]
                    if len([v for v in prev_vals if v is not None]) == 0:
                        results[bs][model][length] = [None]
                        continue
                    
                run_vals = []
                print(f"Profiling {model} (L={length}, B={bs})...", end='', flush=True)
                for r in range(num_runs):
                    status, peak_mb = run_isolated(model, length, bs, precision)
                    if status == 'SUCCESS':
                        run_vals.append(peak_mb)
                    elif status == 'OOM':
                        print(f" [Run {r+1}: OOM]", end='')
                    else:
                        print(f" [Run {r+1}: {status}]", end='')
                
                if len(run_vals) > 0:
                    results[bs][model][length] = run_vals
                    print(f" Avg Peak: {np.mean(run_vals):.1f} MB (±{np.std(run_vals):.1f})")
                else:
                    print(" Failed/OOM")
                    results[bs][model][length] = [None]
                    
    plot_memory_results(results, lengths, num_runs, batch_sizes)
