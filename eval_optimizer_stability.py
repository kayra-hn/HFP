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
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
from transformers import AutoTokenizer
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
from hfp.core.physics_optimizers import AdamW_Thermodynamic, StiffTransientScheduler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model_and_optimizer(optimizer_type, lr, max_steps, device='cuda'):
    # Small 10M parameter model for fast testing
    config = HFPConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        short_len=8,
        bulk_dim=32,
        max_position_embeddings=2048
    )
    model = HFPForCausalLM(config).to(device)
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        # For standard learning rate (convergence test), use Cosine Annealing
        # For stress test (LR=0.5), we don't use scheduler to allow it to crash naturally
        if lr < 0.1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
        else:
            scheduler = None
    elif optimizer_type == 'AdamW_Thermodynamic':
        # Thermodynamic Optimizer automatically adapts to stiff manifolds
        optimizer = AdamW_Thermodynamic(model.parameters(), lr=lr, h_bar=0.1, base_temp=1.0)
        scheduler = StiffTransientScheduler(optimizer, warmup_steps=5, cool_down_factor=0.90)
    else:
        raise ValueError("Unknown optimizer type")
        
    return model, optimizer, scheduler

def get_batch(data, seq_length, batch_size, device):
    if data is None:
        # Synthetic noise (for strict pathalogical stress test)
        input_ids = torch.randint(0, 50257, (batch_size, seq_length), device=device)
        labels = torch.randint(0, 50257, (batch_size, seq_length), device=device)
        return input_ids, labels
    else:
        # Real linguistic data
        ix = torch.randint(len(data) - seq_length, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

def run_training_loop(model, optimizer, scheduler, data, steps, seq_length, batch_size, device):
    model.train()
    losses = []
    
    for step in range(steps):
        input_ids, labels = get_batch(data, seq_length, batch_size, device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Scheduler steps
        if isinstance(scheduler, StiffTransientScheduler):
            scheduler.step(current_loss=loss.item())
        elif scheduler is not None:
            scheduler.step()
            
        losses.append(loss.item())
        
        # If NaN, model exploded
        if math.isnan(loss.item()) or loss.item() > 100:
            # Fill the rest with NaN for plotting
            losses.extend([float('nan')] * (steps - step - 1))
            break
            
    return losses

def load_data():
    data_path = 'tinyshakespeare.txt'
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Running with synthetic data only.")
        return None
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print("Encoding TinyShakespeare...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = np.array(tokenizer.encode(text))
    print(f"Dataset loaded: {len(data):,} tokens.")
    return data

def run_experiment(scenario, lr, steps, num_seeds, data, seq_length=128, batch_size=4, device='cuda'):
    print(f"\n--- Running {scenario} (LR={lr}) over {num_seeds} seeds ---")
    
    results_adam = []
    results_thermo = []
    
    for seed in range(num_seeds):
        print(f"  Seed {seed+1}/{num_seeds}...")
        
        # Run standard AdamW
        set_seed(seed)
        model_adam, opt_adam, sched_adam = create_model_and_optimizer('AdamW', lr, steps, device)
        losses_adam = run_training_loop(model_adam, opt_adam, sched_adam, data, steps, seq_length, batch_size, device)
        results_adam.append(losses_adam)
        
        # Run Thermodynamic AdamW
        set_seed(seed)
        model_thermo, opt_thermo, sched_thermo = create_model_and_optimizer('AdamW_Thermodynamic', lr, steps, device)
        losses_thermo = run_training_loop(model_thermo, opt_thermo, sched_thermo, data, steps, seq_length, batch_size, device)
        results_thermo.append(losses_thermo)
        
    return np.array(results_adam, dtype=np.float32), np.array(results_thermo, dtype=np.float32)

def plot_shaded(ax, data_matrix, color, label):
    # data_matrix: [num_seeds, steps]
    # np.nanmean ignores the NaNs from explosions for a cleaner plot up to the explosion point
    mean = np.nanmean(data_matrix, axis=0)
    std = np.nanstd(data_matrix, axis=0)
    steps = np.arange(data_matrix.shape[1])
    
    ax.plot(steps, mean, color=color, linewidth=2, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
    
    # Mark explosion if it happened early
    if np.isnan(mean).any():
        first_nan = np.argmax(np.isnan(mean))
        if first_nan > 0:
            ax.axvline(x=first_nan, color=color, linestyle='--', alpha=0.5)

def plot_combined_results(adam_stress, thermo_stress, adam_conv, thermo_conv):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Pathological Stress Test ---
    plot_shaded(ax1, adam_stress, color='#ff4757', label='Standard AdamW (Explodes)')
    plot_shaded(ax1, thermo_stress, color='#2ed573', label='AdamW_Thermodynamic (Survives)')
    ax1.set_title("Crash Test: Pathological LR (0.5)\nSynthetic Data", fontsize=14, color='white')
    ax1.set_xlabel("Training Steps", fontsize=12, color='white')
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12, color='white')
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.legend(loc='upper right')
    
    # --- Subplot 2: Realistic Convergence Test ---
    plot_shaded(ax2, adam_conv, color='#ff4757', label='AdamW + CosineAnnealing')
    plot_shaded(ax2, thermo_conv, color='#2ed573', label='AdamW_Thermodynamic')
    ax2.set_title("Legitimate Learning: Normal LR (5e-4)\nReal Data (TinyShakespeare)", fontsize=14, color='white')
    ax2.set_xlabel("Training Steps", fontsize=12, color='white')
    ax2.grid(True, alpha=0.2, color='gray')
    ax2.legend(loc='upper right')
    
    fig.suptitle("Optimizer Validation: Robustness vs. Legitimacy (Multi-Seed Averages)", fontsize=18, y=1.05)
    
    plt.tight_layout()
    plt.savefig('optimizer_stability_results.png', dpi=300, bbox_inches='tight', facecolor='black')
    print("\nSaved high-quality dual plot to 'optimizer_stability_results.png'")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    real_data = load_data()
    
    num_seeds = 5
    
    # 2. Scenario 1: Pathological Stress Test (Synthetic Data, LR=0.5, 50 steps)
    adam_stress, thermo_stress = run_experiment("Pathological Stress Test", lr=0.5, steps=50, num_seeds=num_seeds, data=None, device=device)
    
    # 3. Scenario 2: Realistic Convergence (Real Data, LR=5e-4, 300 steps)
    if real_data is not None:
        adam_conv, thermo_conv = run_experiment("Realistic Convergence Test", lr=5e-4, steps=300, num_seeds=num_seeds, data=real_data, device=device)
    else:
        # Fallback to synthetic if tiny shakespeare is missing
        adam_conv, thermo_conv = run_experiment("Realistic Convergence Test (Synthetic fallback)", lr=5e-4, steps=300, num_seeds=num_seeds, data=None, device=device)
        
    # 4. Plot
    plot_combined_results(adam_stress, thermo_stress, adam_conv, thermo_conv)

if __name__ == "__main__":
    main()
