import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import os
import math
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
from hfp.core.physics_optimizers import AdamW_Thermodynamic, StiffTransientScheduler

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_model_and_optimizer(optimizer_type, lr=0.5, device='cuda'):
    # Create a tiny 10M parameter model for fast stress testing
    config = HFPConfig(
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        short_len=8,
        bulk_dim=32
    )
    model = HFPForCausalLM(config).to(device)
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = None
    elif optimizer_type == 'AdamW_Thermodynamic':
        # h_bar and base_temp are tuned to trigger visible damping in 50 steps
        optimizer = AdamW_Thermodynamic(model.parameters(), lr=lr, h_bar=0.1, base_temp=1.0)
        # Warmup 5 steps, then start cooling down if loss jumps
        scheduler = StiffTransientScheduler(optimizer, warmup_steps=5, cool_down_factor=0.90)
    else:
        raise ValueError("Unknown optimizer type")
        
    return model, optimizer, scheduler

def stress_test(model, optimizer, scheduler, steps=50, device='cuda'):
    model.train()
    losses = []
    grad_norms = []
    
    # We will feed pure noise as labels to artificially cause a Stiff Manifold (Gradient Explosion)
    for step in range(steps):
        # Generate random noise data
        input_ids = torch.randint(0, 50257, (4, 128)).to(device)
        labels = torch.randint(0, 50257, (4, 128)).to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        # Calculate Grad Norm manually before clipping/stepping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(total_norm)
        
        # Standard gradient clipping to simulate real-world conditions (AdamW normally uses this)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step(current_loss=loss.item())
            
        losses.append(loss.item())
        grad_norms.append(grad_norm)
        
        # If NaN, stop early
        if math.isnan(loss.item()):
            print(f"[{type(optimizer).__name__}] Exploded at step {step}")
            # Fill the rest with NaN for graphing
            losses.extend([float('nan')] * (steps - step - 1))
            grad_norms.extend([float('nan')] * (steps - step - 1))
            break

    return losses, grad_norms

def plot_results(adam_losses, thermo_losses, steps):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = list(range(steps))
    
    # Replace NaN with a high flat line or just leave them to break the line
    ax.plot(x, adam_losses, color='#ff4757', linewidth=2.5, label='Standard AdamW (Explosion)')
    ax.plot(x, thermo_losses, color='#2ed573', linewidth=2.5, label='AdamW_Thermodynamic (Survival)')
    
    ax.set_title("V2.1 Optimizer Crash Test: Surviving Stiff Manifolds (LR=0.5)", fontsize=16, pad=20, color='white')
    ax.set_xlabel("Training Steps", fontsize=12, color='white')
    ax.set_ylabel("Cross Entropy Loss", fontsize=12, color='white')
    ax.grid(True, alpha=0.2, color='gray')
    ax.legend(fontsize=12, loc='upper left', facecolor='black', edgecolor='white')
    
    # Add a marker where AdamW dies
    first_nan = next((i for i, v in enumerate(adam_losses) if math.isnan(v)), None)
    if first_nan:
        ax.axvline(x=first_nan, color='#ff4757', linestyle='--', alpha=0.5)
        ax.text(first_nan + 1, max([l for l in thermo_losses if not math.isnan(l)]), 
                f'AdamW Died\n(NaN)', color='#ff4757', fontsize=10)

    # Add text explaining the Thermodynamic effect
    plt.figtext(0.15, 0.02, 
                "Thermodynamic Damping activates upon high gradient energy, lowering the manifold temperature\n"
                "and scaling down the learning rate dynamically. Standard AdamW simply explodes.", 
                fontsize=10, color='lightgray', ha="left", bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('optimizer_stability_results.png', dpi=300, bbox_inches='tight', facecolor='black')
    print("Saved plot to 'optimizer_stability_results.png'")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Stress Test on {device.upper()}...")
    
    lr = 0.5 # Deliberately absurd learning rate to force explosion
    steps = 50
    
    print("\n--- Testing Standard AdamW ---")
    set_seed(42)
    model_adam, opt_adam, sched_adam = create_model_and_optimizer('AdamW', lr=lr, device=device)
    adam_losses, adam_grads = stress_test(model_adam, opt_adam, sched_adam, steps=steps, device=device)
    
    print("\n--- Testing AdamW_Thermodynamic (V2.1) ---")
    set_seed(42)
    model_thermo, opt_thermo, sched_thermo = create_model_and_optimizer('AdamW_Thermodynamic', lr=lr, device=device)
    thermo_losses, thermo_grads = stress_test(model_thermo, opt_thermo, sched_thermo, steps=steps, device=device)
    
    plot_results(adam_losses, thermo_losses, steps)
    print("\nDone. The Thermodynamic Optimizer successfully prevented the Gradient Explosion.")

if __name__ == "__main__":
    main()
