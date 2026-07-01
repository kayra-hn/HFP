import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from hfp import HFPForCausalLM, HFPConfig
from hfp.physics.physics_optimizers import UncertaintyRegularizer, StiffTransientScheduler
import time

# Try to import wandb for cloud logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb is not installed. Cloud logging will be disabled.")

# Try to import bitsandbytes for 8-bit AdamW
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes is not installed. Falling back to standard AdamW. This will consume more VRAM.")

def get_streaming_dataloader(tokenizer, batch_size=4, seq_len=1024):
    """
    Creates an infinite streaming dataloader using HuggingFaceFW/fineweb-edu.
    This prevents downloading terabytes of data to the cloud server disk.
    """
    print("Connecting to HuggingFaceFW/fineweb-edu (Streaming Mode)...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    def tokenize_and_chunk(iterator):
        buffer = []
        for example in iterator:
            tokens = tokenizer(example["text"], truncation=False)["input_ids"]
            buffer.extend(tokens)
            
            # Yield chunks of exactly seq_len
            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)
                
    def batch_generator(token_iterator):
        batch = []
        for chunk in token_iterator:
            batch.append(chunk)
            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []

    token_iterator = tokenize_and_chunk(iter(dataset))
    return batch_generator(token_iterator)

def train_1b_cloud():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- HFP 1B Cloud Training Initializing on {device} ---")
    
    # Hyperparameters
    batch_size = 4
    grad_accum_steps = 8 # Effective batch size = 32
    seq_len = 1024       # Truncated BPTT length for chunked stream
    max_steps = 100000   # Set to desired training steps
    save_interval = 2000
    learning_rate = 3e-4
    
    if WANDB_AVAILABLE:
        wandb.init(project="HFP-1B-Cloud-Training", config={
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "seq_len": seq_len,
            "learning_rate": learning_rate,
            "optimizer": "AdamW8bit" if BNB_AVAILABLE else "AdamW"
        })
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    print("Initializing 1B Parameter HFP Configuration...")
    config = HFPConfig.from_1b_profile(vocab_size=vocab_size)
    model = HFPForCausalLM(config).to(device)
    
    print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Try enabling Gradient Checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient Checkpointing Enabled.")
    
    # Optimizer Selection
    if BNB_AVAILABLE:
        print("Using 8-bit AdamW (bitsandbytes) for 75% VRAM savings on optimizer states.")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    scheduler = StiffTransientScheduler(optimizer, plateau_threshold=5.0, stiffness_p=0.5)
    reg = UncertaintyRegularizer(model, h_bar=0.001)
    
    scaler = torch.cuda.amp.GradScaler() # For Mixed Precision (FP16/BF16)
    
    # Dataloader
    data_generator = get_streaming_dataloader(tokenizer, batch_size=batch_size, seq_len=seq_len)
    
    print("\n🚀 CLOUD TRAINING STARTED...")
    model.train()
    
    start_time = time.time()
    step_loss = 0.0
    
    for step in range(1, max_steps + 1):
        try:
            inputs = next(data_generator).to(device)
        except StopIteration:
            print("Dataset Stream Ended!")
            break
            
        labels = inputs.clone()
        
        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            outputs = model(inputs, labels=labels)
            loss = outputs.loss / grad_accum_steps
            
        # Backward Pass with GradScaler
        scaler.scale(loss).backward()
        step_loss += loss.item() * grad_accum_steps
        
        # Gradient Accumulation Update
        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Physics Regularization
            reg.step()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            actual_step = step // grad_accum_steps
            avg_loss = step_loss / grad_accum_steps
            
            elapsed = time.time() - start_time
            print(f"Step {actual_step} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
            
            if WANDB_AVAILABLE:
                wandb.log({"train_loss": avg_loss, "learning_rate": optimizer.param_groups[0]['lr']})
                
            scheduler.step(avg_loss)
            
            step_loss = 0.0
            start_time = time.time()
            
            # Checkpoint Saving
            if actual_step % save_interval == 0:
                save_path = f"hfp_1b_checkpoint_{actual_step}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"💾 Checkpoint Saved: {save_path}")

    print("Training Completed.")
    torch.save(model.state_dict(), "hfp_1b_final.pt")
    
if __name__ == "__main__":
    train_1b_cloud()
