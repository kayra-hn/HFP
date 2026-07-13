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
import torch.nn as nn
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from transformers import get_cosine_schedule_with_warmup
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
from hfp.core.physics_optimizers import AdamW_Thermodynamic, StiffTransientScheduler
import numpy as np
import os
import argparse
import csv
import math

def get_batch(data, seq_length, batch_size, device):
    # [FIX M1 - CIFT KAYDIRMA] Hem HFPForCausalLM hem GPT2LMHeadModel labels'i
    # ICERIDE kaydirir (HF konvansiyonu). Eskiden y=data[i+1:...] veriliyordu ->
    # hedef fiilen x[t+2] oluyordu ("skip-one"); tum train.py PPL'leri bu hedefle
    # olculmustu. Dogrusu labels=x: icerdeki kaydirma next-token'i kurar.
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    return x.to(device), x.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, seq_length, batch_size, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, seq_length, batch_size, device)
            outputs = model(X, labels=Y)
            losses[k] = outputs.loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_sample(model, tokenizer, device, max_new_tokens=50):
    model.eval()
    context = "The meaning of life is"
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    if hasattr(model, 'hfp') and hasattr(model.hfp, 'bulk_states'):
        for b_state in model.hfp.bulk_states:
            b_state.reset_state()
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_k=40
    )
    out = tokenizer.decode(generated[0].tolist())
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description="Train Baseline vs HFP on TinyShakespeare")
    parser.add_argument("--model", type=str, choices=['hfp', 'gpt2'], default='hfp')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    # [C4] Default: standart AdamW + cosine warmup (sağlıklı, kanıtlanmış).
    # 'thermodynamic' = fizik-ilhamlı opsiyonel knob (lineer relaksasyon damping).
    parser.add_argument("--optimizer", type=str, choices=['adamw', 'thermodynamic'], default='adamw')
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_tag", type=str, default=None, help="Custom tag for log file naming")
    args = parser.parse_args()

    # Seed ayarla (reproducibility)
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data_path = 'tinyshakespeare.txt'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} is missing. Please download it first.")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Metni kucuk parcalar halinde tokenize ederek bellek sorunlarini/uyarilarini onle
    chunk_size = 100000
    data_list = []
    for i in range(0, len(text), chunk_size):
        data_list.extend(tokenizer.encode(text[i:i+chunk_size], truncation=False))
    data = np.array(data_list, dtype=np.int64)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    vocab_size = len(tokenizer)
    hidden_size = 256
    num_layers = 4
    num_heads = 4

    print(f"Initializing {args.model.upper()} model...")
    if args.model == 'gpt2':
        config = GPT2Config(
            vocab_size=vocab_size, n_embd=hidden_size, n_layer=num_layers,
            n_head=num_heads, n_positions=args.seq_length
        )
        model = GPT2LMHeadModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_iters)
        sched_kind = 'cosine'
    else:
        config = HFPConfig(
            vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_layers,
            num_attention_heads=num_heads, max_position_embeddings=args.seq_length,
            short_len=8, bulk_dim=32,
            decay_mode="cubic_flux_chunked",
            key_feature_map="dpfp",
            write_rule="additive",   # K2 karariyla kilitlendi (RESULTS §13)
            ffn_type="standard",
            rec_block=16
        )
        model = HFPForCausalLM(config)
        if args.optimizer == 'thermodynamic':
            # [C4] Opsiyonel fizik-ilhamlı yol (A/B test için)
            optimizer = AdamW_Thermodynamic(model.parameters(), lr=args.learning_rate)
            scheduler = StiffTransientScheduler(optimizer, warmup_steps=args.warmup_steps)
            sched_kind = 'stiff'
        else:
            # [C4] DEFAULT: sağlıklı, kanıtlanmış AdamW + cosine warmup
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_iters)
            sched_kind = 'cosine'

    model.to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M | optimizer={args.optimizer} scheduler={sched_kind}")

    best_val_loss = float('inf')
    patience_counter = 0
    
    log_file = f"{args.log_tag}_log.csv" if args.log_tag else f"{args.model}_log.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss', 'val_perplexity'])

    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, 50, args.seq_length, args.batch_size, device)
            val_ppl = math.exp(losses['val'])
            print(f"\nStep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val ppl {val_ppl:.2f}")
            sample = generate_sample(model, tokenizer, device, max_new_tokens=30)
            print(f"--- Sample Generation ---\n{sample}\n-------------------------")
            
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([iter, losses['train'], losses['val'], val_ppl])
                
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                torch.save(model.state_dict(), f"{args.model}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at step {iter}. Best val loss: {best_val_loss:.4f}")
                    break

        xb, yb = get_batch(train_data, args.seq_length, args.batch_size, device)
        outputs = model(xb, labels=yb)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            if sched_kind == 'stiff':
                scheduler.step(current_loss=loss.item())
            else:
                scheduler.step()

    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
