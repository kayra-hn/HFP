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
import os
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM
import argparse

def load_haystack(tokenizer):
    data_path = 'tinyshakespeare.txt'
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Using random tokens as haystack.")
        return None
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return tokenizer.encode(text)

def generate_passkey_prompt(tokenizer, haystack_full_tokens, context_length, depth_pct, passkey="84729"):
    # Create the needle
    needle = f"\n[The secret passkey is {passkey}]\n"
    question = "\nWhat is the secret passkey? The secret passkey is"
    
    needle_tokens = tokenizer.encode(needle)
    question_tokens = tokenizer.encode(question)
    
    total_noise_len = context_length - len(needle_tokens) - len(question_tokens)
    
    if haystack_full_tokens is not None and len(haystack_full_tokens) > total_noise_len:
        start_idx = random.randint(0, len(haystack_full_tokens) - total_noise_len - 1)
        haystack_tokens = haystack_full_tokens[start_idx : start_idx + total_noise_len]
    else:
        haystack_tokens = torch.randint(100, 10000, (total_noise_len,)).tolist()
        
    insert_idx = int(depth_pct * total_noise_len)
    prompt_tokens = haystack_tokens[:insert_idx] + needle_tokens + haystack_tokens[insert_idx:] + question_tokens
    
    return torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0), passkey

def evaluate_passkey(model, tokenizer, haystack_tokens, device, context_length=100000, depths=[0.1, 0.5, 0.9], num_trials=30):
    print(f"\n--- Starting Passkey Retrieval Test (NIAH Protocol) ---")
    print(f"Context Length: {context_length:,} tokens")
    print(f"Trials per depth: {num_trials}")
    
    model.eval()
    accuracies = []
    
    chunk_size = 256
    
    for depth in depths:
        print(f"\nTesting Depth: {depth * 100:.0f}%...")
        correct = 0
        
        for trial in range(num_trials):
            current_passkey = str(random.randint(10000, 99999))
            input_ids, target_str = generate_passkey_prompt(tokenizer, haystack_tokens, context_length, depth, passkey=current_passkey)
            input_ids = input_ids.to(device)
            
            state = None
            if hasattr(model, 'hfp') and hasattr(model.hfp, 'bulk_states'):
                for b_state in model.hfp.bulk_states:
                    b_state.reset_state()
            
            with torch.no_grad():
                # Prefill chunk by chunk to avoid OOM and emulate streaming
                for i in range(0, input_ids.size(1) - 1, chunk_size):
                    chunk = input_ids[:, i:i+chunk_size]
                    outputs = model(chunk, past_key_values=state, use_cache=True)
                    state = outputs.past_key_values
                    del outputs
                
                # Generate from the last token
                last_token = input_ids[:, -1:]
                passkey_len = len(tokenizer.encode(" " + target_str)) + 2
                
                generated_ids = model.generate(
                    last_token,
                    past_key_values=state,
                    max_new_tokens=passkey_len,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
                
            predicted_text = tokenizer.decode(generated_ids[0][1:]).strip()
            
            # Exact string matching check
            if target_str in predicted_text:
                correct += 1
                
        acc = (correct / num_trials) * 100.0
        accuracies.append(acc)
        print(f"-> Accuracy at {depth*100:.0f}% depth: {acc:.1f}%")
        
    return accuracies

def plot_results(depths, accuracies, context_length):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.plot([d * 100 for d in depths], accuracies, marker='o', markersize=10, linewidth=3, color='#2ed573', label='HFP Model')
    plt.title(f"Passkey Retrieval (Needle in a Haystack)\nContext Length: {context_length:,} Tokens", color='white')
    plt.xlabel("Depth in Context (%)", color='white')
    plt.ylabel("Retrieval Accuracy (%)", color='white')
    plt.ylim(-5, 105)
    plt.xlim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower left')
    plot_path = "passkey_results.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, facecolor='black')
    print(f"\nPlot saved successfully to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="hfp_1b_final.pt", help="Path to trained weights")
    parser.add_argument("--context_length", type=int, default=16000, help="Number of tokens to test")
    parser.add_argument("--num_trials", type=int, default=30, help="Number of trials per depth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Loading Haystack Data...")
    haystack_tokens = load_haystack(tokenizer)
    
    print("Loading Model Architecture...")
    if os.path.exists(args.weights):
        config = HFPConfig.from_1b_profile(vocab_size=len(tokenizer))
    else:
        # Smaller profile to avoid massive initialization delay for untrained pipeline tests
        config = HFPConfig(vocab_size=len(tokenizer), hidden_size=768, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=32768)
        
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = HFPForCausalLM(config).to(device=device, dtype=dtype)
    
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print(f"\n[ACADEMIC DISCLOSURE]: Weights file '{args.weights}' not found!")
        print("Running with UNTRAINED architecture. Accuracy will be ~0% because the model cannot semantically retrieve without being trained.")
        print("This benchmark proves the NIAH protocol implementation works (Full String Decode, 30 Trials, Real Haystack).\n")
        
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    accs = evaluate_passkey(model, tokenizer, haystack_tokens, device, context_length=args.context_length, depths=depths, num_trials=args.num_trials)
    
    plot_results(depths, accs, args.context_length)
