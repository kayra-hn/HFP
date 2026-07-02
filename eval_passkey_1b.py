import torch
import os
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from hfp import HFPForCausalLM, HFPConfig
import argparse

def generate_passkey_prompt(tokenizer, context_length, depth_pct, passkey="84729"):
    """
    Generates a massive haystack context and inserts the passkey at the specified depth.
    """
    # Create the needle
    needle = f" The secret passkey is {passkey}. "
    question = "\nWhat is the secret passkey? The secret passkey is"
    
    needle_tokens = tokenizer.encode(needle)
    question_tokens = tokenizer.encode(question)
    
    # Calculate how much noise (filler) we need
    # We use random tokens from the vocab as the haystack to ensure no semantic guessing
    total_noise_len = context_length - len(needle_tokens) - len(question_tokens)
    
    # Generate random haystack
    haystack_tokens = torch.randint(100, 10000, (total_noise_len,)).tolist()
    
    # Insert the needle at the specific depth
    insert_idx = int(depth_pct * total_noise_len)
    
    # Combine everything
    prompt_tokens = haystack_tokens[:insert_idx] + needle_tokens + haystack_tokens[insert_idx:] + question_tokens
    
    return torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0), passkey

def evaluate_passkey(model, tokenizer, device, context_length=100000, depths=[0.1, 0.5, 0.9]):
    print(f"\n--- Starting 1B Parameter Passkey Retrieval Test ---")
    print(f"Context Length: {context_length:,} tokens")
    
    model.eval()
    accuracies = []
    
    # Since the 1B model is large, we process the sequence in smaller chunks 
    # (e.g. 256) to strictly maintain O(1) local attention VRAM usage on 8GB GPUs.
    chunk_size = 256
    with torch.no_grad():
        for depth in depths:
            print(f"\nTesting Depth: {depth * 100:.0f}%...")
            
            # For robust statistics, we should test multiple different passkeys per depth
            correct = 0
            num_trials = 5
            
            for trial in range(num_trials):
                # Generate a random 5-digit passkey
                current_passkey = str(random.randint(10000, 99999))
                input_ids, target_str = generate_passkey_prompt(tokenizer, context_length, depth, passkey=current_passkey)
                input_ids = input_ids.to(device)
                
                # Stream the 100K tokens through the model to build the bulk_state
                state = None
                
                # Reset the bulk state for a fresh context
                if hasattr(model, 'bulk_state') and hasattr(model.bulk_state, 'reset_state'):
                    model.bulk_state.reset_state()
                
                # Process in chunks
                for i in range(0, input_ids.size(1), chunk_size):
                    chunk = input_ids[:, i:i+chunk_size]
                    # We only care about the logits at the very last step, but we must forward all tokens
                    outputs = model(chunk, past_key_values=state, use_cache=True)
                    logits = outputs.logits
                    state = outputs.past_key_values
                
                # The final prediction is the argmax of the last token's logits
                final_logits = logits[:, -1, :] # Shape: [1, vocab_size]
                predicted_token_id = final_logits.argmax(dim=-1).item()
                predicted_word = tokenizer.decode([predicted_token_id]).strip()
                
                # Because tokenization of "84729" might be multiple tokens, 
                # in a real test we might decode a few tokens. 
                # For this exact matching test, we check if the prediction starts the passkey correctly.
                # Or we check if the target string is in the predicted word (e.g. if the tokenizer groups numbers)
                
                # Simple heuristic: Does the model output the first part of the passkey?
                target_first_token = tokenizer.encode(" " + current_passkey)[0]
                
                if predicted_token_id == target_first_token:
                    correct += 1
                    
            acc = (correct / num_trials) * 100.0
            accuracies.append(acc)
            print(f"-> Accuracy at {depth*100:.0f}% depth: {acc:.1f}%")
            
    return accuracies

def plot_results(depths, accuracies, context_length):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    plt.plot([d * 100 for d in depths], accuracies, marker='o', markersize=10, linewidth=3, color='#8c564b', label='HFP 1B Model')
    
    plt.title(f"Passkey Retrieval (Needle in a Haystack)\nContext Length: {context_length:,} Tokens")
    plt.xlabel("Depth in Context (%)")
    plt.ylabel("Retrieval Accuracy (%)")
    plt.ylim(-5, 105)
    plt.xlim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    
    plot_path = "passkey_1b_results.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved successfully to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="hfp_1b_final.pt", help="Path to trained 1B weights")
    parser.add_argument("--context_length", type=int, default=100000, help="Number of tokens to test")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Loading 1B Model Architecture...")
    config = HFPConfig.from_1b_profile(vocab_size=len(tokenizer))
    
    # We load it in bfloat16 to fit in reasonable VRAM for evaluation
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = HFPForCausalLM(config).to(device=device, dtype=dtype)
    
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print(f"\n[ACADEMIC DISCLOSURE]: Weights file '{args.weights}' not found!")
        print("Running with UNTRAINED architecture to verify O(1) Memory Execution Flow and Associative Matrix dimensions.")
        print("NOTE: Because the model is not trained, it cannot semantically retrieve the passkey.")
        print("This benchmark proves SYSTEM STABILITY and CAPACITY at 100K context, not linguistic accuracy.\n")
        
    depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    accs = evaluate_passkey(model, tokenizer, device, context_length=args.context_length, depths=depths)
    
    plot_results(depths, accs, args.context_length)
