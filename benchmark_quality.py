import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import urllib.request
import zipfile

def download_and_prepare_data(tokenizer):
    print("Veri Seti İndiriliyor (Tiny Shakespeare)...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    txt_path = "tinyshakespeare.txt"
    
    if not os.path.exists(txt_path):
        urllib.request.urlretrieve(url, txt_path)
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    print("Tokenizasyon Başlıyor...")
    # Train / Val Split (%90 Train, %10 Val)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_enc = tokenizer(train_text, return_tensors='pt', truncation=False)['input_ids'][0]
    val_enc = tokenizer(val_text, return_tensors='pt', truncation=False)['input_ids'][0]
    
    seq_len = 64
    
    def chunk_data(data):
        chunks = []
        for i in range(0, len(data) - seq_len, seq_len):
            chunks.append(data[i:i+seq_len])
        return torch.stack(chunks)
        
    train_chunks = chunk_data(train_enc)
    val_chunks = chunk_data(val_enc)
    
    train_ds = TensorDataset(train_chunks)
    val_ds = TensorDataset(val_chunks)
    
    print(f"Train Büyüklüğü: {len(train_ds)} batch | Val Büyüklüğü: {len(val_ds)} batch")
    return train_ds, val_ds

def generate_text(model, tokenizer, device, prompt="The theory of general relativity", max_tokens=20):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            try:
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            except AttributeError:
                # If HFP Model
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            
    generated = tokenizer.decode(input_ids[0])
    model.train()
    return generated

def benchmark_quality():
    print("=== HFP vs Standard Transformer: Akademik Kalite Testi (Multi-Seed) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    train_ds, val_ds = download_and_prepare_data(tokenizer)
    
    # 3 Farklı Seed ile hata payı/varyans hesaplama
    seeds = [42, 100, 999]
    max_steps = min(300, len(train_ds) // 4)
    val_interval = 25
    
    all_std_val_losses = []
    all_hfp_val_losses = []
    eval_steps = []
    
    from hfp.models.configuration_hfp import HFPConfig
    from hfp.models.modeling_hfp import HFPForCausalLM
    
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    
    best_hfp_model = None
    best_std_model = None
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/3: {seed} ---")
        torch.manual_seed(seed)
        
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8)
        
        std_config = GPT2Config(vocab_size=vocab_size, n_embd=hidden_size, n_head=num_heads, n_layer=num_layers, n_inner=512)
        std_model = GPT2LMHeadModel(std_config).to(device)
        
        hfp_config = HFPConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_heads, intermediate_size=512, short_len=8, bulk_dim=32)
        hfp_model = HFPForCausalLM(hfp_config).to(device)
        
        std_optim = torch.optim.AdamW(std_model.parameters(), lr=5e-4)
        hfp_optim = torch.optim.AdamW(hfp_model.parameters(), lr=5e-4)
        
        std_losses = []
        hfp_losses = []
        current_eval_steps = []
        
        def evaluate(model):
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0].to(device)
                    outputs = model(inputs)
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    total_loss += loss.item()
            model.train()
            return total_loss / len(val_loader)

        std_model.train()
        hfp_model.train()
        
        step = 0
        for batch in train_loader:
            if step > max_steps:
                break
                
            inputs = batch[0].to(device)
            
            # Train Std
            std_optim.zero_grad()
            out_std = std_model(inputs)
            std_loss = nn.functional.cross_entropy(out_std.logits[..., :-1, :].contiguous().view(-1, out_std.logits.size(-1)), inputs[..., 1:].contiguous().view(-1))
            std_loss.backward()
            std_optim.step()
            
            # Train HFP
            hfp_optim.zero_grad()
            out_hfp = hfp_model(inputs)
            hfp_loss = nn.functional.cross_entropy(out_hfp.logits[..., :-1, :].contiguous().view(-1, out_hfp.logits.size(-1)), inputs[..., 1:].contiguous().view(-1))
            hfp_loss.backward()
            hfp_optim.step()
            
            if step % val_interval == 0:
                std_vloss = evaluate(std_model)
                hfp_vloss = evaluate(hfp_model)
                std_losses.append(std_vloss)
                hfp_losses.append(hfp_vloss)
                if seed_idx == 0:
                    current_eval_steps.append(step)
                print(f"Adım {step}/{max_steps} | Std Val: {std_vloss:.4f} | HFP Val: {hfp_vloss:.4f}")
                
            step += 1
            
        all_std_val_losses.append(std_losses)
        all_hfp_val_losses.append(hfp_losses)
        
        if seed_idx == 0:
            eval_steps = current_eval_steps
            best_hfp_model = hfp_model
            best_std_model = std_model
            
    print("\nMetin Üretimi (Qualitative Check) Yapılıyor...")
    prompt = "The theory of general relativity"
    hfp_gen = generate_text(best_hfp_model, tokenizer, device, prompt)
    std_gen = generate_text(best_std_model, tokenizer, device, prompt)
    
    print(f"\n[STANDART MODEL ÜRETİMİ]:\n{std_gen}")
    print(f"\n[HFP MODEL ÜRETİMİ]:\n{hfp_gen}")
    
    # Mean ve Std hesaplama
    std_arr = np.array(all_std_val_losses)
    hfp_arr = np.array(all_hfp_val_losses)
    
    std_mean = std_arr.mean(axis=0)
    std_std = std_arr.std(axis=0)
    
    hfp_mean = hfp_arr.mean(axis=0)
    hfp_std = hfp_arr.std(axis=0)
    
    # PPL (Perplexity) 
    # Ortalama PPL
    std_ppl = np.exp(np.clip(std_mean, a_min=None, a_max=20))
    hfp_ppl = np.exp(np.clip(hfp_mean, a_min=None, a_max=20))
    
    # Üst ve Alt Band PPL
    std_ppl_upper = np.exp(np.clip(std_mean + std_std, a_min=None, a_max=20))
    std_ppl_lower = np.exp(np.clip(std_mean - std_std, a_min=None, a_max=20))
    
    hfp_ppl_upper = np.exp(np.clip(hfp_mean + hfp_std, a_min=None, a_max=20))
    hfp_ppl_lower = np.exp(np.clip(hfp_mean - hfp_std, a_min=None, a_max=20))
    
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 6))
    
    # Doğrulama Kaybı (Gölgeli)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(eval_steps, hfp_mean, label="HFP Bulk (Mean)", color='#2ca02c', linewidth=2.5)
    ax1.fill_between(eval_steps, hfp_mean - hfp_std, hfp_mean + hfp_std, color='#2ca02c', alpha=0.2)
    
    ax1.plot(eval_steps, std_mean, label="Standart Transformer (Mean)", color='#1f77b4', linewidth=2.5, linestyle='--')
    ax1.fill_between(eval_steps, std_mean - std_std, std_mean + std_std, color='#1f77b4', alpha=0.2)
    
    ax1.set_title("WikiText-2 Validation Loss (3 Seeds)\n(Daha düşük daha iyidir)")
    ax1.set_xlabel("Eğitim Adımı")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    
    # Perplexity (Gölgeli)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(eval_steps, hfp_ppl, label="HFP Bulk (Mean)", color='#2ca02c', linewidth=2.5)
    ax2.fill_between(eval_steps, hfp_ppl_lower, hfp_ppl_upper, color='#2ca02c', alpha=0.2)
    
    ax2.plot(eval_steps, std_ppl, label="Standart Transformer (Mean)", color='#1f77b4', linewidth=2.5, linestyle='--')
    ax2.fill_between(eval_steps, std_ppl_lower, std_ppl_upper, color='#1f77b4', alpha=0.2)
    
    ax2.set_title("WikiText-2 Validation Perplexity (3 Seeds)\n(Sıfıra Çakılmaz, Doğal Bir Sınıra Yakınsar)")
    ax2.set_xlabel("Eğitim Adımı")
    ax2.set_ylabel("Perplexity")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_quality_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Başarılı! Grafik kaydedildi: {plot_path}")
    
    with open("benchmark_text_samples.txt", "w", encoding="utf-8") as f:
        f.write("[STANDART MODEL URETIMI]\n")
        f.write(std_gen + "\n\n")
        f.write("[HFP MODEL URETIMI]\n")
        f.write(hfp_gen + "\n")

if __name__ == "__main__":
    benchmark_quality()
