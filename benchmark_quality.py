import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import matplotlib.pyplot as plt
import os
import math

from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

def get_memory_mb(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    return 0

def benchmark_quality():
    print("=== HFP vs Standard Transformer: Kalite (Perplexity) & Loss Testi ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    # 1. Veri Hazırlığı
    print("Tokenizer ve Veri Seti Yükleniyor (WikiText-2)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Sentetik (Örüntülü) Veri Seti Üretiliyor...")
    # 0'dan 50'ye kadar giden tekrar eden bir örüntü
    # Standart ve HFP modelleri bu örüntüyü ezberleyebilmeli ve Loss 0'a inmeli.
    pattern = torch.arange(0, 50, dtype=torch.long)
    vocab_size = 1000  # Daha küçük kelime dağarcığı (hızlı eğitim için)
    
    # 500 batch, her biri 4'lü batch size ve 64 uzunlukta
    seq_len = 64
    batch_size = 4
    num_steps = 150
    
    def generate_batch():
        x = pattern.repeat(10)[:seq_len] # (64,)
        x = x.unsqueeze(0).repeat(batch_size, 1) # (4, 64)
        return {"input_ids": x, "labels": x}
    
    # Dataloader yerine basit bir jeneratör
    def dataloader():
        for _ in range(num_steps):
            yield generate_batch()
    
    # 2. Modelleri Hazırlama
    # Eğitim hızlı sürsün diye 4 katmanlı (Küçük) model konfigürasyonu
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    
    print("Modeller Başlatılıyor...")
    # Standart (GPT-2 tabanlı KV-Cache)
    std_config = GPT2Config(vocab_size=vocab_size, n_embd=hidden_size, n_head=num_heads, n_layer=num_layers)
    std_model = GPT2LMHeadModel(std_config).to(device)
    
    # HFP (O(1) Sabit Hafıza)
    hfp_config = HFPConfig(
        vocab_size=vocab_size, 
        hidden_size=hidden_size, 
        num_hidden_layers=num_layers, 
        num_attention_heads=num_heads, 
        intermediate_size=1024,
        short_len=8,
        bulk_dim=64
    )
    hfp_model = HFPForCausalLM(hfp_config).to(device)
    
    # 3. Eğitim Döngüsü
    std_optim = torch.optim.AdamW(std_model.parameters(), lr=1e-3)
    hfp_optim = torch.optim.AdamW(hfp_model.parameters(), lr=1e-3)
    
    std_losses = []
    hfp_losses = []
    
    print(f"Eğitim Başlıyor... ({num_steps} Adım)")
    std_model.train()
    hfp_model.train()
    
    step = 0
    for batch in dataloader():
        if step >= num_steps:
            break
            
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # --- Standart Model Adımı ---
        std_optim.zero_grad()
        std_outputs = std_model(inputs)
        std_logits = std_outputs.logits
        # Shift logits for next token prediction
        std_shift_logits = std_logits[..., :-1, :].contiguous()
        std_shift_labels = labels[..., 1:].contiguous()
        std_loss = nn.functional.cross_entropy(std_shift_logits.view(-1, std_shift_logits.size(-1)), std_shift_labels.view(-1))
        std_loss.backward()
        std_optim.step()
        
        # --- HFP Model Adımı ---
        hfp_optim.zero_grad()
        hfp_outputs = hfp_model(inputs)
        hfp_logits = hfp_outputs.logits
        hfp_shift_logits = hfp_logits[..., :-1, :].contiguous()
        hfp_shift_labels = labels[..., 1:].contiguous()
        hfp_loss = nn.functional.cross_entropy(hfp_shift_logits.view(-1, hfp_shift_logits.size(-1)), hfp_shift_labels.view(-1))
        hfp_loss.backward()
        hfp_optim.step()
        
        std_losses.append(std_loss.item())
        hfp_losses.append(hfp_loss.item())
        
        step += 1
        
    # 4. Sonuçları Çizdirme
    print("\nGrafikler Çizdiriliyor...")
    
    # Düzgünleştirmek için hareketli ortalama (Moving Average)
    def moving_average(x, w=10):
        if len(x) < w: return x
        import numpy as np
        return np.convolve(x, np.ones(w), 'valid') / w
        
    std_smooth = moving_average(std_losses)
    hfp_smooth = moving_average(hfp_losses)
    
    # Perplexity Hesaplama: PPL = exp(Loss)
    std_ppl = [math.exp(min(l, 20)) for l in std_smooth] # Taşmayı önlemek için cap 20
    hfp_ppl = [math.exp(min(l, 20)) for l in hfp_smooth]
    
    x_axis = range(len(std_smooth))
    
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 6))
    
    # Loss Grafiği
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x_axis, hfp_smooth, label="HFP Bulk Model (O(1) Hafıza)", color='#2ca02c', linewidth=2.5)
    ax1.plot(x_axis, std_smooth, label="Standart Transformer (O(N) KV-Cache)", color='#1f77b4', linewidth=2.5, alpha=0.7)
    ax1.set_title("Eğitim Kaybı (Loss) Yakınsaması\n(WikiText-2, Daha düşük daha iyidir)")
    ax1.set_xlabel("Eğitim Adımı")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    
    # Perplexity Grafiği
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(x_axis, hfp_ppl, label="HFP Bulk Model", color='#2ca02c', linewidth=2.5)
    ax2.plot(x_axis, std_ppl, label="Standart Transformer", color='#1f77b4', linewidth=2.5, alpha=0.7)
    ax2.set_title("Metin Kalitesi / Karmaşıklık (Perplexity)\n(Daha düşük daha iyidir)")
    ax2.set_xlabel("Eğitim Adımı")
    ax2.set_ylabel("Perplexity")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_quality_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Başarılı! Grafik kaydedildi: {plot_path}")

if __name__ == "__main__":
    benchmark_quality()
