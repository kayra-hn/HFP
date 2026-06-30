import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoTokenizer
from datasets import load_dataset
from hfp import HFPForCausalLM, HFPConfig
from hfp.physics.physics_optimizers import QuantizedLR, UncertaintyRegularizer, StiffTransientScheduler

def get_dataloader(batch_size=8, seq_len=128):
    print("Veri Seti: Salesforce/wikitext-2-raw-v1 indiriliyor...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")
    
    train_loader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
    return train_loader, vocab_size, tokenizer.pad_token_id

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim Cihazı: {device}")
    
    batch_size = 2  # VRAM çökmesini engellemek için 8'den 2'ye düşürüldü
    grad_accum_steps = 4  # Etkili Batch Size'ı tekrar 8 yapmak için gradyan biriktirme
    seq_len = 128
    epochs = 10
    save_path = "hfp_weights.pt"
    
    train_loader, vocab_size, pad_token_id = get_dataloader(batch_size, seq_len)
    
    # Gerçek boyutlu GPT-2 Small dengi HFP Config (Yaklaşık 124M Parametre)
    config = HFPConfig(
        vocab_size=vocab_size,
        hidden_size=768,          # Gerçek boyut
        num_hidden_layers=12,     # Gerçek derinlik
        num_attention_heads=12,
        intermediate_size=3072,
        bulk_dim=128,
        short_len=16
    )
    
    model = HFPForCausalLM(config).to(device)
    
    # Daha önce kayıtlı bir ağırlık varsa onu yükle (Kaldığı yerden devam etme)
    if os.path.exists(save_path):
        print(f"Önceki kayıt bulundu: {save_path}. Ağırlıklar yükleniyor...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Beyin yüklendi. Kaldığı yerden öğrenmeye devam edecek!")
    else:
        print("Kayıt bulunamadı. Model sıfırdan eğitime başlıyor.")

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    # HFP Fizik Optimizasyonları ve Stiff Zamanlayıcı
    scheduler = StiffTransientScheduler(optimizer, plateau_threshold=5.0, stiffness_p=0.5)
    reg = UncertaintyRegularizer(model, h_bar=0.001)

    print("\n🚀 EĞİTİM BAŞLIYOR...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input_ids"].to(device)
            
            # Forward pass (Padding kısımlarını Loss hesabından çıkar)
            labels = inputs.clone()
            labels[labels == pad_token_id] = -100
            outputs = model(inputs, labels=labels)
            loss = outputs.loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient Accumulation: Belirli bir adıma ulaşınca ağırlıkları güncelle
            if (batch_idx + 1) % grad_accum_steps == 0:
                # [CRITICAL BUG FIX]: Gradient clipping belongs here, not in forward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                reg.step()
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += (loss.item() * grad_accum_steps)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Adım {batch_idx}/{len(train_loader)} | Anlık Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"\n✅ Epoch {epoch} Tamamlandı. Ortalama Loss: {avg_loss:.4f}")
        
        # Her Epoch sonunda beyni kaydet!
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model beyni (Ağırlıklar) kaydedildi: {save_path}\n")

if __name__ == "__main__":
    train()
