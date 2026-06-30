import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

from benchmark import StandardTransformerInference, BulkTransformerInference

# ==========================================
# 1. Konfigürasyon
# ==========================================
VOCAB_SIZE = 1000
HIDDEN_SIZE = 128
NUM_HEADS = 4
FFN_DIM = 256
BATCH_SIZE = 16
SEQ_LEN = 256
STEPS_PER_EPOCH = 1500

# Passkey tokenleri
NOISE_START = 10
PASSKEY_START_TOKEN = 1
PASSKEY_END_TOKEN = 2
QUESTION_TOKEN = 3

# ==========================================
# 2. Sentetik Veri Üretici (Samanlıkta İğne)
# ==========================================
def generate_passkey_batch(batch_size, seq_len, depth_pct=None):
    """
    Rastgele sayılardan oluşan bir dizi (samanlık) üretir ve içine bir şifre (iğne) gizler.
    Dizi formatı: [gürültü..., 1, ŞİFRE, 2, gürültü..., 3]
    Hedef: 3 (Soru tokeni) geldikten sonra ŞİFRE'yi tahmin etmek.
    """
    batch = torch.randint(NOISE_START, VOCAB_SIZE, (batch_size, seq_len))
    targets = torch.zeros(batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        # Eğer derinlik verilmemişse rastgele seç (Eğitim için)
        depth = depth_pct if depth_pct is not None else random.uniform(0.1, 0.9)
        
        passkey_val = random.randint(NOISE_START, VOCAB_SIZE - 1)
        targets[i] = passkey_val
        
        # Soru tokeni her zaman dizinin en sonundadır.
        # Şifrenin gizleneceği maksimum indeks:
        max_idx = seq_len - 5
        insert_idx = int(depth * max_idx)
        
        # İğneyi samanlığa yerleştir
        batch[i, insert_idx] = PASSKEY_START_TOKEN
        batch[i, insert_idx+1] = passkey_val
        batch[i, insert_idx+2] = PASSKEY_END_TOKEN
        
        # Soruyu en sona koy
        batch[i, -1] = QUESTION_TOKEN
        
    return batch, targets

# ==========================================
# 3. Positional Encoding Wrapper (Standart Model İçin)
# ==========================================
# Standart Transformer sıralamayı bilmez, bu yüzden ona konum bilgisi vermeliyiz.
# HFP Bulk Model ise yapısal olarak sırasaldır, kendi içinde zaman algısı vardır.
class PositionalStandardTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, feedforward_dim, max_seq_len=2048):
        super().__init__()
        self.base_model = StandardTransformerInference(vocab_size, hidden_size, num_heads, feedforward_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        
    def forward(self, x, kv_cache=None, step=0):
        emb = self.base_model.embedding(x)
        pos = torch.tensor([[step]] * x.size(0), device=x.device)
        emb = emb + self.pos_emb(pos)
        
        attn_out, new_cache = self.base_model.attn(emb, kv_cache)
        x_norm = self.base_model.norm1(emb + attn_out)
        ffn_out = self.base_model.ffn(x_norm)
        x_out = self.base_model.norm2(x_norm + ffn_out)
        logits = self.base_model.lm_head(x_out)
        return logits, new_cache

# ==========================================
# 4. Eğitim Döngüsü
# ==========================================
def train_model(model, device, name="Model", is_standard=False):
    print(f"\n--- {name} Modeli Eğitiliyor (Görev: Passkey Retrieval) ---")
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(STEPS_PER_EPOCH))
    for step in pbar:
        optimizer.zero_grad()
        x, targets = generate_passkey_batch(BATCH_SIZE, SEQ_LEN)
        x = x.to(device)
        targets = targets.to(device)
        
        state = None
        # O(N) Sequential Eğitim
        if not is_standard and hasattr(model, 'bulk_state'):
            model.bulk_state.reset_state()
            
        for i in range(SEQ_LEN):
            token_input = x[:, i:i+1]
            if is_standard:
                logits, state = model(token_input, state, step=i)
            else:
                logits, state = model(token_input, state)
            
        # Sadece son adımdaki çıktı (Soru tokenine verilen cevap) ile loss hesaplıyoruz
        final_logits = logits.squeeze(1) # (batch, vocab)
        
        loss = criterion(final_logits, targets)
        loss.backward()
        
        # Gradyan patlamasını engellemek için clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            preds = final_logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item() * 100
            pbar.set_description(f"Loss: {loss.item():.4f} | Batch Başarısı: %{acc:.1f}")

# ==========================================
# 5. Değerlendirme Döngüsü
# ==========================================
def evaluate_model(model, device, depths, is_standard=False):
    print(f"Test Ediliyor...")
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for depth in depths:
            correct = 0
            total = 128 # Her derinlik için 128 örnek
            
            for _ in range(total // BATCH_SIZE):
                x, targets = generate_passkey_batch(BATCH_SIZE, SEQ_LEN, depth_pct=depth)
                x = x.to(device)
                targets = targets.to(device)
                
                state = None
                # Test döngüsünde HFPBulkState'i sıfırlamak önemlidir
                if not is_standard and hasattr(model, 'bulk_state'):
                    model.bulk_state.reset_state()
                    
                for i in range(SEQ_LEN):
                    token_input = x[:, i:i+1]
                    if is_standard:
                        logits, state = model(token_input, state, step=i)
                    else:
                        logits, state = model(token_input, state)
                
                final_logits = logits.squeeze(1)
                preds = final_logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
            
            acc = correct / ((total // BATCH_SIZE) * BATCH_SIZE) * 100.0
            accuracies.append(min(acc, 100.0))
            print(f" Derinlik: {depth:.2f} (Dizinin %{int(depth*100)}'i) | Başarı: {acc:.1f}%")
            
    return accuracies

# ==========================================
# 6. Ana Çalıştırma Bloğu
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")
    
    # 1. HFP Bulk Model
    bulk_model = BulkTransformerInference(VOCAB_SIZE, HIDDEN_SIZE, NUM_HEADS, FFN_DIM)
    train_model(bulk_model, device, "HFP Bulk Model")
    
    # 2. Standart Transformer (Adil olması için Positional Encoding ekliyoruz)
    std_model = PositionalStandardTransformer(VOCAB_SIZE, HIDDEN_SIZE, NUM_HEADS, FFN_DIM)
    train_model(std_model, device, "Standart Transformer (KV-Cache)", is_standard=True)
    
    # Derinlikler (0.1 = Şifre en başta, 0.9 = Şifre en sonda)
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    print("\n--- HFP Bulk Model Değerlendirmesi ---")
    bulk_accs = evaluate_model(bulk_model, device, depths)
    
    print("\n--- Standart Transformer Değerlendirmesi ---")
    std_accs = evaluate_model(std_model, device, depths, is_standard=True)
    
    # ==========================================
    # 7. Grafiği Çizdirme
    # ==========================================
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    plt.plot([d * 100 for d in depths], bulk_accs, marker='o', markersize=8, linewidth=3, label='HFP Bulk Model (O(1) Sabit Hafıza)', color='#2ca02c')
    plt.plot([d * 100 for d in depths], std_accs, marker='s', markersize=8, linewidth=3, label='Standart Transformer (KV-Cache)', color='#1f77b4')
    
    plt.title(f"Samanlıkta İğne Arama (Passkey Retrieval)\nBağlam Uzunluğu: {SEQ_LEN} Token")
    plt.xlabel("Şifrenin Gizlendiği Derinlik (% Konum)")
    plt.ylabel("Hatırlama Başarısı (Accuracy %)")
    plt.ylim(-5, 105)
    plt.xlim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    
    # "İdeal Model" referans çizgisi
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(os.path.dirname(__file__), "passkey_results.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\nSonuçlar başarıyla '{plot_path}' konumuna kaydedildi.")
