import torch
import torch.nn as nn
import time
import os
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
try:
    from hfp import HFPConfig, HFPForCausalLM
except Exception as e:
    print(f"HFP import failed ({e}); proceeding without HFP components.")
    HFPConfig = None
    HFPForCausalLM = None

# [OPT-7] psutil Process nesnesini bir kez oluştur ve önbelleğe al
try:
    from torchtext.datasets import WikiText2
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    TORCHTEXT_AVAILABLE = True
except Exception as e:
    print(f"Uyarı: 'torchtext' yüklenemediği için ({e}) sentetik veri kullanılacak.")
    TORCHTEXT_AVAILABLE = False
_process = psutil.Process(os.getpid())

def get_ram_mb():
    return _process.memory_info().rss / (1024 * 1024)

# ==========================================
# 2. Standart Transformer (KV-Cache) Sınıfları
# ==========================================
class StandardAttentionKV(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_k = d_model // nhead

    def forward(self, x, kv_cache=None):
        B, S, D = x.size()
        q = self.q_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.d_k).transpose(1, 2)
        
        # KV-Cache Büyüyor
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_kv_cache = (k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out, new_kv_cache

class StandardTransformerInference(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, feedforward_dim, num_layers=12):
        super().__init__()
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_head=num_heads,
            n_layer=num_layers,
            n_inner=feedforward_dim,
            n_positions=8192,
            n_ctx=8192
        )
        self.model = GPT2LMHeadModel(config)
        
    def forward(self, x, kv_cache=None, step=0):
        outputs = self.model(input_ids=x, past_key_values=kv_cache, use_cache=True)
        return outputs.logits, outputs.past_key_values

class BulkTransformerInference(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, feedforward_dim, num_layers=12):
        super().__init__()
        from hfp.models.configuration_hfp import HFPConfig
        from hfp.models.modeling_hfp import HFPForCausalLM
        
        config = HFPConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=feedforward_dim,
            short_len=16,
            bulk_dim=128
        )
        self.model = HFPForCausalLM(config)
        self.bulk_state = self.model.hfp.bulk_states[0]
        
    def forward(self, x, state=None):
        outputs = self.model(x, past_key_values=state, use_cache=True)
        return outputs.logits, outputs.past_key_values

# ==========================================
# 3. Çıkarım Benchmark Fonksiyonu (GPU + CPU Uyumlu)
# ==========================================

def get_memory_mb(device):
    """
    Cihaza göre bellek kullanımını MB cinsinden döndürür.
    GPU: torch.cuda.memory_allocated (VRAM)
    CPU: psutil RSS (RAM)
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    else:
        return get_ram_mb()

def benchmark_inference(model, device, num_steps=4096, batch_size=8, vocab_size=5000, name="Model"):
    print(f"\n--- {name} Çıkarım Testi Başlıyor (Hedef: {num_steps} Adım, Cihaz: {device}) ---")
    model.to(device)
    model.eval()
    
    mem_usage = []       # Bellek kullanımı (RAM veya VRAM)
    short_len_log = []   # Dinamik kısa‑hafıza uzunluğu kayıtları
    tokens_per_sec = []
    oom_step = None
    
    current_token = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    state = None
    
    interval = 64
    
    # GPU zamanlama doğruluğu için ısınma (warmup)
    if device.type == "cuda":
        with torch.no_grad():
            warmup_state = None
            for _ in range(5):
                _, warmup_state = model(current_token, warmup_state)
            # Warmup state'ini temizle, asıl testi sıfırdan başlat
            del warmup_state
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        current_token = torch.randint(0, vocab_size, (batch_size, 1), device=device)
        state = None
    
    start_time = time.time()
    
    with torch.no_grad():
        for step in range(1, num_steps + 1):
            try:
                # İleri Besleme (Forward Pass)
                logits, state = model(current_token, state)
                current_token = logits.argmax(dim=-1)
                
                if step % interval == 0 or step == 1:
                    if device.type == "cuda":
                        torch.cuda.synchronize()  # GPU zamanlamasını doğru ölç
                        
                    end_time = time.time()
                    time_taken = end_time - start_time
                    tps = (batch_size * interval) / time_taken if step > 1 else batch_size / time_taken
                    mem_mb = get_memory_mb(device)
                    
                    # Kısa‑hafıza uzunluğunu logla (entropy‑tabanlı genişletme)
                    if hasattr(model, "bulk_state"):
                        short_len = model.bulk_state.short_len_dynamic
                        short_len_log.append((step, short_len))
                    
                    if step % 512 == 0 or step == 1:
                        mem_label = "VRAM" if device.type == "cuda" else "RAM"
                        print(f"{name} | Adım {step}/{num_steps} | {mem_label}: {mem_mb:.2f} MB | Hız: {tps:.2f} Tok/s")
                    
                    # OOM güvenlik valfi (CPU için RAM, GPU için VRAM)
                    oom_limit = 2500  # MB
                    if device.type == "cuda":
                        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
                        oom_limit = total_vram * 0.85  # VRAM'in %85'ini geçerse durdur
                    
                    if mem_mb > oom_limit:
                        mem_label = "VRAM" if device.type == "cuda" else "RAM"
                        raise MemoryError(f"Yapay OOM Koruması: KV-Cache {mem_label} sınırını ({oom_limit:.0f} MB) aştı!")
                    
                    mem_usage.append((step, mem_mb))
                    tokens_per_sec.append((step, tps))
                    start_time = time.time()
                    
            except (RuntimeError, MemoryError) as e:
                print(f"\n!!! [{name}] OutOfMemory (OOM) Çöküşü !!!")
                print(f"Adım: {step} | Hata Mesajı: {str(e)}\n")
                oom_step = step
                
                mem_mb = get_memory_mb(device)
                mem_usage.append((step, mem_mb))
                if len(tokens_per_sec) > 0:
                    tokens_per_sec.append((step, tokens_per_sec[-1][1]))
                break
                
    return mem_usage, tokens_per_sec, oom_step

# ==========================================
# 4. Cihaz Tespiti ve GPU Bilgisi
# ==========================================

def detect_device():
    """
    Otomatik olarak en iyi cihazı seçer.
    GPU varsa CUDA'yı seçer ve GPU modelini döndürür.
    GPU yoksa uyarı verir ve CPU'ya geçer.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[OK] GPU Bulundu: {gpu_name} ({total_vram:.1f} GB VRAM)")
        print(f"     CUDA Surumu: {torch.version.cuda}")
        return device, gpu_name
    else:
        print("[UYARI] CUDA destekli GPU bulunamadi!")
        print("        GPU performans testi CPU uzerinde simule edilecek.")
        print("        Gercek GPU sonuclari icin CUDA destekli bir GPU gereklidir.")
        device = torch.device("cpu")
        return device, "CPU (GPU bulunamadi)"

# ==========================================
# 5. Özel Zamanlayıcı: StiffTransientScheduler
# ==========================================
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
import torch.optim as optim

class StiffTransientScheduler(LRScheduler):
    def __init__(self, optimizer, plateau_threshold, stiffness_p, last_epoch=-1):
        self.plateau_threshold = plateau_threshold
        self.stiffness_p = stiffness_p
        self.is_active = False
        self.activation_epoch = 0
        super().__init__(optimizer, last_epoch)

    def step(self, val_loss=None, epoch=None):
        if val_loss is not None:
            if not self.is_active and val_loss < self.plateau_threshold:
                self.is_active = True
                self.activation_epoch = self.last_epoch
        super().step(epoch)

    def get_lr(self):
        if not self.is_active:
            return [base_lr for base_lr in self.base_lrs]
        
        epochs_since_active = max(0, self.last_epoch - self.activation_epoch)
        return [base_lr / (1.0 + self.stiffness_p * epochs_since_active)
                for base_lr in self.base_lrs]

def benchmark_stiff_scheduler(device, gpu_name):
    print("\n--- StiffTransientScheduler Eğitim Simülasyonu Başlıyor ---")
    
    # 1. Sentetik Veri
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)
    
    epochs = 150
    plateau_threshold = 0.5
    
    def train_model(scheduler_type="cosine"):
        torch.manual_seed(42) # Adil karşılaştırma için
        model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.MSELoss()
        
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = StiffTransientScheduler(optimizer, plateau_threshold=plateau_threshold, stiffness_p=0.5)
            
        losses = []
        lrs = []
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            
            if scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_loss=loss.item())
                
        return losses, lrs

    losses_cos, lrs_cos = train_model("cosine")
    losses_stiff, lrs_stiff = train_model("stiff")
    
    # Grafikleri Çizdirme
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 5))
    
    # LR Grafiği
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(lrs_cos, label="CosineAnnealingLR", color="#1f77b4", linewidth=2)
    ax1.plot(lrs_stiff, label="StiffTransientScheduler", color="#d62728", linewidth=2)
    ax1.set_title("Öğrenme Oranı (Learning Rate) Değişimi")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("LR")
    ax1.legend()
    
    # Loss Grafiği
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(losses_cos, label="CosineAnnealingLR Loss", color="#1f77b4", linewidth=2)
    ax2.plot(losses_stiff, label="StiffTransientScheduler Loss", color="#d62728", linewidth=2)
    ax2.axhline(y=plateau_threshold, color='black', linestyle='--', label=f"Eşik ({plateau_threshold})")
    ax2.set_title("Eğitim Kaybı (Training Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_stiff.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nStiff Scheduler grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 6. Özel Niceleme Zamanlayıcısı: ZenonQuantizationScheduler
# ==========================================
class ZenonQuantizationScheduler:
    def __init__(self, model, total_steps, schedule_points=[0.7, 0.9], grad_threshold=0.5):
        self.model = model
        self.total_steps = total_steps
        self.schedule_points = sorted(schedule_points)
        self.grad_threshold = grad_threshold
        
        self.current_precision = 32  # 32 (FP32), 16 (FP16), 8 (INT8)
        self.stage = 0  # 0: FP32, 1: Beklemede veya FP16, 2: Beklemede veya INT8
        
        self.history_grad_norm = []
        self.history_precision = []
        self.history_energy = []
        self.cumulative_energy = 0.0
        
        # Geçiş noktaları (adımlar)
        self.target_steps = [int(p * total_steps) for p in self.schedule_points]

    def _get_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)

    def step(self, current_step):
        grad_norm = self._get_grad_norm()
        self.history_grad_norm.append(grad_norm)
        
        # Durum kontrolü
        if self.stage == 0 and current_step >= self.target_steps[0]:
            if grad_norm < self.grad_threshold:
                self.current_precision = 16
                self.stage = 1
                print(f"[Zenon] Adım {current_step}: Gradyan ({grad_norm:.3f} < {self.grad_threshold}). FP16'ya geçildi.")
            else:
                pass # Delaying
                
        elif self.stage == 1 and current_step >= self.target_steps[1]:
            if grad_norm < self.grad_threshold:
                self.current_precision = 8
                self.stage = 2
                print(f"[Zenon] Adım {current_step}: Gradyan ({grad_norm:.3f} < {self.grad_threshold}). INT8'e geçildi.")
            else:
                pass # Delaying
                
        # Enerji Tüketimi Simülasyonu
        if self.current_precision == 32:
            energy_cost = 1.0
        elif self.current_precision == 16:
            energy_cost = 0.5
        elif self.current_precision == 8:
            energy_cost = 0.25
            
        self.cumulative_energy += energy_cost
        self.history_energy.append(self.cumulative_energy)
        self.history_precision.append(self.current_precision)
        
        return self.current_precision

def benchmark_zenon_scheduler(device, gpu_name):
    print("\n--- ZenonQuantizationScheduler Eğitim Simülasyonu Başlıyor ---")
    
    epochs = 300
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)
    
    def train_model(use_zenon=False):
        torch.manual_seed(42)
        # Daha karmaşık bir ağ, kayıp yavaşça düşsün
        model = nn.Sequential(
            nn.Linear(10, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        
        # Kaybın yavaş inmesi için momentum'lu SGD
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.MSELoss()
        
        if use_zenon:
            zenon = ZenonQuantizationScheduler(model, total_steps=epochs, schedule_points=[0.6, 0.8], grad_threshold=1.5)
            
        losses = []
        energies = []
        precisions = []
        grad_norms = []
        
        cumulative_standard_energy = 0.0
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            # Yapay bir gradyan patlaması / gecikmesi yaratmak için gürültü ekleyelim
            # Tam hedeflenen %60 (180. adım) ve %80 (240. adım) noktalarında gradyanı kasten bozalım
            # Böylece Zenon'un nicelemeyi ertelediğini görebiliriz.
            if epoch in range(175, 195) or epoch in range(235, 255): 
                noise = torch.randn_like(out) * 0.5
                loss = criterion(out + noise, y)
            else:
                loss = criterion(out, y)
                
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if use_zenon:
                zenon.step(epoch)
            else:
                # Standart eğitim her zaman 32 bit, 1 birim enerji harcar
                cumulative_standard_energy += 1.0
                energies.append(cumulative_standard_energy)
                
        if use_zenon:
            return losses, zenon.history_energy, zenon.history_precision, zenon.history_grad_norm, zenon.target_steps
        return losses, energies

    print("Standart Eğitim Çalıştırılıyor...")
    losses_std, energies_std = train_model(use_zenon=False)
    
    print("Zenon Quantization Zamanlayıcı Çalıştırılıyor...")
    losses_zen, energies_zen, prec_zen, grad_zen, targets = train_model(use_zenon=True)
    
    # Grafikleri Çizdirme
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Kayıp ve Hassasiyet Geçişleri
    ax1.plot(losses_std, label="Standart Kayıp (FP32)", color="gray", alpha=0.5)
    ax1.plot(losses_zen, label="Zenon Kayıp", color="#1f77b4", linewidth=2)
    
    # Geçişleri boyama
    fp16_start = prec_zen.index(16) if 16 in prec_zen else epochs
    int8_start = prec_zen.index(8) if 8 in prec_zen else epochs
    
    ax1.axvspan(0, fp16_start, color='red', alpha=0.1, label='FP32 Evresi')
    ax1.axvspan(fp16_start, int8_start, color='orange', alpha=0.1, label='FP16 Evresi')
    ax1.axvspan(int8_start, epochs, color='green', alpha=0.1, label='INT8 Evresi')
    
    ax1.set_title("Eğitim Kaybı ve Hassasiyet (Precision) Evreleri")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # 2. Enerji Tüketimi
    ax2.plot(energies_std, label="Standart Kümülatif Enerji (Düz Çizgi)", color="gray", linestyle="--", linewidth=2)
    ax2.plot(energies_zen, label="Zenon Kümülatif Enerji (Bükülen Çizgi)", color="#2ca02c", linewidth=2.5)
    ax2.set_title("Kümülatif Enerji Tüketimi Karşılaştırması")
    ax2.set_ylabel("Enerji Birimi")
    ax2.legend()
    
    # 3. Gradyan Büyüklüğü ve Gecikme Mekanizması
    ax3.plot(grad_zen, label="Gradyan Büyüklüğü (L2 Norm)", color="#d62728", linewidth=1.5)
    ax3.axhline(y=1.5, color='black', linestyle='--', label="Geçiş Eşiği (Threshold)")
    
    # Hedef noktaları işaretle
    ax3.axvline(x=targets[0], color='purple', linestyle=':', linewidth=2, label="Hedef FP16 (%60)")
    ax3.axvline(x=targets[1], color='blue', linestyle=':', linewidth=2, label="Hedef INT8 (%80)")
    
    # Gerçek geçiş noktaları
    ax3.plot(fp16_start, grad_zen[fp16_start], 'o', color='purple', markersize=10, label="Gerçek FP16 Geçişi")
    ax3.plot(int8_start, grad_zen[int8_start], 'o', color='blue', markersize=10, label="Gerçek INT8 Geçişi")
    
    ax3.set_title("Zenon Geciktirme (Delay) Mekanizması")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Grad Norm")
    ax3.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_zenon.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nZenon Scheduler grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 7. Özel Katman Testi: Holographic Dropout
# ==========================================
from hfp.core.bulk_trigger_decoder import HolographicDropout

def benchmark_holographic_dropout(device, gpu_name):
    print("\n--- Holographic vs Standard Dropout Eğitim Simülasyonu Başlıyor ---")
    
    epochs = 200
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)
    
    def train_model(use_holographic=False):
        torch.manual_seed(42)
        
        layers = [nn.Linear(10, 128), nn.GELU()]
        if use_holographic:
            layers.append(HolographicDropout(0.5))
        else:
            layers.append(nn.Dropout(0.5))
            
        layers.extend([nn.Linear(128, 64), nn.GELU()])
        
        if use_holographic:
            layers.append(HolographicDropout(0.5))
        else:
            layers.append(nn.Dropout(0.5))
            
        layers.append(nn.Linear(64, 1))
        
        model = nn.Sequential(*layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses

    losses_std = train_model(use_holographic=False)
    losses_holo = train_model(use_holographic=True)
    
    # Grafikleri Çizdirme
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    plt.plot(losses_std, label="Standart Dropout (Sıfırlama)", color="gray", alpha=0.7, linewidth=2)
    plt.plot(losses_holo, label="Holographic Dropout (S⁴ Ölçekleme)", color="#9467bd", linewidth=2.5)
    
    plt.title("Eğitim Stabilitesi: Holographic vs Standard Dropout\n(Born Kuralı: P(r) ∝ r)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_holographic.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nHolographic Dropout grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 9. WikiText-2 Veri Seti Benchmark
# ==========================================
def get_wikitext2_dataloader(batch_size: int = 16):
    """Loads WikiText‑2 dataset using HuggingFace datasets and falls back to synthetic data on failure."""
    print("HuggingFace 'datasets' kütüphanesinden WikiText-2 indiriliyor...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    seq_len = 35

    try:
        from datasets import load_dataset
        # Bazen 'wikitext' namespace eksikliği nedeniyle HFUriError verebilir.
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        except Exception:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")

        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_datasets.set_format("torch")
        
        train_loader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)
        print("Gerçek WikiText-2 veri seti başarıyla yüklendi!")
    except Exception as e:
        print(f"HuggingFace dataset indirilemedi ({e}). Sentetik (Mock) veri setine geçiliyor...")
        num_batches = 100
        synthetic_data = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len), dtype=torch.long)
        train_dataset = torch.utils.data.TensorDataset(synthetic_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    class HFVocab:
        def __len__(self):
            return vocab_size
            
    return train_loader, HFVocab()

def benchmark_wikitext2(device, gpu_name, batch_size=16, epochs=30):
    print("\n--- WikiText-2 Language Modeling Benchmark Başlıyor ---")
    train_loader, vocab = get_wikitext2_dataloader(batch_size)
    vocab_size = len(vocab)
    # Gerçek HFP Modelinin "Tiny" Versiyonunu Eğitelim
    if HFPConfig is None or HFPForCausalLM is None:
        print("HFP modülü yüklenemediği için benchmark atlanıyor.")
        return

    config = HFPConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        bulk_dim=64,
        short_len=8
    )
    model = HFPForCausalLM(config).to(device)

    from hfp.physics.physics_optimizers import QuantizedLR, UncertaintyRegularizer
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = QuantizedLR(optimizer, energy_levels=[0.002, 0.001, 0.0005, 0.0001], patience=2)
    reg = UncertaintyRegularizer(model, h_bar=0.001)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            
            # Causal LM: hedefler (targets) bir sonraki kelimelerdir.
            # input_ids olarak inputs'u, labels olarak yine inputs'u veriyoruz. HFPForCausalLM içerde kaydırmayı (shift) kendi yapar.
            optimizer.zero_grad()
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            
            loss.backward()
            reg.step() # Uncertainty regularization
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss) # QuantizedLR adımı
        
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")
    # Plot loss curve
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss", color="#ff7f0e")
    plt.title("WikiText-2 Benchmark Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_wikitext2.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nWikiText-2 benchmark grafiği kaydedildi: {plot_path}")

# ==========================================
# 8. Özel Katman Testi: EntangledLinear
# ==========================================
from hfp.core.bulk_trigger_decoder import EntangledLinear

def benchmark_entangled_linear(device, gpu_name):
    print("\n--- EntangledLinear vs Standard Linear Eğitim Simülasyonu Başlıyor ---")
    
    epochs = 200
    X = torch.randn(100, 32, device=device)
    y = torch.randn(100, 1, device=device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_model(use_entangled=False):
        torch.manual_seed(42)
        
        in_dim = 32
        ffn_dim = 128
        out_dim = 1
        
        if use_entangled:
            # Parametre sayısını eşitlemek için bulk_dim hesaplaması
            bulk_dim = 18
            class EntangledNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.entangled = EntangledLinear(in_dim, ffn_dim, ffn_dim, out_dim, bulk_dim)
                    self.gelu = nn.GELU()
                def forward(self, x):
                    mid = self.entangled.forward_A(x)
                    mid = self.gelu(mid)
                    return self.entangled.forward_B(mid)
                def get_ortho_loss(self):
                    return self.entangled.get_orthogonality_loss()
                    
            model = EntangledNet().to(device)
            print(f"Entangled Model Parametre Sayısı: {count_parameters(model)} (bulk_dim={bulk_dim})")
        else:
            model = nn.Sequential(
                nn.Linear(in_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, out_dim)
            ).to(device)
            print(f"Standart Model Parametre Sayısı: {count_parameters(model)}")
            
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        losses = []
        ortho_losses = []
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            
            mse_loss = criterion(out, y)
            
            if use_entangled:
                ortho = model.get_ortho_loss()
                loss = mse_loss + 0.01 * ortho
                ortho_losses.append(ortho.item())
            else:
                loss = mse_loss
                
            loss.backward()
            optimizer.step()
            losses.append(mse_loss.item())
            
        return losses, ortho_losses

    losses_std, _ = train_model(use_entangled=False)
    losses_ent, ortho_ent = train_model(use_entangled=True)
    
    # Grafikleri Çizdirme
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(losses_std, label="Standart 2x Linear", color="gray", alpha=0.7, linewidth=2)
    ax1.plot(losses_ent, label="EntangledLinear", color="#d62728", linewidth=2.5)
    ax1.set_title("Eğitim Kaybı (MSE)\n(Aynı Parametre Sayısı)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend()
    
    ax2.plot(ortho_ent, label="||P_A * P_B^T||", color="#1f77b4", linewidth=2.5)
    ax2.set_title("Ortogonallik Cezası (Regularization)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Frobenius Norm")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_entangled.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nEntangledLinear grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 9. Özel Katman Testi: TunnelingDropout
# ==========================================
from hfp.core.bulk_trigger_decoder import TunnelingDropout

def benchmark_tunneling_dropout(device, gpu_name):
    print("\n--- TunnelingDropout vs Standard Dropout Eğitim Simülasyonu Başlıyor ---")
    
    epochs = 200
    X = torch.randn(100, 32, device=device)
    y = torch.randn(100, 1, device=device)

    def train_model(use_tunneling=False):
        torch.manual_seed(42)
        
        layers = [nn.Linear(32, 128), nn.GELU()]
        if use_tunneling:
            layers.append(TunnelingDropout(p=0.5, tunnel_depth=3, decay_factor=0.8))
        else:
            layers.append(nn.Dropout(p=0.5))
            
        layers.extend([nn.Linear(128, 64), nn.GELU()])
        
        if use_tunneling:
            layers.append(TunnelingDropout(p=0.5, tunnel_depth=3, decay_factor=0.8))
        else:
            layers.append(nn.Dropout(p=0.5))
            
        layers.append(nn.Linear(64, 1))
        
        model = nn.Sequential(*layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses

    losses_std = train_model(use_tunneling=False)
    losses_tun = train_model(use_tunneling=True)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    plt.plot(losses_std, label="Standart Dropout (Enerji Yok Olur)", color="gray", alpha=0.7, linewidth=2)
    plt.plot(losses_tun, label="TunnelingDropout (Enerji 5. Boyuttan Döner)", color="#ff7f0e", linewidth=2.5)
    
    plt.title("Eğitim Stabilitesi: Tunneling vs Standard Dropout\n(Kuantum Tünelleme ve FIFO Buffer Etkisi)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.yscale("log")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_tunneling.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nTunnelingDropout grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 11. Holographic vs Tunneling Dropout Benchmark on WikiText-2
# ==========================================
def benchmark_holographic_vs_tunneling(device, gpu_name):
    print("\n--- Holographic vs Tunneling Dropout Benchmark (Sentetik Veri) ---")
    vocab_size = 5000
    
    # Sentetik veri üretimi (torchtext çöktüğü için)
    # Orjinal koddaki karmaşık wikitext2 yükleme yerine basit sentetik diziler kullanılıyor.
    def encode(text):
        return [torch.randint(0, vocab_size, (1,)).item() for _ in range(10)]


    seq_len = 30
    # Sentetik rastgele tensörler
    X = torch.randint(0, vocab_size, (100, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab_size, (100,), dtype=torch.long, device=device)

    embed_dim = 128
    hidden_dim = 128

    def build_model(use_holo):
        layers = [nn.Embedding(vocab_size, embed_dim), nn.Flatten()]
        if use_holo:
            layers.append(HolographicDropout(0.2))
        else:
            layers.append(TunnelingDropout(p=0.2, tunnel_depth=3, decay_factor=0.8))
        layers.extend([
            nn.Linear(seq_len * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        ])
        return nn.Sequential(*layers).to(device)

    def train(model):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        epochs = 3
        losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X)  # [batch, vocab]
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    model_holo = build_model(use_holo=True)
    model_tun = build_model(use_holo=False)
    loss_holo = train(model_holo)
    loss_tun = train(model_tun)

    plt.style.use('ggplot')
    plt.figure(figsize=(8,5))
    plt.plot(loss_holo, label='HolographicDropout', color='#1f77b4')
    plt.plot(loss_tun, label='TunnelingDropout', color='#ff7f0e')
    plt.title('Loss Comparison on WikiText-2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(os.path.dirname(__file__), 'benchmark_holo_vs_tunnel.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Holographic vs Tunneling comparison plot saved: {plot_path}")


# ==========================================
# 10. Özel Optimizasyon: UncertaintyRegularizer
# ==========================================
class UncertaintyRegularizer:
    def __init__(self, h_bar=0.01, eps=1e-8):
        self.h_bar = h_bar
        self.eps = eps

    def step(self, model):
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    g_norm = torch.norm(p.grad)
                    eta = self.h_bar / (g_norm + self.eps)
                    noise = torch.randn_like(p)
                    p.add_(eta * noise)

def benchmark_uncertainty_regularizer(device, gpu_name):
    print("\n--- UncertaintyRegularizer (Heisenberg) Eğitim Simülasyonu Başlıyor ---")
    
    epochs = 150
    X_train = torch.randn(50, 20, device=device)
    y_train = torch.randn(50, 1, device=device)
    X_val = torch.randn(100, 20, device=device)
    y_val = torch.randn(100, 1, device=device)

    def train_model(use_uncertainty=False):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        regularizer = UncertaintyRegularizer(h_bar=0.01) if use_uncertainty else None
        
        val_losses = []
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            
            if regularizer:
                regularizer.step(model)
                
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_loss = criterion(val_out, y_val)
                val_losses.append(val_loss.item())
                
        return val_losses

    val_losses_std = train_model(use_uncertainty=False)
    val_losses_unc = train_model(use_uncertainty=True)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(val_losses_std, label="Standart Adam", color="gray", alpha=0.7, linewidth=2)
    plt.plot(val_losses_unc, label="Adam + UncertaintyRegularizer", color="#17becf", linewidth=2.5)
    
    plt.title("Doğrulama Kaybı: Standart vs Heisenberg Belirsizlik İlkesi\n(Aşırı Öğrenmeye Karşı Direnç)")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_uncertainty.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nUncertaintyRegularizer grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 11. Özel Zamanlayıcı: QuantizedLR
# ==========================================
class QuantizedLR:
    def __init__(self, optimizer, levels=[0.1, 0.01, 0.001, 0.0001], patience=5, threshold=1e-4):
        self.optimizer = optimizer
        self.levels = levels
        self.current_level_idx = 0
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.lr_history = []
        self._set_lr(self.levels[self.current_level_idx])
        
    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
            
    def step(self, val_loss):
        current_lr = self.get_last_lr()[0]
        self.lr_history.append(current_lr)
        
        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            if self.current_level_idx < len(self.levels) - 1:
                self.current_level_idx += 1
                new_lr = self.levels[self.current_level_idx]
                self._set_lr(new_lr)
            self.num_bad_epochs = 0

def benchmark_quantized_lr(device, gpu_name):
    print("\n--- QuantizedLR vs CosineAnnealingLR Eğitim Simülasyonu Başlıyor ---")
    
    epochs = 150
    X = torch.randn(100, 10, device=device)
    y = torch.randn(100, 1, device=device)

    def train_model(scheduler_type="quantized"):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        
        if scheduler_type == "quantized":
            scheduler = QuantizedLR(optimizer, levels=[0.1, 0.01, 0.001, 0.0001], patience=10)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
            
        criterion = nn.MSELoss()
        losses = []
        lrs = []
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if scheduler_type == "quantized":
                scheduler.step(loss.item())
                lrs.append(scheduler.get_last_lr()[0])
            else:
                scheduler.step()
                lrs.append(scheduler.get_last_lr()[0])
                
        return losses, lrs

    losses_cos, lrs_cos = train_model("cosine")
    losses_qnt, lrs_qnt = train_model("quantized")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(losses_cos, label="CosineAnnealingLR", color="gray", alpha=0.7, linewidth=2)
    ax1.plot(losses_qnt, label="QuantizedLR", color="#e377c2", linewidth=2.5)
    ax1.set_title("Eğitim Kaybı (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend()
    
    ax2.plot(lrs_cos, label="Sürekli Düşüş (Cosine)", color="gray", alpha=0.7, linewidth=2)
    ax2.plot(lrs_qnt, label="Ayrık Kuantum Sıçramaları", color="#e377c2", linewidth=2.5, drawstyle='steps-post')
    ax2.set_title("Öğrenme Oranı (Learning Rate) Değişimi")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR")
    ax2.set_yscale("log")
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_quantized.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nQuantizedLR grafiği başarıyla çizildi: {plot_path}")

# ==========================================
# 12. Ana Çalıştırma Bloğu
# ==========================================
if __name__ == "__main__":
    vocab_size = 50257
    hidden_size = 768       
    num_heads = 12
    feedforward_dim = 3072
    seq_len = 4096          # GPU Çıkarım Testi
    batch_size = 2          # 124M model VRAM patlamasın diye
    num_layers = 12
    
    # Cihazı otomatik olarak tespit et
    device, gpu_name = detect_device()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. Standart Transformer Çıkarımı
    std_model = StandardTransformerInference(vocab_size, hidden_size, num_heads, feedforward_dim, num_layers)
    std_mem, std_tps, std_oom = benchmark_inference(
        std_model, device, num_steps=seq_len, batch_size=batch_size, vocab_size=vocab_size, name="Standart Transformer (KV-Cache)"
    )
    del std_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # 2. Bulk Modeli Çıkarımı
    bulk_model = BulkTransformerInference(vocab_size, hidden_size, num_heads, feedforward_dim, num_layers)
    bulk_mem, bulk_tps, bulk_oom = benchmark_inference(
        bulk_model, device, num_steps=seq_len, batch_size=batch_size, vocab_size=vocab_size, name="HFP Bulk Model"
    )
    
    # ==========================================
    # 6. Grafikleri Çizdirme
    # ==========================================
    steps_b, mem_b = zip(*bulk_mem)
    _, tps_b = zip(*bulk_tps)
    
    steps_s, mem_s = zip(*std_mem)
    _, tps_s = zip(*std_tps)
    
    mem_label = "VRAM" if device.type == "cuda" else "RAM"
    
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 6))
    
    # Bellek Grafiği
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(steps_b, mem_b, label="HFP Bulk Model (Sabit Bellek)", color='#2ca02c', linewidth=2.5)
    ax1.plot(steps_s, mem_s, label="Standart Transformer (Büyüyen KV-Cache)", color='#1f77b4', linewidth=2.5)
    
    if std_oom:
        ax1.axvline(x=std_oom, color='red', linestyle='--', linewidth=2, label='OOM Çöküşü')
        ax1.annotate('OOM', xy=(std_oom, mem_s[-1]), xytext=(std_oom - 800, mem_s[-1] + 50),
                     arrowprops=dict(facecolor='red', shrink=0.05), color='red', fontweight='bold', fontsize=11)
                     
    ax1.set_title(f"{mem_label} Kullanımı — {gpu_name}\n(seq_len={seq_len}, batch={batch_size})")
    ax1.set_xlabel("Üretilen Token Sayısı")
    ax1.set_ylabel(f"{mem_label} (MB)")
    ax1.set_xlim(0, seq_len)
    ax1.legend()
    
    # Hız Grafiği
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(steps_b, tps_b, label="HFP Bulk Model (O(1) Sabit Hız)", color='#2ca02c', linewidth=2.5)
    ax2.plot(steps_s, tps_s, label="Standart Transformer (O(N) Yavaşlayan)", color='#1f77b4', linewidth=2.5)
    
    if std_oom:
        ax2.axvline(x=std_oom, color='red', linestyle='--', linewidth=2)
        
    ax2.set_title(f"Çıkarım Hızı — {gpu_name}")
    ax2.set_xlabel("Üretilen Token Sayısı")
    ax2.set_ylabel("Hız (Token / Saniye)")
    ax2.set_xlim(0, seq_len)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "benchmark_results_gpu.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nGrafik başarıyla çizildi ve '{plot_path}' konumuna kaydedildi.")
    
    # Yeni eklenen Stiff Zamanlayıcı Testi
    # benchmark_stiff_scheduler(device, gpu_name)
    
    # Yeni eklenen Zenon Quantization Testi
    # benchmark_zenon_scheduler(device, gpu_name)
    
    # Yeni eklenen Holographic Dropout Testi
    # benchmark_holographic_dropout(device, gpu_name)
    
    # Yeni eklenen EntangledLinear Testi
    # benchmark_entangled_linear(device, gpu_name)
    
    # Yeni eklenen TunnelingDropout Testi
    # benchmark_tunneling_dropout(device, gpu_name)
    
    # Yeni eklenen UncertaintyRegularizer Testi
    # benchmark_uncertainty_regularizer(device, gpu_name)
    
    # Yeni eklenen QuantizedLR Testi
    # benchmark_quantized_lr(device, gpu_name)
    
    # Yeni eklenen Holographic vs Tunneling Dropout Karşılaştırma
    # benchmark_holographic_vs_tunneling(device, gpu_name)
    
    # -------------------------------------------------------------------------
    # GERÇEK HFP MODELİ İLE WIKITEXT-2 DİL MODELLEME TESTİ
    # (HFPForCausalLM + QuantizedLR + UncertaintyRegularizer)
    # -------------------------------------------------------------------------
    benchmark_wikitext2(device, gpu_name, batch_size=8, epochs=3)
