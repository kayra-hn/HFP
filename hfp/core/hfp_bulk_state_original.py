import torch
import torch.nn as nn

class HFPBulkState(nn.Module):
    """
    HFPBulkState: Transformer Modelleri için Hiyerarşik Bellek Yönetim Sistemi.

    Bu sınıf bilgiyi üç farklı zaman ölçeğinde saklar ve yönetir:
    - short: Son 8 token (tam vektör temsilleri).
    - medium: Her 32 token'da bir güncellenen hareketli ortalama (moving average) vektörü.
    - long: Her 128 token'da bir güncellenen ve önemli bilgileri seçip saklayan vektör.
    """

    def __init__(self, hidden_size, short_len=8, medium_freq=32, long_freq=128, medium_momentum=0.1):
        super(HFPBulkState, self).__init__()
        self.hidden_size = hidden_size
        self.short_len = short_len
        self.medium_freq = medium_freq
        self.long_freq = long_freq
        self.medium_momentum = medium_momentum
        
        # Uzun vadeli bellek için önem kapısı (Gating Mechanism).
        # Hangi bilgilerin uzun vadeli belleğe aktarılacağını öğrenir.
        self.importance_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # [OPT-3] Gate Dropout: Eğitimde aşırı öğrenmeyi (overfitting) engeller,
        # inference'ta otomatik devre dışı kalır.
        self.gate_dropout = nn.Dropout(0.1)
        
        # [OPT-6] DRY: Durum değişkenlerini reset_state() ile başlat
        self.reset_state()

    def reset_state(self):
        """Yeni bir dizi/epoch başladığında bellek durumlarını sıfırlar."""
        self.token_count = 0
        self.short_memory = None
        self.medium_memory = None
        self.long_memory = None
        # [OPT-1] Ring Buffer için dairesel dizin ve doluluk sayacı
        self.write_idx = 0
        self.fill_count = 0

    def _get_short_view(self):
        """
        [OPT-1] Ring buffer'dan sadece dolu slotları döndürür.
        PyTorch'ta dilimleme bir 'view' (görünüm) operasyonudur,
        yeni tensör tahsis etmez — sıfır bellek maliyeti.
        """
        if self.training:
            return self.short_memory
        if self.fill_count < self.short_len:
            return self.short_memory[:, :self.fill_count, :]
        return self.short_memory

    def _get_short_mean(self):
        """
        [OPT-2] Short memory'nin ortalamasını hesaplar.
        Yardımcı metot olarak çıkarıldı, böylece medium ve long
        güncellemelerinde tekrar hesaplama yapılmaz.
        """
        view = self._get_short_view()
        return view.mean(dim=1)

    def update(self, x, past_state=None):
        if past_state is not None:
            self.short_memory, self.medium_memory, self.long_memory, self.token_count, self.write_idx, self.fill_count = past_state
        elif not self.training and self.token_count == 0:
            self.reset_state()

        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.size()
        device = x.device
        dtype = x.dtype
        
        # Batch boyutu değişirse (örn. son batch) state'i resetle
        if self.short_memory is not None and self.short_memory.size(0) != batch_size:
            self.reset_state()
        
        # [CRITICAL FIX] Önceki batch'in computation graph'ını kopar (Truncated BPTT)
        if self.short_memory is not None:
            self.short_memory = self.short_memory.detach()
        if self.medium_memory is not None:
            self.medium_memory = self.medium_memory.detach()
        if self.long_memory is not None:
            self.long_memory = self.long_memory.detach()
        
        if self.medium_memory is None:
            self.medium_memory = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.long_memory is None:
            self.long_memory = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            
        for i in range(seq_len):
            if self.training:
                if self.short_memory is None:
                    self.short_memory = x[:, i:i+1, :]
                else:
                    self.short_memory = torch.cat([self.short_memory, x[:, i:i+1, :]], dim=1)
                    if self.short_memory.size(1) > self.short_len:
                        self.short_memory = self.short_memory[:, -self.short_len:, :]
            else:
                if self.short_memory is None:
                    self.short_memory = torch.zeros(batch_size, self.short_len, self.hidden_size, device=device, dtype=dtype)
                self.short_memory[:, self.write_idx, :] = x[:, i, :]
                self.write_idx = (self.write_idx + 1) % self.short_len
                self.fill_count = min(self.fill_count + 1, self.short_len)

            self.token_count += 1
            context_summary = self._get_short_mean()
            
            if self.token_count % self.medium_freq == 0:
                self.medium_memory = (1.0 - self.medium_momentum) * self.medium_memory + \
                                      self.medium_momentum * context_summary
                                      
            combined_features = torch.cat([self.medium_memory, context_summary], dim=-1)
            gate = torch.sigmoid(self.gate_dropout(self.importance_gate(combined_features)))
            self.long_memory = (1.0 - gate) * self.long_memory + gate * context_summary
            
        new_past_state = (self.short_memory, self.medium_memory, self.long_memory, self.token_count, self.write_idx, self.fill_count)
        return self._get_short_view(), self.medium_memory, self.long_memory, new_past_state

# Örnek Kullanım:
if __name__ == "__main__":
    batch_size = 2
    hidden_size = 512
    seq_length = 200 # Toplam 200 tokenlık bir sequence
    
    # Modeli oluştur
    memory_system = HFPBulkState(hidden_size=hidden_size)
    
    # Rastgele veri üret
    dummy_input = torch.randn(batch_size, seq_length, hidden_size)
    
    # Belleği güncelle
    short_mem, medium_mem, long_mem = memory_system.update(dummy_input)
    
    print(f"Toplam İşlenen Token: {memory_system.token_count}")
    print(f"Short Memory Boyutu: {short_mem.shape}")   # (2, 8, 512)
    print(f"Medium Memory Boyutu: {medium_mem.shape}") # (2, 512)
    print(f"Long Memory Boyutu: {long_mem.shape}")     # (2, 512)
