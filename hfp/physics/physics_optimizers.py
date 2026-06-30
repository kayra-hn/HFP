import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

class UncertaintyRegularizer:
    """
    Heisenberg Belirsizlik İlkesine dayalı regülarizasyon (Uncertainty Regularization).
    Büyük gradyanların kesin konum (momentum belirsiz), küçük gradyanların ise 
    kesin momentum (konum belirsiz) olduğu varsayımıyla ağırlıklara ters orantılı 
    gürültü (noise) enjekte eder: w += η * randn_like(w), η = ħ / (||g|| + ε)
    """
    def __init__(self, model, h_bar=0.001, eps=1e-8):
        self.model = model
        self.h_bar = h_bar
        self.eps = eps

    def step(self):
        """
        Optimizer step'ten SONRA veya ÖNCE çağrılabilir. 
        Ağırlıklara gradyan büyüklüğüne göre gürültü ekler.
        """
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    # Gradyanın büyüklüğünü hesapla (L2 norm)
                    g_norm = torch.norm(param.grad)
                    
                    # Belirsizlik prensibine göre enjekte edilecek gürültü genliği
                    # Büyük gradyan -> Küçük Gürültü (Daha Az Belirsizlik)
                    # Küçük gradyan -> Büyük Gürültü (Daha Çok Belirsizlik)
                    # [KRITIK GÜNCELLEME]: Gürültü patlamasını önlemek için üst sınır koyuyoruz
                    eta = torch.clamp(self.h_bar / (g_norm + self.eps), max=self.h_bar * 10)
                    
                    # Gürültüyü ağırlıklara uygula
                    noise = eta * torch.randn_like(param)
                    param.add_(noise)

class QuantizedLR(object):
    """
    Kuantize Edilmiş Enerji Seviyeleri tabanlı Öğrenme Oranı Zamanlayıcısı.
    lr değerini yumuşak bir şekilde düşürmek yerine, yalnızca ayrık (discrete)
    enerji seviyelerine atlayarak düşürür. Bir plateau tespit ettiğinde bir alt seviyeye geçer.
    """
    def __init__(self, optimizer, energy_levels=[5e-4, 1e-4, 5e-5, 1e-5], patience=2, threshold=1e-4):
        self.optimizer = optimizer
        self.energy_levels = sorted(energy_levels, reverse=True) # En yüksekten en düşüğe
        self.patience = patience
        self.threshold = threshold
        
        self.current_level_idx = 0
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        
        # Başlangıç LR'sini en yüksek enerji seviyesine eşitle
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.energy_levels[self.current_level_idx]
        
    def step(self, metrics, epoch=None):
        # Plateau algılama
        current_loss = metrics
        
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        # Eğer sabır tükenirse bir alt enerji seviyesine "Kuantum Sıçraması" yap
        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            if self.current_level_idx < len(self.energy_levels) - 1:
                self.current_level_idx += 1
                new_lr = self.energy_levels[self.current_level_idx]
                
                print(f"[QuantizedLR] Plateau algılandı! Enerji seviyesi düşürüldü -> {new_lr}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

if __name__ == "__main__":
    # Test
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # 1. Uncertainty Regularizer Test
    loss = model(torch.randn(1, 10)).sum()
    loss.backward()
    
    reg = UncertaintyRegularizer(model, h_bar=0.1)
    reg.step()
    print("Uncertainty Regularizer step successful!")
    
    # 2. QuantizedLR Test
    scheduler = QuantizedLR(optimizer, energy_levels=[0.1, 0.01, 0.001], patience=1)
    scheduler.step(10.0) # Ilk loss
    scheduler.step(10.0) # Plateau -> düşmeli
    assert optimizer.param_groups[0]['lr'] == 0.01
    print("QuantizedLR jump successful!")
