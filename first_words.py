from hfp import HFPConfig
from hfp import HFPForCausalLM
import torch

# Modeli yükle (rastgele ağırlıklarla - henüz eğitilmedi)
config = HFPConfig(vocab_size=50000, hidden_size=256)
model = HFPForCausalLM(config)
model.eval()

# İlk soruyu sor
input_ids = torch.tensor([[1, 15, 23, 48]])  # Rastgele başlangıç tokenları

print("=" * 50)
print("HFP BulkTrigger İlk Konuşma Testi")
print("=" * 50)
print("\nSoru: Merhaba, sen kimsin?\n")

# Modelin cevabını üret
with torch.no_grad():
    output = model.generate(
        input_ids, 
        max_new_tokens=30, 
        do_sample=True, 
        temperature=0.8
    )

print("Modelin Cevabı:")
print(output)
print("\n" + "=" * 50)
print("Not: Model henüz eğitilmediği için cevaplar rastgele token'lardan oluşur.")
print("Bu, bir bebeğin çıkardığı ilk sesleri duymak gibidir.")
print("=" * 50)
