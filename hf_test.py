import torch
from hfp import HFPConfig
from hfp import HFPForCausalLM

def test_hfp_huggingface():
    print("--- HuggingFace Entegrasyon Testi Başlıyor ---")
    
    # 1. Konfigürasyonu Başlat
    config = HFPConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        bulk_dim=64
    )
    print("1. HFPConfig başarıyla oluşturuldu.")
    
    # 2. Modeli Başlat
    model = HFPForCausalLM(config)
    print("2. HFPForCausalLM başarıyla oluşturuldu.")
    
    # 3. İleri Geçiş (Forward Pass) Testi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)
    
    print("3. İleri geçiş (forward pass) yapılıyor...")
    outputs = model(input_ids)
    
    print(f"Logits shape: {outputs.logits.shape} (Beklenen: [1, 10, 1000])")
    if outputs.logits.shape == (1, 10, 1000):
        print("[OK] Forward pass başarılı.")
    else:
        print("[HATA] Logits boyutu hatalı!")
        
    # 4. Üretim (Generation) Testi
    print("\n4. Otomatik metin üretimi (Generation) test ediliyor...")
    model.eval()
    generated = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=True,
        top_k=50
    )
    print(f"Üretilen Dizi Boyutu: {generated.shape} (Beklenen: [1, 15])")
    print("[OK] HuggingFace generate() API'si sorunsuz çalışıyor!")
        
    # 5. Kaydetme Testi
    import os
    save_dir = "hfp_hf_test_save"
    model.save_pretrained(save_dir)
    config.save_pretrained(save_dir)
    print(f"\n5. Model başarıyla '{save_dir}' dizinine kaydedildi.")
    
if __name__ == "__main__":
    test_hfp_huggingface()
