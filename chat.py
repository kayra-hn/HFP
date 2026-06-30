import torch
from transformers import AutoTokenizer
from hfp import HFPForCausalLM, HFPConfig
import sys

def main():
    print("==================================================")
    print("🤖 HFP Yeni Doğan (Sayıklayan) Modele Hoş Geldiniz!")
    print("==================================================")
    
    # 1. Cihaz tespiti
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SİSTEM] Cihaz: {device.type.upper()} üzerinden çalıştırılıyor.")
    
    # 2. Tokenizer (GPT-2'nin standart kelime sözlüğünü kullanıyoruz)
    print("[SİSTEM] Kelime sözlüğü (Tokenizer) yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # 3. Model Kurulumu (Eğitimdeki boyutların birebir aynısı)
    print("[SİSTEM] HFPForCausalLM Modeli oluşturuluyor (124M Parametre Ağır Siklet)...")
    config = HFPConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        bulk_dim=128,
        short_len=16
    )
    model = HFPForCausalLM(config).to(device)
    
    import os
    save_path = "hfp_weights.pt"
    if os.path.exists(save_path):
        print(f"[BAŞARI] Eğitilmiş beyin dosyası bulundu ({save_path}). Yükleniyor...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("🤖 HFP Model: Bilinç ve hafıza başarıyla yüklendi! Artık anlamlı konuşabilirim.")
    else:
        print("[UYARI] Eğitilmiş beyin dosyası bulunamadı! Lütfen önce 'train.py' dosyasını çalıştırın.")
        print("Şu an tamamen rastgele (sayıklayan) modda çalışıyor.\n")

    model.eval() # Çıkarım (Sohbet) modu
    
    print("\n[HAZIR] Sistem aktif. Modele bir başlangıç cümlesi yazın (Çıkmak için 'q' veya 'quit' yazın).")

    while True:
        try:
            user_input = input("Siz: ")
            if user_input.strip().lower() in ['q', 'quit', 'çıkış']:
                print("Görüşmek üzere!")
                break
            if not user_input.strip():
                continue
            
            # Cümleyi sayılara (Token) çevir ve GPU/CPU'ya gönder
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            
            # Modelden kendi cümlelerini uydurmasını isteyelim
            # Sadece input_ids'i gönderiyoruz çünkü HFP mimarisi attention_mask kullanmıyor.
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=30,      # Maksimum 30 kelime daha uydur
                    do_sample=True,         # Rastgeleliği aç
                    temperature=0.8,        # Yaratıcılık / Sayıklama seviyesi
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Cümleyi sadece onun eklediği kısım olarak ayıralım
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Cümleyi sadece onun eklediği kısım olarak ayıralım (İsterseniz tam cümleyi de görebilirsiniz)
            print(f"\nHFP Model: {response}\n")
            
        except KeyboardInterrupt:
            print("\nÇıkış yapılıyor...")
            break
        except Exception as e:
            print(f"\n[HATA]: {e}")

if __name__ == "__main__":
    main()
