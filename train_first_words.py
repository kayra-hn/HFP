import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader
from hfp import HFPConfig, HFPForCausalLM
from hfp import UncertaintyRegularizer, QuantizedLR

def main():
    print("1. Tokenizer yükleniyor...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # GPT2 tokenizer'da pad token olmadığı için ekliyoruz
    tokenizer.pad_token = tokenizer.eos_token

    print("2. Veri seti yükleniyor (Salesforce/wikitext-2-raw-v1, %10)...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:10%]")
    
    def tokenize_function(examples):
        # Uzun metinleri kesiyoruz, maksimum uzunluk 128
        return tokenizer(examples["text"], truncation=True, max_length=128)
        
    print("Veri seti tokenize ediliyor...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Dataset'i PyTorch tensörlerine çevir
    tokenized_datasets.set_format("torch")
    
    # DataCollator modelin anlayacağı formata (input_ids, labels) çevirir
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # DataLoader oluştur (DataCollator pad/batch işlemlerini yapar)
    dataloader = DataLoader(tokenized_datasets, batch_size=2, collate_fn=data_collator)

    print("3. Model başlatılıyor...")
    config = HFPConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        bulk_dim=64,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = HFPForCausalLM(config)

    print("4. Eğitim döngüsü (Custom PyTorch Loop) hazırlanıyor...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # [Kuantum Optimizasyon Eklentileri]
    # h_bar 0.005, çok ufak bir gürültü ekleyeceğiz
    regularizer = UncertaintyRegularizer(model, h_bar=0.005)
    # LR seviyeleri: 5e-5'ten başlayıp 1e-6'ya kadar kuantum sıçraması yapacak
    # patience=10 (her 10adımda düşme yapmasın diye epoch bazlı ya da adım bazlı olabilir.
    # Adım bazlı yapıyoruz, her 50 adımda bir plato kontrolü yapacağız)
    scheduler = QuantizedLR(optimizer, energy_levels=[5e-5, 1e-5, 5e-6, 1e-6], patience=5, threshold=0.05)
    
    model.train()
    epochs = 3
    gradient_accumulation_steps = 4
    
    print(f"Eğitim başlıyor... (Epoch: {epochs}, Batch Size: 2)")
    
    for epoch in range(epochs):
        total_loss = 0
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Gradient Accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                
                # Heisenberg Belirsizlik Regülarizasyonu (Ağırlıklara Gürültü Ekle)
                regularizer.step()
                
                optimizer.zero_grad()
                
            if step % 10 == 0:
                print(f"Adım {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {optimizer.param_groups[0]['lr']}")
                
            # Quantized LR: Sadece belli adımlarda değerlendir
            if step % 50 == 0 and step > 0:
                # O anki ortalama kaybı scheduler'a ver
                current_avg_loss = total_loss / (step + 1)
                scheduler.step(current_avg_loss)
                
        print(f"Epoch {epoch+1} Ortalama Loss: {total_loss / len(dataloader):.4f}")
    
    print("\n5. Model kaydediliyor...")
    model.save_pretrained("./trained_hfp_baby")
    tokenizer.save_pretrained("./trained_hfp_baby")
    
    print("\n6. Eğitim sonrası konuşma testi yapılıyor...")
    model.eval()
    
    prompt = "Merhaba, sen kimsin?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        
    cevap = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("=" * 50)
    print("HFP BulkTrigger Eğitim Sonrası Testi")
    print("=" * 50)
    print(f"\nSoru: {prompt}\n")
    print("Modelin Cevabı:")
    print(cevap)
    print("\n" + "=" * 50)
    print("Eğitim başarıyla tamamlandı ve test edildi!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
