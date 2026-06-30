import torch
from transformers import AutoTokenizer
from hfp import HFPForCausalLM, HFPConfig

def run_benchmark():
    print("[1] Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("[2] HFP Modeli (124M) başlatılıyor...")
    config = HFPConfig(vocab_size=len(tokenizer))
    model = HFPForCausalLM(config)
    model.eval()
    
    prompt = "The universe is a holographic"
    print(f"\n[3] Girdi Cümlesi: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("[4] Üretim (Inference) başlatılıyor...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("🎯 BENCHMARK TEST SONUCU (Rastgele Ağırlıklar):")
    print("="*50)
    print(response)
    print("="*50)
    print("Durum: BAŞARILI! Model çökmeden nedensel (causal) token üretebiliyor.")

if __name__ == "__main__":
    run_benchmark()
