import os
from huggingface_hub import HfApi, login

def clean_hf():
    print("🧹 Hugging Face Temizlik Aracına Hoş Geldiniz!")
    token = input("Lütfen hf_ ile başlayan Hugging Face Access Token'ınızı yapıştırın: ").strip()
    
    if not token.startswith("hf_"):
        print("Hata: Geçersiz token.")
        return
        
    api = HfApi()
    username = api.whoami(token=token)["name"]
    repo_id = f"{username}/HFP-O1-Memory-Model"
    
    files_to_delete = [
        "benchmark_quality_results.png",
        "benchmark_results_wikitext2.png",
        "HF_MODEL_CARD.md",
        "benchmark_quality.py",
        "passkey_test.py",
        "push_hf_benchmarks.py",
        "train.py",
        "train_1b_cloud.py",
        "train_first_words.py",
        "chat.py",
        "first_words.py",
        "decode_first_words.py",
        "benchmark.py",
        "benchmark_test.py",
        "hfp/physics/physics_optimizers.py",
        "hfp/physics/__init__.py",
        "hfp/core/hfp_bulk_state_original.py"
    ]
    
    print(f"\n{repo_id} deposundaki eski/gereksiz dosyalar siliniyor...")
    
    deleted_count = 0
    for file in files_to_delete:
        try:
            api.delete_file(path_in_repo=file, repo_id=repo_id, repo_type="model", token=token)
            print(f"✅ Başarıyla silindi: {file}")
            deleted_count += 1
        except Exception as e:
            # Dosya zaten yoksa hata verir, sorun değil.
            pass
            
    print(f"\nTemizlik tamamlandı! Toplam {deleted_count} eski dosya Hugging Face'den tamamen kazındı.")

if __name__ == "__main__":
    clean_hf()
