import os
import shutil
from huggingface_hub import HfApi, login
from hfp.models.configuration_hfp import HFPConfig
from hfp.models.modeling_hfp import HFPForCausalLM

def prepare_hf_repo():
    print("Hugging Face Yükleme Aracına Hoş Geldiniz! 🚀\n")
    token = input("Lütfen hf_ ile başlayan Hugging Face Access Token'ınızı yapıştırın: ").strip()
    
    if not token.startswith("hf_"):
        print("Hata: Token 'hf_' ile başlamalıdır. Lütfen doğru token kopyaladığınızdan emin olun.")
        return
        
    print("\nGiriş yapılıyor...")
    try:
        login(token=token)
    except Exception as e:
        print(f"Giriş başarısız oldu: {e}")
        return
        
    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/HFP-O1-Memory-Model"
    
    print(f"\nHub üzerinde '{repo_id}' reposu oluşturuluyor...")
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    except Exception as e:
        print(f"Uyarı: {e}")
        
    print("Model konfigürasyonu ve mimarisi (O(1) Memory) paketleniyor...")
    
    # Hugging Face için konfigürasyon
    config = HFPConfig(
        vocab_size=50000,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        short_len=8,
        bulk_dim=64
    )
    
    # trust_remote_code=True için zorunlu mapping (Klasör hiyerarşisi korunacak)
    config.auto_map = {
        "AutoConfig": "hfp.models.configuration_hfp.HFPConfig",
        "AutoModelForCausalLM": "hfp.models.modeling_hfp.HFPForCausalLM"
    }
    
    # Modeli başlat (Henüz eğitilmemiş, saf mimari)
    model = HFPForCausalLM(config)
    
    save_dir = "hfp_upload_temp"
    os.makedirs(save_dir, exist_ok=True)
    
    # Ağırlıkları ve Config'i kaydet
    model.save_pretrained(save_dir, safe_serialization=True)
    config.save_pretrained(save_dir)
    
    print("Özel mimari kodları kopyalanıyor (trust_remote_code için)...")
    # Tüm hfp klasörünü olduğu gibi kopyala ki relative import'lar bozulmasın
    if os.path.exists(f"{save_dir}/hfp"):
        shutil.rmtree(f"{save_dir}/hfp")
    shutil.copytree("hfp", f"{save_dir}/hfp")
    
    print("\nModel Hub'a Pushlanıyor... (Bu işlem internet hızınıza bağlı olarak 1-2 dakika sürebilir)")
    
    # Model ve Kodları yükle
    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial commit: O(1) Memory HFP Architecture Core"
    )
    
    print("Grafikler ve Benchmark sonuçları yükleniyor...")
    # Grafiği Yükle
    for img in ["benchmark_results_gpu.png", "benchmark_results_wikitext2.png"]:
        if os.path.exists(img):
            try:
                api.upload_file(
                    path_or_fileobj=img,
                    path_in_repo=img,
                    repo_id=repo_id,
                    repo_type="model"
                )
            except Exception:
                pass
        
    print("\n" + "="*50)
    print("🚀 YÜKLEME BAŞARILI! 🚀")
    print(f"Model Sayfanız: https://huggingface.co/{repo_id}")
    print("\nArtık dünyanın her yerinden araştırmacılar mimarinizi şu kodla indirip test edebilir:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{repo_id}', trust_remote_code=True)")
    print("="*50)

if __name__ == "__main__":
    prepare_hf_repo()
