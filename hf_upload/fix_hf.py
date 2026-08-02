from huggingface_hub import HfApi

repo_id = "kayrahan35/HFP-O1-Memory-Model"
api = HfApi()

print("1. Hugging Face'e baglaniliyor...")
# Yanlislikla yuklenen klasoru hub'dan sil
try:
    print("2. Yanlislikla yuklenen 'hf_release' klasoru siliniyor...")
    api.delete_folder(path_in_repo="hf_release", repo_id=repo_id)
    print("Klasor basariyla silindi.")
except Exception as e:
    print(f"Klasor silinirken bir uyari olustu (zaten silinmis olabilir): {e}")

# hf_release icindeki dosyalari root dizine yukle (eskilerin uzerine yazar)
print("3. Dogru dosyalar kok dizine (root) yukleniyor (eskilerin uzerine yazilacak)...")
api.upload_folder(
    folder_path="hf_release",
    repo_id=repo_id,
    path_in_repo="." # Kok dizine yukle
)
print("4. Islem tamam! Hugging Face sayfanizi yenileyebilirsiniz.")
