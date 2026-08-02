# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
