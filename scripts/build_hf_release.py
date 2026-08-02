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

"""
scripts/build_hf_release.py - HuggingFace Yayın Paketi Derleyici.

Kanonik `hfp/core/` ve `hfp/models/` kod tabanından HuggingFace Hub uyumlu
yayın paketini programmatik olarak derler. Takipsiz dizinlerdeki elle senkronizasyon
hatalarını ve kod sürüklenmesini (drift) önler.

Kullanım:
    python scripts/build_hf_release.py [--output_dir dist/hf_release] [--check-only]
"""

import argparse
import os
import py_compile
import shutil
import sys

# Proje kök dizini
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Kanonik kaynak dosyaları -> Hedef dosya adı eşleşmeleri
CANONICAL_FILES = {
    os.path.join(REPO_ROOT, "hfp", "core", "bulk_trigger_decoder.py"): "bulk_trigger_decoder.py",
    os.path.join(REPO_ROOT, "hfp", "core", "hfp_bulk_state.py"): "hfp_bulk_state.py",
    os.path.join(REPO_ROOT, "hfp", "core", "hfp_config.py"): "hfp_config.py",
    os.path.join(REPO_ROOT, "hfp", "core", "hfp_utils.py"): "hfp_utils.py",
    os.path.join(REPO_ROOT, "hfp", "models", "configuration_hfp.py"): "configuration_hfp.py",
    os.path.join(REPO_ROOT, "hfp", "models", "modeling_hfp.py"): "modeling_hfp.py",
    os.path.join(REPO_ROOT, "LICENSE"): "LICENSE",
}


def build_release(output_dir: str, check_only: bool = False):
    print(f"=== HFP HuggingFace Yayın Paketi İnşası ===")
    print(f"Kanonik Kaynak: {REPO_ROOT}")
    print(f"Hedef Dizin  : {output_dir}")

    # Dosya varlık kontrolü
    missing_sources = [src for src in CANONICAL_FILES if not os.path.exists(src)]
    if missing_sources:
        print(f"[HATA] Eksik kanonik kaynak dosyaları var: {missing_sources}")
        sys.exit(1)

    if check_only:
        print("[KONTROL] Tüm kanonik kaynak dosyaları mevcut. Sözdizim doğrulaması yapılıyor...")
        for src in CANONICAL_FILES:
            if src.endswith(".py"):
                py_compile.compile(src, doraise=True)
        print("[BAŞARILI] Tüm kanonik dosyalar geçerli ve derlenebilir.")
        return

    os.makedirs(output_dir, exist_ok=True)

    copied_files = []
    for src, dst_name in CANONICAL_FILES.items():
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy2(src, dst_path)
        copied_files.append(dst_name)

        # Pythondosya doğrulama
        if dst_name.endswith(".py"):
            py_compile.compile(dst_path, doraise=True)

    print("-" * 50)
    print(f"[BAŞARILI] {len(copied_files)} dosya kanonik kaynaktan başarıyla kopyalandı ve doğrulandı:")
    for fname in copied_files:
        print(f"  - {fname}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="HFP HuggingFace Yayın Paketi Derleyici")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "dist", "hf_release"),
        help="Yayın paketinin oluşturulacağı hedef dizin (varsayılan: dist/hf_release)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Sadece kanonik kaynakların doğruluğunu ve varlığını kontrol eder, kopyalama yapmaz",
    )
    args = parser.parse_args()
    build_release(output_dir=args.output_dir, check_only=args.check_only)


if __name__ == "__main__":
    main()
