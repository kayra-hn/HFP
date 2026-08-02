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

# Elle yazılmış takipli varlıklar -> Hedef dosya adı eşleşmeleri
AUTHORED_ASSETS = {
    os.path.join(REPO_ROOT, "hf_upload", "GRAFT_MODEL_CARD.md"): "GRAFT_MODEL_CARD.md",
    os.path.join(REPO_ROOT, "hf_upload", "hf_release", "README.md"): "README.md",
    os.path.join(REPO_ROOT, "hf_upload", "YAYIN_ADIMLARI.md"): "YAYIN_ADIMLARI.md",
    os.path.join(REPO_ROOT, "hf_upload", "fix_hf.py"): "fix_hf.py",
    os.path.join(REPO_ROOT, "hf_upload", "make_hf_checkpoint.py"): "make_hf_checkpoint.py",
    os.path.join(REPO_ROOT, "hf_upload", "hf_release", "config.json"): "config.json",
    os.path.join(REPO_ROOT, "hf_upload", "hf_release", "generation_config.json"): "generation_config.json",
}


def build_release(output_dir: str, check_only: bool = False):
    print(f"=== HFP HuggingFace Yayın Paketi İnşası ===")
    print(f"Kanonik Kaynak: {REPO_ROOT}")
    print(f"Hedef Dizin  : {output_dir}")

    all_sources = {**CANONICAL_FILES, **AUTHORED_ASSETS}

    # Dosya varlık kontrolü
    missing_sources = [src for src in all_sources if not os.path.exists(src)]
    if missing_sources:
        print(f"[HATA] Eksik kaynak dosyaları var: {missing_sources}")
        sys.exit(1)

    if check_only:
        print("[KONTROL] Tüm kaynak dosyaları (kanonik + elle yazılmış) mevcut. Sözdizim doğrulaması yapılıyor...")
        for src in all_sources:
            if src.endswith(".py"):
                py_compile.compile(src, doraise=True)
        print(f"[BAŞARILI] {len(all_sources)} kaynak dosyanın tamamı geçerli ve derlenebilir.")
        return

    os.makedirs(output_dir, exist_ok=True)

    copied_canonical = []
    copied_authored = []

    for src, dst_name in CANONICAL_FILES.items():
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy2(src, dst_path)
        copied_canonical.append(dst_name)
        if dst_name.endswith(".py"):
            py_compile.compile(dst_path, doraise=True)

    for src, dst_name in AUTHORED_ASSETS.items():
        dst_path = os.path.join(output_dir, dst_name)
        shutil.copy2(src, dst_path)
        copied_authored.append(dst_name)
        if dst_name.endswith(".py"):
            py_compile.compile(dst_path, doraise=True)

    print("-" * 60)
    print(f"[BAŞARILI] Toplam {len(copied_canonical) + len(copied_authored)} varlık paketlendi:")
    print(f"  Kanonik Kaynaklar ({len(copied_canonical)}): {', '.join(copied_canonical)}")
    print(f"  Elle Yazılmış Varlıklar ({len(copied_authored)}): {', '.join(copied_authored)}")
    print("-" * 60)


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
