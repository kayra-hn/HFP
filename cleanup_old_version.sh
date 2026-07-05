#!/usr/bin/env bash
# [GUVENLI TEMIZLIK] Eski/mukerrer HFP dosyalarini _archive_old/'a TASIR (SILMEZ).
# hfp_arch/ (kanonik surum) ELLENMEZ. Once bu listeyi inceleyin, sonra kosun.
# Geri almak icin: _archive_old/ icindekileri geri tasiyin. Repo icinde calisir:
#   cd HFP_Project && bash cleanup_old_version.sh
set -eu
cd "$(dirname "$0")"
DEST="_archive_old"
mkdir -p "$DEST"

# hfp_arch, .git, ve arsiv klasoru HARIC tasinacak eski ogeler:
ITEMS=(
  hfp                     # eski core (bugli)
  hfp_upload_temp
  hfp_hf_test_save
  trained_hfp_baby
  hfp_weights.pt          # ~521 MB
  scratch                 # fizik sembolik hesaplari (arsivde saklanir)
  benchmark_results_gpu.png
  optimizer_stability_results.png
  passkey_1b_results.png
  eval_memory_scaling.py
  eval_optimizer_stability.py
  eval_passkey.py
  train.py
  debug_memory.py
  hf_test.py
  push_to_hf.py
  clean_hf_repo.py
  run_wrapper.py
  error.log
  __pycache__
)

echo "Asagidakiler $DEST/'a TASINACAK (silinmeyecek). hfp_arch/ korunur."
for it in "${ITEMS[@]}"; do
  [ -e "$it" ] && echo "  - $it"
done
read -r -p "Devam edilsin mi? [e/H] " ans
[ "$ans" = "e" ] || { echo "Iptal."; exit 0; }

for it in "${ITEMS[@]}"; do
  if [ -e "$it" ]; then
    mv -v "$it" "$DEST/"
  fi
done
echo "Tamam. Kanonik surum: hfp_arch/ (bkz. hfp_arch/REPO_STRUCTURE.md)."
echo "Geri almak icin _archive_old/ icinden geri tasiyabilirsiniz."
