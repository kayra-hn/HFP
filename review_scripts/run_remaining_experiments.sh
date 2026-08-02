#!/usr/bin/env bash
# Hyper Flux Projection (HFP) — O(1)-memory causal language model
# Copyright (C) 2026 Kayrahan Yılmaz  —  AGPL-3.0
#
# [FAZ 1 SURUCUSU] Evde/GPU'da acik kalan deney hucrelerini SONUNA KADAR kosar.
# Scriptler zaman-butcesi asilinca checkpoint'e yazip cikar; bu surucu her hucreyi
# "FINAL/EGITIM BITTI" cikana dek yeniden cagirir. GPU varsa scriptler otomatik
# CUDA kullanir. Repo kokunden calistirin:  bash review_scripts/run_remaining_experiments.sh
set -u
cd "$(dirname "$0")/.."                       # repo koku
export HFP_CKPT_DIR="${HFP_CKPT_DIR:-checkpoints}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
BUDGET="${BUDGET:-90}"                         # tek cagri basina saniye (GPU'da buyutun)
PY="${PY:-python}"

run_until_done () {                            # $1=etiket  $2..=komut
  local tag="$1"; shift
  echo ">>> $tag"
  for i in $(seq 1 200); do
    out="$("$@" 2>&1)"; echo "$out"
    echo "$out" | grep -Eq "FINAL|EGITIM BITTI|EGITIM BITTI ->" && { echo ">>> $tag TAMAM"; return 0; }
    echo "$out" | grep -Eq "CKPT" || { echo ">>> $tag beklenmeyen cikti, duruyorum"; return 1; }
  done
  echo ">>> $tag: 200 cagriyi asti"; return 1
}

# --- 1) Harici GLA baseline (yeni) — 3 seed ---
for s in 0 1 2; do
  run_until_done "gla-baseline s$s" $PY review_scripts/baseline_compare.py $s $BUDGET
done

# --- 2) Saf-bellek ablasyonu (yeni) — 3 seed (her cagri full+pure iki kolu surdurur) ---
for s in 0 1 2; do
  run_until_done "pure-mem s$s" $PY review_scripts/pure_memory_ablation.py $s $BUDGET
done

# --- 3) DPFP x uzunluk-genellemesi: eksik seed 0 ve 1 (Ek 9'da yalniz s2 vardi) ---
for s in 0 1; do
  LG_VARIANT=dpfp run_until_done "dpfp-lg train s$s" env LG_VARIANT=dpfp $PY review_scripts/length_gen.py train $s $BUDGET
  echo ">>> dpfp-lg eval s$s"; LG_VARIANT=dpfp $PY review_scripts/length_gen.py eval $s
done

# --- 4) Streaming-mix: eksik seed 1 ve 2 (Ek 12'de yalniz s0 vardi), 4 kol ---
for s in 1 2; do
  for combo in "additive elu" "delta elu" "additive dpfp" "delta dpfp"; do
    set -- $combo
    run_until_done "stream-mix $1/$2 s$s" $PY review_scripts/streaming_mix.py "$1" "$2" "$s" $BUDGET
  done
done

# --- 5) FAZ 1B: cubic_flux ADIL testi (2x2 retention x feature-map, uzun-ufuk) ---
# Ana izgara: lr 1e-3, 3 seed x {exp,cubic} x {elu,dpfp}
for s in 0 1 2; do
  for ret in exp cubic_flux_chunked; do
    for fm in elu dpfp; do
      run_until_done "clh $ret/$fm lr1e-3 s$s" $PY review_scripts/cubic_longhorizon.py $ret $fm 1e-3 $s $BUDGET
    done
  done
done
# cubic LR-dayanikliligi (duyarli mod), seed 0, iki fmap, iki ek LR (INCELEME uyarisi)
for lr in 3e-4 3e-3; do
  for fm in elu dpfp; do
    run_until_done "clh cubic/$fm lr$lr s0" $PY review_scripts/cubic_longhorizon.py cubic_flux_chunked $fm $lr 0 $BUDGET
  done
done

echo "=== TUM ACIK HUCRELER TAMAM. Sonuclar: $HFP_CKPT_DIR/*.txt ==="
echo "Faz 1b karar: cubic_lh_results.txt -> cubic_flux_chunked+dpfp, 256+ kovasinda"
echo "  exp+dpfp'yi 3 seed'de >2 SE geciyor mu? (onceden yazilan basari kriteri)"
echo "Sonrasi: DENEY_SONUCLARI.md + RESULTS.md + osf_companion.tex'i bu sayilarla guncelleyin."
