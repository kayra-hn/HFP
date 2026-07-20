# graft_run1 — Qwen2.5-1.5B graft koşusu checkpoint'leri

Kaynak: `notebooks/colab_graft_qwen_v3_kaggle.ipynb`, Kaggle T4, fp32.
Sonuçlar: `RESULTS.md` §15 (dürüst kayıt; PPL 1.996x KALDI, needle MISS).
Config: cubic_flux_chunked + hybrid + dpfp, rec_block=16, tek-indeksli katmanlar.

| dosya | içerik | not |
|---|---|---|
| hfp_graft_stage1_son.pt | S1 700. adım (final) | S1: seq 1024, WT-103, MSE 0.965→0.116 (2026-07-15) |
| hfp_graft_stage2_250.pt | S2 250. adım | S2: seq 128 (T4 bellek sapması), 2026-07-18 |
| hfp_graft_stage2_500.pt | S2 500. adım | " |
| hfp_graft_final.pt      | S2 600. adım (final) | Validasyon bu ağırlıklarla: PPL 15.88, needle MISS |

Not: Kaggle'dan inen dosyalar `.zip` uzantısıyla gelir; torch checkpoint'i
zaten zip formatındadır, uzantı `.pt` yapılınca doğrudan `torch.load` ile açılır.
Yedek: Google Drive `MyDrive/hfp_graft_ckpt/`.

## graft_run2 / graft_run5 (ek arşiv)

- `graft_run2/hfp_graft_stage1_son.pt` — Run 2 S1 finali (out_gain init 0.1; MSE 0.067 plato; alpha 0.143).
- `graft_run5/hfp_graft_final.pt` — Run 5 finali (mesafe-müfredatlı cross-chunk S2).
  Parmak izi: out_gain ort 0.237, alpha ort 0.145. Sonuçlar: RESULTS §15f
  (temiz-kelime needle @512/@8192/@16384 BULDU, @2048 anomali; PPL 13.04).
  ⚠️ Dosya kimliği dersi: "hfp_graft_final" adlı eski (Run 1, 17:26) zip bir ara
  "run5" etiketiyle dolaştı — checkpoint kimliği her zaman parmak iziyle
  (out_gain ort: Run1≈0.75, Run2+≈0.1-0.3) doğrulanır.
