# HFP Projesi Tekrarlanabilirlik Denetim Raporu (Ağustos 2026 — 4. Tur Güncellenmiş)

**Tarih:** 2026-08-02  
**Denetçi:** Antigravity (Teknik Denetçi)  
**Kapsam:** `RESULTS.md` §1 - §36 Bölümleri Tekrarlanabilirlik ve Çalıştırılabilirlik Yolları (`reproduce.py` Otomasyonu)  

---

## 1. Genel Durum Özeti

`RESULTS.md` belgesindeki 36 bölümün yeniden üretim yolları incelenmiş, yerel CPU ortamında `reproduce.py` otomasyon script'i ile ampirik olarak çalıştırılarak doğrulanmıştır:

- **KOŞULABİLİR / BAŞARILI (13 Bölüm):** `reproduce.py` tarafından kısa parametrelerle koşturulmuş ve sıfır hata kodu (`exit code 0`) ile geçtiği doğrulanmıştır (§1, §2, §3, §5, §7, §11, §13, §14, §16, §19, §21, §28, §29).
- **KOŞULABİLİR / ATLANDI-ÖNKOŞUL (1 Bölüm):** §4 (`interference_eval.py`), §3 eğitimi çıktısı olan `checkpoints/lg_0_final.pt` bağımlılığına sahiptir. Önkoşul checkpoint'i bulunmadığında `ATLANDI-ONKOSUL` olarak işaretlenir; kırık kod değildir.
- **KOŞULABİLİR / AĞIR CPU (3 Bölüm):** §6, §26, §27 bölümleri CPU üzerinde 120sn+ zamanaşımına uğramaktadır; `reproduce.py --heavy` bayrağı ile çalıştırılabilir.
- **KOŞULAMAZ-CHECKPOINT (8 Bölüm):** Önceden eğitilmiş model ağırlık dosyalarını (`.pt` / `checkpoints/`) gerektirir (`.gitignore` kuralları gereği depoda yer almaz).
- **KOŞULAMAZ-GPU (7 Bölüm):** Qwen2.5-1.5B tam distilasyon veya VRAM/gecikme ölçümü GPU (T4/P100) ortamı gerektirir.
- **KOŞULAMAZ-VERİ (1 Bölüm):** WikiText-2 harici veri setini gerektirir.
- **KOMUT-YOK (3 Bölüm):** Tarihsel sentez, teorik çerçeve veya geri çekilmiş kararlardır.

---

## 2. Bölüm Bazlı Tekrarlanabilirlik Tablosu

| § | Başlık | Üretim Yolu (`reproduce.py`) | Durum | Eksik Olan / Not |
|---|---|---|---|---|
| 1 | Methodological finding: supervision density gates learnability | `python review_scripts/dense_retention.py exp additive 1e-3 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 8.04s). |
| 2 | Retention-law and write-rule comparison (3 seeds, ctx 160) | `python review_scripts/dense_retention.py cubic_flux additive 1e-3 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 8.33s). |
| 3 | Length generalization (3 seeds) — main positive result | `python review_scripts/length_gen.py train 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 7.42s). |
| 4 | The memory is interference-limited, not decay-limited | `python review_scripts/interference_eval.py 0` | ATLANDI-ÖNKOŞUL | `checkpoints/lg_0_final.pt` (§3 eğitimi) gerektirir. |
| 5 | Capacity axis (DPFP feature map) — first clear mechanism win | `LG_VARIANT=dpfp python review_scripts/length_gen.py train 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 7.50s). |
| 6 | cubic_flux long-horizon advantage (Validated) | `python run_experiment.py --task retention --steps 10 ...` | AĞIR-CPU | CPU'da 120s+ (zaman aşımı). `--heavy` ile koşulabilir. |
| 7 | Initial Language Modeling Viability | `python run_experiment.py --task lm --steps 10 --seq 128 ...` | KOŞULABİLİR (PASS) | Yok (CPU'da 71.68s). |
| 8 | Current recipe | Reçete ve hiperparametre sentezi | KOMUT-YOK | Dokümantasyon özetidir. |
| 9 | Parked / negative results (honest ledger) | Dürüst olumsuz sonuçlar kaydı | KOMUT-YOK | Tarihsel dökümdür. |
| 10 | Language Modeling Validation (WikiText-2) | `notebooks/colab_lm_ablation.ipynb` | KOŞULAMAZ-VERİ | WikiText-2 harici verisi gerektirir. |
| 11 | Training-length cliff applies to LM as well | `python review_scripts/length_gen.py train 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 6.83s). |
| 12 | External family baseline: GLA (K1 decision — WITHDRAWN) | `python review_scripts/baseline_compare.py` | KOMUT-YOK | Metrik hatası nedeniyle geri çekildi. |
| 13 | Write-rule decision at long evaluation lengths | `HR_WRITE=additive python review_scripts/hard_retention.py exp 1e-3 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 19.87s). |
| 14 | Metric-artifact disclosure and length-degradation diagnosis | `HR_STEPS=10 python review_scripts/hard_retention.py exp 1e-3 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 23.25s). |
| 15 | Qwen2.5-1.5B graft, full 2-stage distillation run | `notebooks/colab_graft_qwen_v3_kaggle.ipynb` | KOŞULAMAZ-GPU | Qwen2.5-1.5B ve GPU gerektirir. |
| 16 | K1 gate, clean re-run: GLA family baseline v2 | `python review_scripts/baseline_compare.py 0 1.0` | KOŞULABİLİR (PASS) | `reproduce.py` CLI argüman sırası düzeltildi (CPU'da 3.96s). |
| 17 | Görev C — lifetime retention (cubic's natural-habitat test) | `LT_STEPS=10 python review_scripts/lifetime_retention.py exp 0 1.0` | KOŞULABİLİR (PASS) | `set_rng_state(st["trng"].cpu())` EOL/RNG fix uygulandı. |
| 18 | Görev D — write-sparsity sweep | `notebooks/kaggle_write_sparsity_sweep.ipynb` | KOŞULAMAZ-GPU | Kaggle GPU ortamı gerektirir. |
| 19 | Görev E — carry curriculum | `CC_STEPS=10 python review_scripts/carry_curriculum.py exp 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 7.79s). |
| 20 | Görev F — matched probe: failure is FIRST chunk boundary | `python review_scripts/matched_probe.py exp 0` | KOŞULAMAZ-CHECKPOINT | `carryv1_exp_s0.pt` checkpoint'i gerektirir. |
| 21 | Görev G — TBPTT intervention | `CC_BPTT=1 CC_STEPS=10 python review_scripts/carry_curriculum.py exp 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 8.27s). |
| 22 | PRE-REGISTERED: graft density (Run 7) & multi-seed | `notebooks/kaggle_graft_diagnostics_v1.ipynb` | KOŞULAMAZ-GPU | GPU gerektirir. |
| 23 | On-device VRAM / latency showcase | `notebooks/bench_vram_latency_v14.ipynb` | KOŞULAMAZ-GPU | CUDA GPU gerektirir. |
| 24 | Per-layer linearization-cost map (Faz-A) | `notebooks/layer_linearization_probe_v1.ipynb` | KOŞULAMAZ-GPU | GPU gerektirir. |
| 25 | Root-cause fix attempt — student-forward distillation | `notebooks/colab_graft_qwen_v3_kaggle.ipynb` | KOŞULAMAZ-GPU | GPU gerektirir. |
| 26 | Cubic in its native regime — extrapolation test | `python review_scripts/cubic_stabilize.py exp 0 1.0` | AĞIR-CPU | CPU'da 120s+ (zaman aşımı). `--heavy` ile koşulabilir. |
| 27 | Improving cubic: the η sweep | `HFP_ETA_LOG_MIN=-6.0 python review_scripts/cubic_stabilize.py cubic_flux_chunked 0 1.0` | AĞIR-CPU | CPU'da 120s+ (zaman aşımı). `--heavy` ile koşulabilir. |
| 28 | Cubic, decisive test | `python review_scripts/cubic_longhorizon.py exp elu 1e-3 0 1.0` | KOŞULABİLİR (PASS) | Yok (CPU'da 7.97s). |
| 29 | parallel_cubic — removing the sequential z-scan | `python review_scripts/verify_claims.py` | KOŞULABİLİR (PASS) | Yok (CPU'da 7.99s). |
| 30 | Memory-organ capability test | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | Graft checkpoint'i ve GPU gerektirir. |
| 31 | Write-path gradient across chunk boundaries | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | Checkpoint ve GPU gerektirir. |
| 32 | Delta write rule for cross-boundary storage | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | Checkpoint ve GPU gerektirir. |
| 33 | Masked recall supervision | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | Checkpoint ve GPU gerektirir. |
| 34 | Trainable projections for the memory path | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | Checkpoint ve GPU gerektirir. |
| 35 | Chaining across boundaries: capacity × write rule | `notebooks/parallel_cubic_v29.ipynb` | KOŞULAMAZ-GPU | Kaggle GPU sweep ortamı gerektirir. |
| 36 | Retention law at LM scale, with cache confound removed | `notebooks/kaggle_graft_diagnostics_v1.ipynb` | KOŞULAMAZ-CHECKPOINT | Graft checkpoint'leri gerektirir. |

---
*Rapor Sonu.*
