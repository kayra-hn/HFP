# HFP Projesi Tekrarlanabilirlik Denetim Raporu (Ağustos 2026)

**Tarih:** 2026-08-02  
**Denetçi:** Ajan 3 (Teknik Denetçi)  
**Kapsam:** `RESULTS.md` §1 - §36 Bölümleri Tekrarlanabilirlik ve Çalıştırılabilirlik Yolları  

---

## 1. Genel Durum Özetı

`RESULTS.md` belgesindeki 36 bölümün yeniden üretim yolları incelenmiş, yerel CPU ortamında bağımsız betikler koşturularak doğrulanmıştır:

- **KOŞULABİLİR (17 Bölüm):** Bağımsız Python betikleri (`review_scripts/` ve `run_experiment.py`) ile yerel CPU ortamında harici veri/GPU gereksinimi olmadan veya mevcut yerel verilerle çalıştırılabilir.
- **KOŞULAMAZ-CHECKPOINT (8 Bölüm):** Önceden eğitilmiş model ağırlık dosyalarını (`.pt` / `checkpoints/`) gerektirir. `.gitignore:7,33` kuralları gereği ağırlıklar git deposunda yer almadığı için checkpoint yeniden eğitilmeden doğrudan çalıştırılamaz.
- **KOŞULAMAZ-GPU (7 Bölüm):** Qwen2.5-1.5B tam distilasyon, VRAM/gecikme ölçümü veya Kaggle/Colab GPU (T4/P100) altyapısı gerektirir.
- **KOŞULAMAZ-VERİ (1 Bölüm):** WikiText-2 harici veri setini gerektirir.
- **KOMUT-YOK (3 Bölüm):** Tarihsel sentez, teorik çerçeve veya geri çekilmiş/süpersede edilmiş kararlardır; tekil çalıştırma komutu yoktur.

---

## 2. Bölüm Bazlı Tekrarlanabilirlik Tablosu

| § | Başlık | Üretim Yolu | Durum | Eksik Olan / Not |
|---|---|---|---|---|
| 1 | Methodological finding: supervision density gates learnability | `python review_scripts/dense_retention.py exp additive 1e-3 0` | KOŞULABİLİR | Yok (CPU'da doğrulanmıştır). |
| 2 | Retention-law and write-rule comparison (3 seeds, ctx 160) | `python review_scripts/dense_retention.py {exp\|cubic_flux} {additive\|delta} 1e-3 {0\|1\|2}` | KOŞULABİLİR | Yok. |
| 3 | Length generalization (3 seeds) — main positive result | `python review_scripts/length_gen.py train 0` && `python review_scripts/length_gen.py eval 0` | KOŞULABİLİR | Yok. |
| 4 | The memory is interference-limited, not decay-limited | `python review_scripts/interference_eval.py 0` | KOŞULABİLİR | Yok. |
| 5 | Capacity axis (DPFP feature map) — first clear mechanism win | `LG_VARIANT=dpfp python review_scripts/length_gen.py train 0` | KOŞULABİLİR | Yok. |
| 6 | cubic_flux long-horizon advantage (Validated) | `python run_experiment.py --task retention --steps 1500 --context 96 --max_gap 64 --local_window 16 --decay_mode cubic_flux` | KOŞULABİLİR | Yok. |
| 7 | Initial Language Modeling Viability | `python run_experiment.py --task lm --steps 1500 --seq 128 --decay_mode cubic_flux` | KOŞULABİLİR | `tinyshakespeare.txt` yerelde mevcut (takipsiz). |
| 8 | Current recipe | Reçete ve hiperparametre sentezi | KOMUT-YOK | Dokümantasyon özetidir; tekil betik yok. |
| 9 | Parked / negative results (honest ledger) | Dürüst olumsuz sonuçlar kaydı | KOMUT-YOK | Tarihsel olumsuz sonuçlar dökümüdür. |
| 10 | Language Modeling Validation (WikiText-2) | `notebooks/colab_lm_ablation.ipynb` | KOŞULAMAZ-VERİ | WikiText-2 harici verisi ve Colab GPU ortamı gerektirir. |
| 11 | Training-length cliff applies to LM as well (3 seeds, negative result) | `python review_scripts/length_gen.py train {0\|1\|2}` | KOŞULABİLİR | Yok. |
| 12 | External family baseline: GLA (K1 decision — WITHDRAWN) | `python review_scripts/baseline_compare.py` | KOMUT-YOK | Metrik hatası nedeniyle geri çekildi (§16 ile yenilendi). |
| 13 | Write-rule decision at long evaluation lengths (K2 — recipe locked) | `HR_WRITE=additive python review_scripts/hard_retention.py exp 0` | KOŞULABİLİR | Yok. |
| 14 | Metric-artifact disclosure and length-degradation diagnosis | `python review_scripts/hard_retention.py exp 0` | KOŞULABİLİR | Yok. |
| 15 | Qwen2.5-1.5B graft, full 2-stage distillation run (§15a-§15h) | `notebooks/colab_graft_qwen_v3_kaggle.ipynb` | KOŞULAMAZ-GPU | Qwen2.5-1.5B ağırlıkları ve GPU (T4) gerektirir. |
| 16 | K1 gate, clean re-run: GLA family baseline v2 (3 seeds) | `python review_scripts/baseline_compare.py exp 0` | KOŞULABİLİR | Yok. |
| 17 | Görev C — lifetime retention (cubic's natural-habitat test) | `python review_scripts/lifetime_retention.py exp 0` | KOŞULABİLİR | Yok. |
| 18 | Görev D — write-sparsity sweep: capacity account REJECTED | `notebooks/kaggle_write_sparsity_sweep.ipynb` | KOŞULAMAZ-GPU | Kaggle GPU sweep ortamı gerektirir. |
| 19 | Görev E — carry curriculum: rejected too, train/eval mismatch | `python review_scripts/carry_curriculum.py exp 0` | KOŞULABİLİR | `checkpoints/carryv1_exp_s0.pt` üretir. |
| 20 | Görev F — matched probe: failure is FIRST chunk boundary | `python review_scripts/matched_probe.py exp 0` | KOŞULAMAZ-CHECKPOINT | Önce §19 koşularak `checkpoints/carryv1_exp_s0.pt` üretilmelidir. |
| 21 | Görev G — TBPTT intervention: accuracy unmoved | `CC_BPTT=1 python review_scripts/carry_curriculum.py exp 0` | KOŞULABİLİR | Yok. |
| 22 | PRE-REGISTERED: graft density (Run 7) & multi-seed (Runs 8-9) | `notebooks/kaggle_graft_diagnostics_v1.ipynb` | KOŞULAMAZ-GPU | GPU + Qwen2.5-1.5B ağırlıkları gerektirir. |
| 23 | On-device VRAM / latency showcase (hybrid vs full KV-cache) | `notebooks/bench_vram_latency_v14.ipynb` | KOŞULAMAZ-GPU | CUDA GPU ve VRAM profilleme araçları gerektirir. |
| 24 | Per-layer linearization-cost map (Faz-A) | `notebooks/layer_linearization_probe_v1.ipynb` | KOŞULAMAZ-GPU | GPU + Qwen2.5-1.5B ağırlıkları gerektirir. |
| 25 | Root-cause fix attempt — student-forward distillation | `notebooks/colab_graft_qwen_v3_kaggle.ipynb` | KOŞULAMAZ-GPU | GPU + Qwen2.5-1.5B ağırlıkları gerektirir. |
| 26 | Cubic in its native regime — extrapolation test (pre-registered) | `python review_scripts/cubic_stabilize.py exp 0` | KOŞULABİLİR | Yok. |
| 27 | Improving cubic: the η (plateau-scale) sweep | `HFP_ETA_LOG_MIN=-6.0 python review_scripts/cubic_stabilize.py cubic_flux_chunked 0` | KOŞULABİLİR | Yok. |
| 28 | Cubic, decisive test (pre-registered before the run) | `python review_scripts/cubic_longhorizon.py exp 0` | KOŞULABİLİR | Yok. |
| 29 | parallel_cubic — removing the sequential z-scan | `python review_scripts/verify_claims.py` | KOŞULABİLİR | Yok. |
| 30 | Memory-organ capability test (pre-registered) | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | `hfp_graft_exp_g6*_final.pt` (6-katman graft) `.pt` dosyasını ve GPU gerektirir. |
| 31 | Write-path gradient across chunk boundaries (pre-registered) | `notebooks/memory_capability_v30.ipynb` (filter: `g6W_s0`) | KOŞULAMAZ-CHECKPOINT | `g6W_s0` checkpoint'i ve GPU gerektirir. |
| 32 | Delta write rule for cross-boundary storage (pre-registered) | `notebooks/memory_capability_v30.ipynb` (filter: `g6WD_s0`) | KOŞULAMAZ-CHECKPOINT | `g6WD_s0` checkpoint'i ve GPU gerektirir. |
| 33 | Masked recall supervision (pre-registered) | `notebooks/memory_capability_v30.ipynb` (filter: `g6_s1`) | KOŞULAMAZ-CHECKPOINT | `g6_s1` checkpoint'i ve GPU gerektirir. |
| 34 | Trainable projections for the memory path (pre-registered) | `notebooks/memory_capability_v30.ipynb` | KOŞULAMAZ-CHECKPOINT | `own_proj=True` checkpoint'i ve GPU gerektirir. |
| 35 | Chaining across boundaries: capacity × write rule (pre-registered) | `notebooks/parallel_cubic_v29.ipynb` | KOŞULAMAZ-GPU | Kaggle GPU sweep ortamı gerektirir. |
| 36 | Retention law at LM scale, with cache confound removed | `notebooks/kaggle_graft_diagnostics_v1.ipynb` | KOSHULAMAZ-CHECKPOINT | `checkpoints/graft_run5/hfp_graft_final.pt` ve `checkpoints/graft_run6_exp/hfp_graft_exp_final.pt` ağırlıklarını gerektirir. |

---
*Rapor Sonu.*
