# HFP Projesi Teknik Denetim Raporu (Ağustos 2026)

**Tarih:** 2026-08-02  
**Denetçi:** Ajan 3 (Teknik Denetçi)  
**Kapsam:** Codebase Mekanik, İddia/Bilim ve Tasarım İncelemesi  
**Lisans & Proje:** AGPL-3.0, Kayrahan Yılmaz  

---

## 1. Yönetici Özeti ve Denetim Özellikleri

- **Toplam Bulgu Sayısı:** 10
- **Sınıf Dağılımı:**
  - **Sınıf 1 — Mekanik (Uygulandı):** 5 bulgu
  - **Sınıf 2 — İddia / Bilim (Raporlandı, dokunulmadı):** 3 bulgu
  - **Sınıf 3 — Tasarım (Önerildi, dokunulmadı):** 2 bulgu
- **Hiç Bakılmayan / İnceleme Dışı Tutulan Alanlar:**
  - `_legacy_reference/` iç kod mantığı (yalnızca import edilmediği doğrulandı)
  - `notebooks/*.ipynb` Colab/Kaggle çalışma zamanı ve GPU execution çıktıları
  - `docs/*.tex` TeX derleme zinciri
  - `checkpoints/` ikili (binary) model ağırlık içerikleri
  - GPU gerektiren uzun süreli eğitim koşuları (kota §36 için ayrılmıştır)

---

## 2. Bulgular Tablosu

| # | Sınıf | Dosya:Satır | Bulgu | Kanıt | Eylem |
|---|---|---|---|---|---|
| 1 | Sınıf 1 | `hf_upload/hfp_bulk_state.py:1-427` & `hf_upload/hfp_config.py:1-49` | `hf_upload/` paket dosyaları kanonik `hfp/core/` kodundan sapmış (BEKLEYEN #9). | `git diff --no-index --stat hfp/core/hfp_bulk_state.py hf_upload/hfp_bulk_state.py` (210 satır fark). | `hfp/core/` dosyaları `hf_upload/` klasörüne kopyalanarak kanonik duruma eşitlendi. |
| 2 | Sınıf 1 | `.gitignore:45` & `tinyshakespeare.txt` | `tinyshakespeare.txt` `.gitignore`'da olmasına rağmen git index'inde takip ediliyordu (BEKLEYEN #10). | `git ls-files tinyshakespeare.txt` -> `tinyshakespeare.txt` | `git rm --cached tinyshakespeare.txt` çalıştırıldı (Commit: `02f7149`). |
| 3 | Sınıf 1 | `pyproject.toml:21-25` | `pyproject.toml` bağımlılık listesi `matplotlib` ve `huggingface_hub` paketlerini içermiyordu. | `pyproject.toml` vs `requirements.txt` farkı. | `dependencies` listesine `matplotlib>=3.7` ve `huggingface_hub>=0.23` eklendi (Commit: `6e7e699`). |
| 4 | Sınıf 1 | `.github/workflows/ci.yml:31` | CI `requirements.txt` yerine eksik paket kuruyordu; `train.py` ve `run_experiment.py` CI'da hiç çalıştırılmıyordu. | `.github/workflows/ci.yml` dosya incelemesi (`pip install "transformers>=4.40" numpy`). | CI bağımlılıkları `requirements.txt`'e çekildi ve `train.py` + `run_experiment.py` kuru-koşu adımları eklendi (Commit: `22988c5`). |
| 5 | Sınıf 1 | `hfp/models/modeling_hfp.py:82-84` | `HFPModel` sınıfının `past_key_values` 7-elemanlı tuple yapısı belgelenmemişti. | `HFPModel._offset_from_state` `past_key_values_list[0][3]` erişimi. | `HFPModel` sınıfına 7 elemanlı tuple yapısını açıklayan docstring eklendi (Commit: `c1256fb`). |
| 6 | Sınıf 2 | `README.md:134-138` (ve `RESULTS.md:§14`) | GLA karşılaştırmasının etiket kaydırma hatası nedeniyle revizyonda olduğu dürüstçe belirtilmektedir. | `README.md:134`: "A metric artifact (double-shifted labels; RESULTS §14)..." | **DESTEKLİ / RAPORLANDI:** `RESULTS.md` otorite kaydıdır, düzenlenmedi. Dürüstlük uyarısı korundu. |
| 7 | Sınıf 2 | `README.md:120-121` | "cubic_flux_chunked paired with DPFP outperforms exponential baseline" iddiası seyrek rejimde geçerlidir. | `RESULTS.md:§28a` (n=16 seed, p=0.0124 ön-kayıtlı test). | **DESTEKLİ / RAPORLANDI:** İddia RESULTS §28a ile tam uyumludur. Metin aynen korundu. |
| 8 | Sınıf 2 | `review_scripts/matched_probe.py:65` vs `review_scripts/memory_probe_cpu.py:61` | 2. ajanın scriptlerinde `MP_` ortam değişkeni ön eki çakışmaktadır (`MP_TRIALS` 8 vs 30). | `matched_probe.py:66` vs `memory_probe_cpu.py:61`. | **RAPORLANDI:** `review_scripts/` 2. ajanın sahipliğindedir. 2. ajanın `MP_` ön eklerini ayrıştırması önerildi. |
| 9 | Sınıf 3 | `eval_passkey.py:76-78` & `train.py:57-59` | State sıfırlama için scriptler `model.hfp.bulk_states` iç yapısına erişmektedir. | `for b_state in model.hfp.bulk_states: b_state.reset_state()` | **ÖNERİLDİ:** `HFPForCausalLM` seviyesinde birleşik bir `reset_states()` metodu eklenmesi önerildi, uygulanmadı. |
| 10 | Sınıf 3 | `train.py:135-144` vs `run_experiment.py:321` | `train.py` ve `run_experiment.py` varsayılan decay ve feature_map değerlerinde farklılaşmaktadır. | `train.py` line 139 vs `run_experiment.py` line 321. | **ÖNERİLDİ:** `train.py` için `--decay_mode` CLI bayrağı eklenmesi önerildi, uygulanmadı. |

---

## 3. Doğrulama ve Mühürleme

Tüm Sınıf 1 düzeltmeleri sonrasında aşağıdaki doğrulama komutları koşturulmuş ve yeşil geçmiştir:

1. `python smoke_test.py` -> **ALL SMOKE TESTS PASSED — GATE APPROVED**
2. `python review_scripts/verify_claims.py` -> **TOPLAM: 17/17 PASS**

---
*Rapor Sonu.*
