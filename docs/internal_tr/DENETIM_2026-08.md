# HFP Projesi Teknik Denetim Raporu (Ağustos 2026 — 2. Tur Güncellenmiş)

**Tarih:** 2026-08-02  
**Denetçi:** Ajan 3 (Teknik Denetçi)  
**Kapsam:** Codebase Mekanik, İddia/Bilim, Tasarım İncelemesi, `grafting.py` Test Suite'i ve Tekrarlanabilirlik Denetimi  
**Lisans & Proje:** AGPL-3.0, Kayrahan Yılmaz  

---

## 1. Yönetici Özeti ve Denetim Özellikleri

- **Toplam Bulgu Sayısı:** 12
- **Sınıf Dağılımı:**
  - **Sınıf 1 — Mekanik (Uygulandı):** 6 bulgu
  - **Sınıf 2 — İddia / Bilim (Raporlandı, dokunulmadı):** 3 bulgu
  - **Sınıf 3 — Tasarım (Önerildi / Analiz Edildi, dokunulmadı):** 3 bulgu
- **Hiç Bakılmayan / İnceleme Dışı Tutulan Alanlar:**
  - `_legacy_reference/` iç kod mantığı (yalnızca import edilmediği doğrulandı)
  - `notebooks/*.ipynb` Colab/Kaggle canlı GPU çalışma zamanı çıktısı
  - `docs/*.tex` TeX derleme zinciri
  - `checkpoints/` ikili (binary) model ağırlık içerikleri
  - GPU gerektiren uzun süreli eğitim koşuları (kota §36 için ayrılmıştır)

---

## 2. Bulgular Tablosu

| # | Sınıf | Dosya:Satır | Bulgu | Kanıt | Eylem & Doğrulama Kapsamı |
|---|---|---|---|---|---|
| 1 | Sınıf 3 | `hf_upload/hfp_bulk_state.py` & `hfp_config.py` | `hf_upload/` takipsiz (.gitignore:58). 1. turdaki kopyalama işlemi takipsiz dizinde yapıldı (BEKLEYEN #9). | `git status` + `.gitignore:58` | **YENİDEN İNŞA EDİLDİ (1a):** `hf_release/` (2026-07-05) ile karşılaştırıldı (+177/-33 satır diff: `_cubic_zscan`, `parallel_cubic`, `write_rule=="delta"`, `clamp_max(0.0)` ve `HFP_ETA_LOG_*` farkı). *Dosyalara dokunulmadı.* |
| 2 | Sınıf 3 | `hf_upload/` & `hf_release/` | Yayın paketi sapma analizi (BEKLEYEN #9). 2. turdaki 1 ve 0 sayılarının `hf_upload ↔ hf_release` arası kopyalar arası fark olduğu, kanonik `hfp/core/` ile sapmanın sürdüğü tespit edildi. | `diff` matrisi analizi | **ERRATUM EKLENDİ / OTOMATİZE EDİLDİ (1b, Görev 2):** Erratum nota geçirildi. Sapmanın %100 SÜRÜKLENME olduğu anlaşıldı. `scripts/build_hf_release.py` yazıldı. |
| 3 | Sınıf 1 | `.gitignore:45` & `tinyshakespeare.txt` | `tinyshakespeare.txt` `.gitignore`'da olmasına rağmen git index'inde takip ediliyordu (BEKLEYEN #10). | `git ls-files tinyshakespeare.txt` -> `tinyshakespeare.txt` | `git rm --cached tinyshakespeare.txt` çalıştırıldı (Commit: `02f7149`). *Kapsam: git status.* |
| 4 | Sınıf 1 | `pyproject.toml:21-25` | `pyproject.toml` bağımlılık listesi `matplotlib` ve `huggingface_hub` paketlerini içermiyordu. | `pyproject.toml` vs `requirements.txt` | `dependencies` listesine `matplotlib>=3.7` ve `huggingface_hub>=0.23` eklendi (Commit: `6e7e699`). *Kapsam: setup/build.* |
| 5 | Sınıf 1 | `.github/workflows/ci.yml` | CI `requirements.txt` yerine eksik paket kuruyordu; `train.py` verisetisiz CI'da patlıyordu. | `train.py:101` FileNotFoundError (Commit `562a703` uyarısı) | CI `requirements.txt`'e çekildi. `train.py` bağımlılığı nedeniyle kaldırıldı; sentetik `run_experiment --task recall` ve `verify_graft.py` CI'a eklendi (Commit `22988c5`, `1626a2f`). |
| 6 | Sınıf 1 | `hfp/models/modeling_hfp.py:82-84` | `HFPModel` sınıfının `past_key_values` 7-elemanlı tuple yapısı belgelenmemişti. | `HFPModel._offset_from_state` `past_key_values_list[0][3]` erişimi | `HFPModel` sınıfına 7 elemanlı tuple şemasını açıklayan docstring eklendi (Commit: `c1256fb`). *Kapsam: docstring.* |
| 7 | Sınıf 1 | `review_scripts/verify_graft.py` (yeni) | `grafting.py` (519 satır) hiçbir test kapsamında değildi; `reset_streaming` ve modlar korumasızdı. | `smoke_test.py` ve `verify_claims.py`'nin `grafting.py`'yi kapsamaması | **VERIFY_GRAFT.PY YAZILDI (Görev 2):** CPU'da, ağsız (~5 sn) T1-T5 tüm değişmezleri doğrulayan test yazıldı ve CI'a eklendi (Commit: `1626a2f`). *Kapsam: `verify_graft.py`.* |
| 8 | Sınıf 2 | `README.md:134-138` (ve `RESULTS.md:§14`) | GLA karşılaştırmasının etiket kaydırma hatası nedeniyle revizyonda olduğu dürüstçe belirtilmektedir. | `README.md:134`: "double-shifted labels..." | **DESTEKLİ / RAPORLANDI:** `RESULTS.md` otorite kaydıdır, düzenlenmedi. Dürüstlük uyarısı korundu. |
| 9 | Sınıf 2 | `README.md:120-121` | "cubic_flux_chunked paired with DPFP outperforms exponential baseline" iddiası seyrek rejimde geçerlidir. | `RESULTS.md:§28a` (n=16 seed, p=0.0124 ön-kayıtlı test) | **DESTEKLİ / RAPORLANDI:** İddia RESULTS §28a ile tam uyumludur. Metin aynen korundu. |
| 10 | Sınıf 2 | `review_scripts/matched_probe.py:65` vs `memory_probe_cpu.py:61` | 2. ajanın scriptlerinde `MP_` ortam değişkeni ön eki çakışmaktadır (`MP_TRIALS` 8 vs 30). | `matched_probe.py:66` vs `memory_probe_cpu.py:61` | **RAPORLANDI:** `review_scripts/` 2. ajanın sahipliğindedir. 2. ajanın `MP_` ön eklerini ayrıştırması önerildi. |
| 11 | Sınıf 3 | `eval_passkey.py:76-78` & `train.py:57-59` | State sıfırlama için scriptler `model.hfp.bulk_states` iç yapısına erişmektedir. | `for b_state in model.hfp.bulk_states: b_state.reset_state()` | **ÖNERİLDİ:** `HFPForCausalLM` seviyesinde birleşik bir `reset_states()` metodu eklenmesi önerildi, uygulanmadı. |
| 12 | Sınıf 3 | `train.py:135-144` vs `run_experiment.py:321` | `train.py` ve `run_experiment.py` varsayılan decay ve feature_map değerlerinde farklılaşmaktadır. | `train.py` line 139 vs `run_experiment.py` line 321 | **ÖNERİLDİ:** `train.py` için `--decay_mode` CLI bayrağı eklenmesi önerildi, uygulanmadı. |

---

## 3. Görev 1a & 1b — `hf_upload/` Durum ve Sapma Analizi

- **1a. `hf_upload/` Rejeksiyon Değişiklikleri (`hf_release` 2026-07-05 Arşivi ile Karşılaştırma):**
  - `hfp_bulk_state.py`: 1. turda `hfp/core/` ile eşitleme sonucu `+177, -33` satır değişmiştir. TorchScript `_cubic_zscan`, `parallel_cubic` modu, `write_rule == "delta"` mantığı, `clamp_max(0.0)` üstel taşma koruması ve `DECAY_LAM_MIN/MAX` çok-ölçekli decay init'i eklenmiştir.
  - `hfp_config.py`: `+15` satır eklenmiştir. `ETA_LOG_MIN` (-4.0) ve `ETA_LOG_MAX` (-2.0) alanları ile `HFP_ETA_LOG_*` ortam değişkeni override mantığı gelmiştir.
  - *Not: `hf_upload/` takipsiz (gitignored) olduğu için bu değişiklikler git kaydı oluşturmamıştır.*

- **1b. Kalan İki Sapmanın Analizi ve Karşılaştırma Matrisi Düzeltmesi (BEKLEYEN #9):**
  - **GERİ ÇEKME KAYDI / RETRACTION NOTE (4. Tur — 2026-08-02, Commit `c0e1795`):** 3. Tur brief'inde iddia edilen "528 ve 232 satırlık sapma var" gerekçesi yanlıştır. 528 ve 232 sayıları içerik farkı değil, Windows (CRLF) vs Linux (LF) satır sonu (EOL) artefaktıdır (`bulk_trigger_decoder.py`: 264 satır × 2 = 528; `hfp_utils.py`: 116 satır × 2 = 232). Satır sonları normalize edildiğinde 2. Turdaki `hfp_utils.py` için 0 satır fark ve `bulk_trigger_decoder.py` için yalnızca 2 satırlık kozmetik import farkı matrisi tamamen DOĞRUDUR. 3. turda yazılan "farklar karıştı" düzeltme notu bu nedenle GERİ ÇEKİLMİŞ ve 2. Tur analizi onaylanmıştır.
  - Satır sonu normalize edilmiş gerçek içerik fark matrisi (`A ↔ B: N satır (satır sonu normalize)` formatında):

```
                     hf_upload ↔ hfp/core   hf_release ↔ hfp/core   hf_upload ↔ hf_release
bulk_trigger_decoder        0 satır                2 satır                 2 satır
hfp_utils                   0 satır                0 satır                 0 satır
hfp_bulk_state              0 satır              210 satır               210 satır
hfp_config                  0 satır               15 satır                15 satır
```
  - **Sonuç:** Kopyalar arası farkların %100'ü SÜRÜKLENME (code drift) olup `scripts/build_hf_release.py` otomasyon script'i ile tamamen çözülmüştür.

---

## 4. Görev 2 — `grafting.py` Test Suite (`review_scripts/verify_graft.py`)

`hfp/models/grafting.py` (519 satır) modülünün doğrulanması için `review_scripts/verify_graft.py` oluşturulmuş ve CI'a eklenmiştir:

- **T1 — Katman Hedefleme ve Dondurma:** `graft_llama` seçilen katmanları `HFPGraftAttention`'a çevirir, kalan katmanlar ve base projeksiyonlar `requires_grad=False` kalır. (GEÇTİ)
- **T2 — `reset_streaming` Durum Temizliği (KRİTİK):**
  - Akış A ➔ Akış B ➔ Çıktı X
  - `reset_streaming()` ➔ Akış B ➔ Çıktı Y
  - Taze Model (sıfır durum) ➔ Akış B ➔ Çıktı Z
  - **Doğrulanan Değişmez:** `Y == Z` (bit-bit aynı; `diff=0.00e+00`), `X != Y` (durum taşıma etkisi; `diff=0.5706`). (GEÇTİ)
- **T3 — `enable_streaming(m, False)`:** Akış kapatıldığında `_stream_state` temizlenir ve `streaming=False` olur. (GEÇTİ)
- **T4 — Graft Modları Davranışı:** `teacher` modu orijinal softmax attention ile bit-bit aynı çıktı üretir (`diff=0.00e+00`). `student` ve `teacher_forcing` modları farklılaşır. (GEÇTİ)
- **T5 — Checkpoint Yükleme Güvenliği:** `load_checkpoint_safe(model, state_dict)` yazılmış; 0 tensör eşleştiğinde `ValueError` fırlattığı doğrulanmıştır. (GEÇTİ)

---

## 5. Doğrulama ve Kombinasyon Mührü

Tüm değişikliklerden sonra üç test grubu birlikte koşturulmuş ve tam yeşil geçmiştir:

1. `python smoke_test.py` ➔ **ALL SMOKE TESTS PASSED — GATE APPROVED**
2. `python review_scripts/verify_claims.py` ➔ **TOPLAM: 17/17 PASS**
3. `python review_scripts/verify_graft.py` ➔ **ALL GRAFT VERIFICATION TESTS PASSED — GATE APPROVED**

---
*Rapor Sonu.*
