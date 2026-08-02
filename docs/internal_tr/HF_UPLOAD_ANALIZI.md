# HFP HuggingFace Yayın Paketi ve Üç Kopya Analizi (HF_UPLOAD_ANALIZI.md)

**Tarih:** 2026-08-02  
**Yazar:** Antigravity (Teknik Denetçi)  
**Kapsam:** `hfp/core/` & `hfp/models/` (kanonik), `hf_upload/` (kök takipsiz), `hf_upload/hf_release/` (2026-07-05 takipsiz arşiv)  
**Lisans:** AGPL-3.0  

---

## 1. Durum Özeti

HFP projesinde HuggingFace model yayınlama kodları üç farklı dizinde kopyalar halinde bulunmaktadır:
1. `hfp/core/` ve `hfp/models/` — **Kanonik Kod Tabanı** (takipli, güncel)
2. `hf_upload/` — Kök dizindeki takipsiz (`.gitignore:58`) yayın çalışma alanı
3. `hf_upload/hf_release/` — 2026-07-05 tarihli eski yayın arşivi (takipsiz)

Elle senkronizasyon denemeleri (1. ve 2. turlar) kopyalar arasında yeni uyumsuzluklar yaratmış ve sürüklenmeyi (code drift) engellemeye yetmemiştir. Bu raporda tüm kopyalar arasındaki farklar satır satır incelenmiş, farkların doğası (**SÜRÜKLENME** vs **KASITLI UYARLAMA**) analiz edilmiştir.

---

## 2. Dosya Bazlı Fark ve Sınıflandırma Tablosu

| Dosya | Kanonik Konum | `hf_release` Karşılaştırması (`A ↔ B`) | Sınıflandırma | Gerekçe / Detay |
|---|---|---|---|---|
| `bulk_trigger_decoder.py` | `hfp/core/` | `hf_release ↔ hfp/core: 2 satır` | **SÜRÜKLENME** | Kanonik kodda kullanılmayan `compute_entropy_map` import'u temizlenmiştir. HF'ye özel hiçbir uyarlama içermez. |
| `hfp_utils.py` | `hfp/core/` | `hf_release ↔ hfp/core: 0 satır` | **SÜRÜKLENME / AYNI** | Dosya kanonik sürüm ile bit-bit birebir aynıdır. |
| `hfp_bulk_state.py` | `hfp/core/` | `hf_release ↔ hfp/core: 210 satır` | **SÜRÜKLENME** | Kanonik `hfp/core/hfp_bulk_state.py` geliştirilmiş; `@torch.jit.script _cubic_zscan`, `parallel_cubic` modu, `write_rule == "delta"` mantığı ve `.clamp_max(0.0)` üstel taşma koruması eklenmiştir. `hf_release` kopyası geride kalmıştır. |
| `hfp_config.py` | `hfp/core/` | `hf_release ↔ hfp/core: 15 satır` | **SÜRÜKLENME** | Kanonik `hfp/core/hfp_config.py` içine §27 `ETA_LOG_MIN` (-4.0), `ETA_LOG_MAX` (-2.0) ve `HFP_ETA_LOG_*` ortam değişkeni override mantığı eklenmiştir. `hf_release` kopyası geride kalmıştır. |
| `configuration_hfp.py` | `hfp/models/` | `hf_release ↔ hfp/models: 0 satır` | **SÜRÜKLENME / AYNI** | Dosya kanonik `hfp/models/configuration_hfp.py` ile bit-bit aynıdır. |
| `modeling_hfp.py` | `hfp/models/` | `hf_release ↔ hfp/models: 22 satır` | **SÜRÜKLENME** | Kanonik `hfp/models/modeling_hfp.py` içine 1. turda docstring ve sınıf tanımı düzenlemesi gelmiştir (`c1256fb`). `hf_release` kopyası geride kalmıştır. |

---

## 3. Bulgular ve Değerlendirme

- **KASITLI UYARLAMA Tespiti:** Yapılan detaylı kod diff incelemesinde `hf_upload/` veya `hf_release/` içindeki hiçbir PyTorch modülünde HF'ye özel kasıtlı bir kod/import uyarlamasına (**KASITLI UYARLAMA**) **rastlanmamıştır**.
- Kanonik `hfp/core/` ve `hfp/models/` kodları bağımsız göreli (relative) importlar (`from .hfp_config import config`) kullandığı için, hem `hfp` paketi altında hem de tekil bir HF reposu dizininde sıfır modifikasyonla çalışmaktadır.
- Tespit edilen %100 fark **SÜRÜKLENME** (kanonik kodun ilerlemesi, kopyaların durması) kaynaklıdır.

---

## 4. Karar ve Çözüm Planı

1. **Otomatize Yayın Script'i (`scripts/build_hf_release.py`):**
   Yayın paketini kanonik `hfp/core/` ve `hfp/models/` kaynaklarından programmatic olarak toplayan, doğrulayan (byte-compile / import check) takipli bir build script'i yazılmıştır (`scripts/build_hf_release.py`).
2. **Kopyaların Silinmesi Önerisi:**
   Build script'i sayesinde yayın paketi anlık olarak kanonik kottan üretilebildiği için, takipsiz `hf_upload/` ve `hf_upload/hf_release/` kopyalarının silinmesi önerilmektedir. (Bu kopyalar takipsiz dizinde yer aldığı için silme işlemi kullanıcının onayına bırakılmıştır; ajan tarafından onay alınmadan silinmemiştir).
