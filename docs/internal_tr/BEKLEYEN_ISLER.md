# Bekleyen İşler (2026-07-19 itibarıyla) — unutulmasın listesi

Öncelik sırasıyla; her biri bitince tarih + sonuç referansıyla işaretlenir.

## Şimdi (bu hafta)

1. **[PARK EDİLDİ] Küçük-ölçek ömür hattı (§17-§21):** beş açıklama
   elendi/zayıfladı; güncel şüpheliler okuma-yolu seyreltmesi + decay-gradyan
   tesisatı. Devam edecek kişi için başlangıç adımları RESULTS §21 sonunda.
   (Görev C/D/E/F/G scriptleri ve notebook'ları repoda.)
2. **Commit + push** (bu dosya dahil): §15-15h zinciri, §16/K1, LICENSE paketi,
   Görev C scripti. CI yeşilini doğrula.

## Görev C sonucuna bağlı kararlar

3. **Varsayılan reçete kararı:** Görev C cubic nişini kanıtlamazsa → varsayılan
   `decay_mode='exp'` yapılır (graft + LM reçeteleri), cubic flag olarak kalır;
   README'nin "demonstrated results" listesinden §6/cubic iddiası Görev C
   sonucuna göre güncellenir. Kanıtlarsa → "uzun-ömür/cihaz-içi nişi" olarak
   belgelenir, vizyon belgesine bağlanır.
4. **README/RESULTS iddia güncellemesi:** graft bölümüne §15f-h özetini ekle
   (retrieval protokole ait; yasa-bağımsız), K1-geçti notu (§16).

## Graft hattı — açık teknik işler

5. **PPL 1.6× uçurumu:** graft yoğunluğu deneyi (13→6 katman, tek değişken).
   Kriter taslağı: PPL ≤1.2×'e inerken ızgara ≥%80 kalmalı.
6. **exp'in 2048-erken deliği (0/3) + cubic'in 4096 çukuru:** güvenilirlik
   mühendisliği — müfredat yoğunluğu/karışım oranı taraması.
7. **Eğitilmiş-çift girişimi anomalisi:** "copper mountain" @512 ıskalarken
   temiz kelimeler 9/9 (§15h yan gözlem). Ucuz teşhis: eğitim-kelimeli T5
   mini-ızgarası.
8. **Çok-seed graft replikasyonu:** Run 5/6 protokolü × 3 eğitim seed'i
   (manşet iddia öncesi şart).

## Repo / yayın hijyeni

9. **hf_upload senkron ihlali:** `hfp_bulk_state/bulk_trigger_decoder/
   hfp_config/hfp_utils` kanonikten sapmış (≤960 satır diff). Karşılaştır,
   kanonikle eşitle, HF paketini yeniden doğrula.
10. **`tinyshakespeare.txt` git'ten çıkarma:** `git rm --cached` (dosya yerelde
    kalır; .gitignore zaten kapsıyor).
11. **Kaggle/Colab çıktı arşiv düzeni:** Run 3/4 finalleri arşivde eksik
    (yalnız 1/2/5/6 var) — Kaggle version output'larından tamamla (opsiyonel).
12. **Lisans Kapı-2:** HF'ye ağırlık yüklemeden önce
    `LISANS_KARAR_REHBERI.md` checklist'i (ağırlık lisansı + Qwen atfı).

## Ufukta (karar gerektirir)

13. Makale yeniden konumlandırma: "cross-chunk distilasyonla O(1) graft +
    retention-yasası ablasyonu" ana hikâye; §6/Görev C sonucu eklenince taslak
    (`docs/paper3_ml_architecture.tex`) revize.
14. Cihaz-içi vitrin ölçümü: 16k-128k bağlamda VRAM/latency eğrisi
    (hibrit vs saf KV-cache) — vizyonun somut grafiği.
15. GLA baseline'ının NaN'ı: in-house wrapper'daki kararsızlığın kökü
    (adil kıyas hijyeni; düşük öncelik).
