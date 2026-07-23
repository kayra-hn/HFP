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

5. **[TAMAM — 2026-07-21]** PPL uçurumu ÇÖZÜLDÜ (§22a): 6-katman graft →
   PPL 1.112×, needle 4/4 (2048 dahil). Yeni referans reçete: GRAFT_N=6.
6. **exp'in 2048-erken deliği (0/3) + cubic'in 4096 çukuru:** güvenilirlik
   mühendisliği — müfredat yoğunluğu/karışım oranı taraması.
7. **Eğitilmiş-çift girişimi anomalisi:** "copper mountain" @512 ıskalarken
   temiz kelimeler 9/9 (§15h yan gözlem). Ucuz teşhis: eğitim-kelimeli T5
   mini-ızgarası.
8. **[TAMAM — 2026-07-23]** Çok-seed replikasyon TAMAMLANDI (§22b/§22c):
   3/3 seed geçti, PPL 1.111-1.112×, needle 12/12. Ön-kayıtlı ≥2/3 kriteri
   karşılandı. 6-katman referans reçetesi çok-seed onaylı.

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

## Yoğunluk-itme hattı (RESULTS §23 Pareto'sunu ilerletmek)

16. **[KOŞUYA HAZIR — Faz-A] Katman-başı linearize maliyeti haritası:**
    `notebooks/layer_linearization_probe_v1.ipynb`. Tüm 28 katmanı graft'la,
    Stage-1 teacher_forcing, per-katman NMSE oku (bağımsız, tek koşu). Çıktı:
    ucuz/pahalı katman sıralaması → ilkeli seçim. Ön-kayıtlı H1 (seçim
    manevrası ≥%20?), H2 (ilk/son pahalı mı?). PROXY — Faz-B ile doğrulanır.
17. **[TAMAM — 2026-07-23, NEGATİF] Faz-B:** guarded-cheapest-13 → PPL 1.697×
    (odd-13 1.6× ile ~aynı, hatta biraz kötü), needle 1/4. Sonuç: (a) NMSE PPL'e
    TRANSFER ETMİYOR; (b) §22a uçurumu seçim artefaktı DEĞİL, yoğunluk 13'te
    gerçek. "Akıllı seçim" kaldıracı KAPANDI. Detay RESULTS §24a.
18. **[KISMEN — stabilizasyon KAPANDI] 13-kat uçurum teşhisi:** stabilize S2
    (LR 1e-4, warmup 150, 900) → 1.795× (kötüleşti); üç 13-kat config 1.6-1.8×
    (§24c). Uçurum SAĞLAM. Teşhis: S1 per-kat MSE ~0.089 (iyi) ama uçtan-uca 1.8×
    → sorun **birikimli hata** (compounding), per-kat kapasite değil. Kapanan
    kaldıraçlar: seçim (§24a) + stabilizasyon (§24c). Açık (daha zor/opsiyonel):
    (a) GERÇEK incremental curriculum (6 eğit→dondur→gruplar hâlinde ekle;
    compounding'e doğrudan saldırır), (b) daha güçlü primitif (delta/gated),
    (c) daha büyük state. NOT: §23 maliyet-hendeği derinleştirme HENÜZ çözülmedi;
    ürün/moat çerçevesinde varsayılmamalı.

## Ufukta (karar gerektirir)

13. Makale yeniden konumlandırma: "cross-chunk distilasyonla O(1) graft +
    retention-yasası ablasyonu" ana hikâye; §6/Görev C sonucu eklenince taslak
    (`docs/paper3_ml_architecture.tex`) revize.
14. **[TAMAM — 2026-07-23]** Cihaz-içi vitrin ölçümü YAPILDI (RESULTS §23):
    4k-128k, hibrit vs saf KV-cache. 128k'da VRAM %8 tasarruf, decode ~%21
    hızlanma, graft state 9.5 MB SABIT (grafted katmanlarda ~85× O(1) bellek).
    `notebooks/bench_vram_latency_v14.ipynb`, grafik `docs/assets/`.
    Açık kalan: DynamicCache doğrudan katman-sayımı (-1 döndü, notebook
    düzeltildi; bir sonraki koşuda doğrudan teyit gelecek).
15. GLA baseline'ının NaN'ı: in-house wrapper'daki kararsızlığın kökü
    (adil kıyas hijyeni; düşük öncelik).
