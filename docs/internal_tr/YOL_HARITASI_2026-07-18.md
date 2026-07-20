# HFP Yol Haritası — 2026-07-18 (Qwen graft §15 sonrası)

Amaç: **önce garanti altına al → ucuz küçük testlerle pürüzleri gider →
kanıtlanmış yetenekleri eksiksiz ve savunulabilir hale getir.** Büyük GPU
yatırımı ancak ucuz testler yön gösterdikten sonra.

## Durum özeti (dürüst)

**Kanıtlı (küçük ölçek, çok-seed, reprodüksiyonlu):**
- O(1) sabit state + chunk-tutarlılık + causal doğruluk (CI'da her push'ta)
- Train-short → infer-long uzunluk genellemesi (§3) — ana pozitif sonuç
- DPFP kapasite kazancı (§5); girişim-sınırlı bellek teşhisi (§4)
- Eval-time window + PE-tiling ile uzunluk-stabil O(1) çıkarım reçetesi (§14)

**Kanıtsız / olumsuz:**
- cubic_flux kısa-bağlam üstünlüğü replike olmadı (dürüst defterde)
- **§15 Qwen graft:** pipeline uçtan uca ÇALIŞIYOR ama PPL 2× bozuldu,
  needle tamamen ıskaladı. Sapmalar: zero-shot 2627 (>1000), S2 seq=128.

## Faz 0 — GARANTİ ALTINA AL (bugün, ücretsiz, ~1 saat)

- [ ] Kaggle'da son koşunun **Save Version**'ı alındı mı doğrula
      (`hfp_graft_final.pt`, `stage2_250/500.pt` output'ta olmalı).
- [ ] Üç checkpoint'i (`stage1_son`, `stage2_500`, `final`) tek yere topla:
      Drive `hfp_graft_ckpt/` + yerel `HFP_Project/checkpoints/` (git'e LFS'siz
      koyma; .gitignore'da kalsın, Drive yedek yeter).
- [ ] Git: RESULTS.md §15 + güncel notebook + AGENTS.md/GELISTIRME_PLANI
      commit + push (`main`, CI yeşil olmalı). Tag öner: `graft-run-1`.
- [ ] jcode ile yerel doğrulama: `python smoke_test.py` +
      `python review_scripts/verify_claims.py` (temiz taban teyidi).

## Faz 1 — UCUZ TESTLER (her biri ≤30 dk, tek T4 oturumu, sırayla)

Amaç: §15'in "neden"ini büyük eğitim yapmadan ayrıştırmak. Hepsi
`notebooks/`'a küçük hücreler olarak eklenebilir; S1/S2 GEREKTİRMEZ.

- [ ] **T1 — Needle kontrol testi (EN ÖNCE; dürüstlük kontrolü).**
      Aynı needle testini **teacher modda** (saf Qwen) koş. Teacher da
      bulamıyorsa test kurgusu kırık demektir (ör. filler döngüsü, tokenizasyon)
      ve §15'in needle sonucu modeli değil testi ölçmüş olur. Kriter:
      teacher @2048 BULDU olmalı; değilse önce testi onar.
- [ ] **T2 — out_gain init A/B (zero-shot).** Eğitimsiz graft'ta
      `out_gain`'i 1.0 → 0.1 çekip yalnız zero-shot PPL ölç (forward-only).
      Kriter: PPL 2627'den belirgin düşerse (<1000 hedef) S1'i bu init'le
      yeniden koşmaya değer. (Not: init değişirse eski checkpoint'ler o
      operating point için geçersiz — tam S1 tekrarı gerekir, bilerek karar ver.)
- [ ] **T3 — alpha/parametre otopsisi (CPU, dakikalar).** `final.pt`'den
      kafa-başına `sigmoid(alpha_logit)`, `out_gain`, `sigmoid(decay)`
      dağılımlarını çıkar. alpha ~0.13'te kaldı: bellek yolu ne kadar
      kullanılıyor? Ölü kafa var mı? Bu, S2'nin neyi öğren(e)mediğini gösterir.
- [ ] **T4 — kısa-needle taban.** Needle'ı L=256/512'de (S2'nin gördüğü
      rejim) koş. Kısa mesafede de MISS ise sorun bağlam uzunluğu değil,
      retrieval'ın hiç öğrenilmemesi → Faz 2'de recall-verisi şart demektir.

## Faz 2 — HEDEFLİ DÜZELTME (tek Kaggle oturumu, Faz 1 sonucuna göre TEK seçim)

Faz 1 hangi hipotezi işaret ederse ONA yatırım yap (hepsine birden değil):

- T1 kırıksa → testi düzelt, §15 needle satırına düzeltme notu düş, yeniden ölç.
- T2 kazandıysa → küçük-init ile **S1'i yeniden koş** (700 adım ~9.5h biliniyor;
  Kaggle 12h'e sığar), sonra S2.
- T4 "hiç öğrenmemiş" derse → **S2 verisine sentetik recall karışımı** ekle
  (WT-103 %80 + key-value passaj %20), 600 adım seq 128'de bile recall
  sinyali verir; mini-needle (512) ile ara ölçüm.
- Hiçbiri net değilse → S2'yi seq 256'ya çıkarmayı dene (bf16 autocast'lı
  teacher forward ile bellek düşer — sayısal riski önce 50 adımlık smoke ile ölç).

## Faz 3 — KANITLARI EKSİKSİZLEŞTİR (paralel, GPU'suz; jcode ile)

- [ ] GELISTIRME_PLANI Adım 1: repo düzeni (davranış değişmeden; smoke+verify yeşil).
- [ ] GELISTIRME_PLANI Adım 2: README/RESULTS tutarlılık geçişi — her iddiaya
      "kanıtlı (script) / kanıtsız / olumsuz" etiketi; §15 dahil. "Yapabildiklerini
      kanıtla" hedefinin özü bu: küçük-ölçek kanıtlı sonuçlar (O(1),
      train-short→infer-long, DPFP, §14 reçetesi) net vitrine çıksın.
- [ ] CI'ya Faz 1 T3 tarzı ucuz bir "checkpoint sanity" scripti ekle
      (alpha/out_gain aralık kontrolü) — gelecekteki koşularda erken uyarı.

## Kurallar (değişmez)

- Tek seferde TEK değişken; her koşunun tam komutu/config'i RESULTS'a.
- Kriterler ÖNCEDEN yazılı (yukarıdaki gibi); sonuç ne çıkarsa kaydedilir.
- AGPL-3.0 başlıkları ve dürüstlük kuralları (AGENTS.md) her adımda geçerli.
- 12h+ tek koşuya, Faz 1'in üç ucuz cevabı alınmadan girilmez.
