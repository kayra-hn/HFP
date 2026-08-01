# DEVİR — yeni sohbet için başlangıç belgesi

*Bu dosya, projeyi sıfırdan devralan bir asistanın okuyacağı ilk belgedir.
Güncellenme: 2026-07-28.*

---

## 1. Proje tek cümlede

**HFP:** önceden eğitilmiş bir LLM'in (Qwen2.5-1.5B) attention katmanlarının bir
kısmını **O(1)-bellekli** (sabit boyutlu, bağlamla büyümeyen) recurrent bir
mekanizmayla değiştirme çalışması. Sahibi **Kayrahan Yılmaz**, tek başına,
bedava GPU katmanlarıyla (Kaggle/Colab) çalışıyor. Lisans **AGPL-3.0**, telif
tamamen kendisinde (çift-lisanslama kapısı açık tutulmalı).

---

## 2. Okuma sırası

1. **`AGENTS.md`** (repo kökü) — çalışma kuralları. **İki kural ödünsüz:**
   bilimsel dürüstlük ve lisans/IP koruması.
2. **`RESULTS.md`** — otorite deney kaydı. **Başındaki ERRATUM'u mutlaka oku.**
   Bölümler §1'den §35'e; ~2/3'ü negatif sonuç.
3. **`docs/internal_tr/YOL_HARITASI_2026-07-28.md`** — güncel yol haritası (v2).
4. **`docs/PROJECT_SUMMARY.md`** — tek sayfa dış özet (İngilizce).
5. **`docs/internal_tr/YAYIN_PLANI.md`** — cepte bekleyen yayın planı.
6. **`docs/internal_tr/BEKLEYEN_ISLER.md`** — iş defteri.

---

## 3. Nerede duruyoruz (2026-07-28)

### Doğrulanmış ve kalıcı
- **6-katman O(1) graft, iki Pareto noktası:**
  149.910 param (%0.01) → **PPL 1.111×** (3 seed) | ~19M param (%1.3) → **1.043×**
  (tek seed; `≤1.05×` kriterini ilk kez geçti)
- **128k bağlamda:** ~%8 tepe VRAM, ~%21 decode hızlanması; graft state 9.5 MB **sabit**
- **Yoğunluk duvarı:** 6→13 katman 1.6-1.8× PPL; dört bağımsız config aynı bantta.
  Sebep **birikimli hata (compounding)**, per-katman kapasite değil (S1 MSE ~0.089 iyi)
- **Retention yasası rejime bağlı:** yoğun-yazımda fark yok (§15h); seyrek-yazımda
  ön-kayıtlı, eşleşmiş, n=16: **+0.48 nat, 13/16 seed, p=0.012** (§28a)
- **Metodolojik katkı:** hibrit attention/O(1) modellerde görünen uzun-menzil geri
  getirme tamamen artık **KV-cache** tarafından taşınabilir; alanda genelde cache
  açıkken ölçülüyor (§30 + erratum)

### KAPANAN hatlar — yeniden açılmayacak
- **Yoğunluk itme:** katman seçimi (§24a), stabilizasyon (§24c), exposure-bias
  (§25a) — üçü de negatif. ~6 katman pratik tavan.
- **Graft ile hafıza organı (§30-§34):** dört ön-kayıtlı müdahale
  (`no_grad` kaldırma, state detach kapatma, maskeli recall denetimi, kendi
  eğitilebilir q/k/v projeksiyonları) — **hepsi %0**. Üst sınır %100 iken.
  Graft yolu hafıza için kapandı.
- **Cubic:** bulgu geçerli, mühendislik nedeniyle **rafta** (sıralı z-taraması,
  mobil ihraç sorunu, dar stabilite penceresi). Raf kaldırma koşulları
  BEKLEYEN #21'de.

### Hazırlanmış ama KOŞULMAMIŞ (kayıt için)
- **§29 `parallel_cubic`:** cubic'in sıralı z-taramasını kaldıran blok-donmuş form.
  **Kapı testi koşuldu ve geçti** (rec_block=1'de sıralı cubic ile **bit-bit aynı**,
  1.16× hız). Kalite deneyi (n=16) koşulmadı; cubic raflandığı için gerekçesi zayıf.
- **§32 delta yazım (graft'ta):** hazırlandı, koşulmadı; §33a/§34a graft-hafıza
  yolunu kapatınca gerekçesi ortadan kalktı.

### Şu an koşan / bekleyen

- **AKTİF: §29b** (`notebooks/parallel_cubic_v29.ipynb`) — `parallel_cubic`'in
  kalite deneyi. 3 kol (exp / cubic sıralı / parallel_cubic) × 16 eşleşmiş seed,
  seed-major. Birincil metrik `val_cross`. Ön-kayıt RESULTS §29b'de.
  **Stratejik gerekçe:** `cubic_flux` bu projedeki **tek** prior-art olmayan
  mimari öğe (O(1) state, DPFP, delta, gated decay, graft — hepsinin sahibi
  başkası). Kazandığı rejim = seyrek-yazım/uzun-ömür = cihaz-içi kişisel hafıza
  rejimi (§26). Raf koşullarından #1 §29 kapı testiyle kapandı, #2 aynı
  değişiklikle açılıyor, #3 (LM ölçeğinde tekrar) hiç koşulmadı.
  **Uyarı:** bu koşu §28a'yı temiz metrikte yeniden ölçüyor; §28a'nın etkisi
  zayıflayabilir. Ön-kayda öyle yazıldı.

- **§35 — TERK EDİLDİ (2026-08-01), kapanmadı.** v4 ön-kaydı yazıldı ve
  başlatıldı, **hiçbir v4 sonucu görülmeden** kullanıcı tarafından durduruldu.
  Karar stratejik, ampirik değil — hiçbir veri kararı etkilemedi, öyle kaydedildi.
  v4 ön-kaydı, durma kuralı ve 2-koşu bütçesi olduğu gibi duruyor; hat yeniden
  açılırsa geçerli. §35a'nın geçersizlik bulguları (araç eşiğin altında,
  metrik kirli) her gelecek denemede geçerli.
- **§35a — v3 KOŞULDU ve GEÇERSİZ (2026-08-01).** Negatif sonuç DEĞİL, geçersiz
  koşu. İki bağımsız kusur: (1) araç §26b yeteneğini üretemedi — taban K=0'da
  **%8.5**, §26b'de %33.2; bir sınırı geçemeyen modelde "neden ikinciyi
  geçemiyor" sorulamaz. (2) birincil metrik kirliydi — B chunk etiketleri
  chunk-içi çiftleri de içeriyor, P=4'te denetlenen 5 tokenın 4'ü chunk-içi;
  gözlenen 2.2974 taşıma tamamen şansta olsa da çıkacak değer.
  Kök sebep: v2 kapsam kısıntısı **CPU** ölçümüne göre yapıldı, GPU'ya geçilince
  tekrar ölçülmedi. v3'ün tamamı T4'te **833 sn** sürdü (limitin %1.9'u).
  Ayrıca: `ESKI` sondanın K=0'da %100'ü üst sınır DEĞİL — o dizi
  `[tgt_k, v, tgt_k]`, üç token, `local_window=8` içinde.

- **§35 v4** (`notebooks/chain_capacity_v35.ipynb`): küçük-ölçek modelde
  **kapasite × yazım kuralı** taraması. Merkezi soru: state neden **bir** sınırı
  geçiyor da **ikincisini** geçemiyor? (§26b, `exp` kolu: K=0'da **%33.2**, seed
  aralığı %8.5–69.5; K=2'de şansa düşüyor. **Dikkat:** eskiden bu belgede yazan
  "%33-53" yanlıştı — 33.2 `exp` kolunun, 53.4 `cubic` kolunun ortalamasıdır;
  §35 `exp` sabitliyor.)
  **v4 tasarımı:** araç ayarları §26b/§27/§28a ile birebir geri alındı
  (CTX 256, 1200 adım, BS 8, CARRY_MAX 16, P 6, DIST_EVERY 64),
  **2 kol × 6 seed** (taban nu2/additive vs nu4/delta).
  **ARAÇ GEÇERLİLİK KAPISI:** taban kol K=0'da seed-ortalaması **≥%25** değilse
  koşu geçersiz ilan edilir ve hipotez hakkında hiçbir şey yazılmaz. (v3'te bu
  kapı yoktu, bu yüzden sonuç gibi görünen bir sayı üretti.)
  **Birincil metrik: `val_cross`** — yalnız cross-chunk hedef tokenındaki kayıp,
  eşleşmiş, eşik Δ ≤ −0.15 nat + 6/6 işaret. **Teşhis: `val_inchunk`** ayrı
  hesaplanır (araç öğrendi mi?). Sonda ikincil/betimleyici.
  **Bu bir TARAMA deneyi** — n=6'da 6/6 için p=0.0156; "kanıtlandı" denmeyecek.

  **DURMA KURALI (ön-kayıtlı, koşudan önce yazıldı):** kapı geçilmezse hat
  **kapanır**; SİNYAL çıkarsa tek izinli devam ayrıştırma+güç koşusu ve
  sonrası için **yeni karar** gerekir; TERS/SONUÇSUZ çıkarsa hat **kapanır**.
  **Sert bütçe: bu hatta v4 dahil en fazla 2 koşu**, sonra yayın hattına dönülür.
  Gerekçe: §30-§35 arası altı bölümün tamamı negatif ya da geçersiz ve her biri
  "bir sonraki müdahale tutar" diye başladı.
  Tam ön-kayıt RESULTS §35'te.

---

## 4. Çalışma kuralları (uyulması şart)

- **Ön-kayıt:** her koşudan ÖNCE kriterler **RESULTS.md'ye** yazılır — otorite kayıt
  orasıdır. Notebook markdown'ı veya yol haritası **ön-kayıt sayılmaz** (notebook koşu
  sırasında düzenlenebilir ve nbstrip filtresi altındadır). Bu kural bir kez ihlal
  edildi (§35 önce yalnız notebook/haritadaydı), koşudan önce fark edilip düzeltildi.
- **Negatifler yayınlanır.** Kayıtların ~2/3'ü negatif; bu bir özellik, kusur değil.
- **Güç kontrolü:** iki kol da şanstaysa "null" değil **"sonuçsuz"** raporlanır.
- **Üst-sınır kolu:** bir sonda başarısızsa, base modelin o sondayı çözüp
  çözemediği ayrıca ölçülür (model sınırı mı mekanizma sınırı mı?).
- **Hata bulunca erratum yazılır**, sessizce düzeltilmez (bkz. RESULTS başı).
- **Tek değişken:** kollar arası fark tek bir parametre olmalı.
- **Commit mesajı:** ne yapıldı + neden + hangi §. Türkçe, ASCII (Türkçe karakter
  kullanma, git log bozuluyor).
- **Push kullanıcıya ait.** Asistan commit eder, kullanıcı push eder.

---

## 5. Ortam tuzakları (hepsi yaşandı)

- **Notebook'lar GPU assert'i içerir** — CPU ortamında açılmaz. CPU testi için
  `review_scripts/memory_probe_cpu.py` gibi **script**'ler var, terminalden koşar.
- **`python review_scripts/x.py`** → `No module named hfp`. Script'lerde sys.path
  düzeltmesi var; yoksa `PYTHONPATH=.` ekle.
- **`git pull` çakışması:** notebook çalışınca çıktı değişir →
  `git checkout -- notebooks/<dosya>.ipynb && git pull`. Kalıcı çözüm için
  `.gitattributes`'ta nbstrip filtresi var (kurulumu dosyada yazılı).
- **Checkpoint adlandırma:** `hfp_graft_{mode}_g{N}{SEL}_s{seed}_{tag}.pt`.
  `SEL` kol etiketi (W=yazma-BPTT, M=maskeli denetim, P=kendi projeksiyonlar,
  D=delta). **Farklı kollar asla aynı ada yazmamalı** — bir kez üzerine yazma
  riski oldu, düzeltildi.
- **Checkpoint seçerken isme göre sıralama YAPMA** — `g6W` alfabetik olarak
  `g6_`'dan önce gelir, yanlış dosya seçilir. mtime veya açık filtre kullan.
- **`strict=False` ile yükleme sessizce başarısız olabilir** → yükleme sonrası
  eşleşen tensör sayısı ve eğitim izi (örn. `out_gain` init'ten sapmış mı) kontrol edilmeli.
- **Kaggle:** `ChunkLoadError` = tarayıcı önbelleği, sert yenile/gizli sekme.
  Batch limiti 12 saat. Qwen'i **Add Input**'tan ekle (HF Xet 403'ünü atlar).
  Aynı notebook'u interaktif + Save Version aynı anda koşturma (loglar çift basar,
  iki süreç aynı dosyalara yazar).
- **Colab:** Drive mount edilirse checkpoint'ler kalıcı; edilmezse oturumla gider.
- **Lightning:** disk kalıcı, ama GPU kredisi bitebilir.
- **Süre tahminleri:** ölçmeden tahmin etme. CPU'da küçük-ölçek eğitim beklenenden
  ~6× yavaş çıktı (1200 adım ≈ 95 dk). İlk 100 adımın süresini ölç, oradan çarp.

---

## 6. Kullanıcı bağlamı (tavsiye kalibrasyonu için)

- 21 yaşında, tek başına, **bedava compute** ile çalışıyor. Kaggle haftalık 30 saat.
- **Yakın vadede gelir istiyor**; tam zamanlı iş istemiyor, bağımsız kalmak istiyor.
- Uzun vadeli hayali: **cihaz-içi, mahrem, unutmayan kişisel hafıza asistanı.**
  Bu hedefe duygusal bağı güçlü ve projeyi bırakmayacağını net söyledi.
- **Dürüstlük bekliyor ve kaldırıyor.** Kötü haberi yumuşatma; ama ürün ile bulgu
  arasındaki farkı sürekli netleştirmek gerekiyor (bu ayrım tekrar tekrar karıştı).
- Sık düşülen tuzak: her negatiften sonra "bir deney daha" zinciri. Yol haritasında
  **"yapmayacaklarımız"** listesi bunun için var; ona sadık kal.

---

## 7. Sıradaki karar noktaları

| ne zaman | soru |
|---|---|
| §35 sonrası | sinyal var mı? (eşleşmiş Δ ≤ −0.15 nat, 4/4 seed) |
| §35 sinyal verirse | tek izinli devam: ayrıştırma + güç (nu4/additive vs nu2/delta, 8-12 seed) |
| §35 ters çıkarsa | girişim hipotezi çürüdü, hat kapanır |
| §35 sonuçsuz çıkarsa | KARAR Kayrahan'ın: 8-12 seed'e çıkar mı, yoksa park edip yayına/okuma-yolu teşhisine mi geçer? ("etki yok" YAZILMAZ) |
| istenirse | yayın planını cepten çıkar (yeni deney gerektirmiyor; tek borç 1.043×'in çok-seed replikasyonu) |
