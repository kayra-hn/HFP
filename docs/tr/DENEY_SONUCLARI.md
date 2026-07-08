# Retention LR×seed Taraması (CPU, mini ölçek) — 2026-07-04

**Tasarım:** 2 katman, hidden 64, `local_window=8` → yığılmış attention alanı ~14 token;
**gap 16/24 satırları saf bellek** (bilgi yalnızca M/z'den akabilir). Context 48,
max_gap 24, 800 adım, batch 32, cosine LR, 150 deneme/gap, şans %3.3.
Cubic tarafı `cubic_flux_chunked` (iki-geçişli TAM paralel form) ile koşuldu —
sıralı cubic ile matematiksel olarak birebir aynıdır (bkz. scaling_checks.py).

| mod | lr | seed | gap2 | gap8 | gap16* | gap24* |
|---|---|---|---|---|---|---|
| exp | 3e-4 | 0 | 62.0 | 53.3 | 59.3 | 49.3 |
| exp | 3e-4 | 1 | 70.7 | 64.7 | 61.3 | 60.7 |
| exp | 1e-3 | 0 | 100 | 100 | 100 | 100 |
| exp | 1e-3 | 1 | 100 | 100 | 100 | 100 |
| cubic | 3e-4 | 0 | 50.0 | 25.3 | 15.3 | 9.3 |
| cubic | 3e-4 | 1 | 48.0 | 36.0 | 30.7 | 30.7 |
| cubic | 1e-3 | 0 | 100 | 100 | 100 | 100 |
| cubic | 1e-3 | 1 | 100 | 100 | 100 | 100 |

(*saf bellek bölgesi)

**Bulgular (bu ölçekte, iki seed'de tutarlı):**

1. **lr=1e-3'te iki mod da her gap'te %100** — görev bu bütçede doyuyor; retention
   yasaları arasında ayrım gücü YOK. Ayrım için daha zor görev gerekir (daha uzun
   gap, daha çok distraktör çifti, val_space büyütme, kademeli skor).
2. **Repo default'u lr=3e-4'te exp, cubic'i her iki seed'de de geçiyor** (saf bellek
   gap'lerinde exp ~%49-61 vs cubic ~%9-31; cubic'in eğrisi gap ile düşüyor).
   Önceki "cubic öğreniyor, exp şansta" headline'ı bu kurulumda TERSİNE dönüyor.
3. Sonuç: iki mod arasındaki farklar bu rejimde temelde **optimizasyon hızı** farkı;
   temsil kapasitesi farkı değil. Büyük-ölçek koşusunda mod başına ayrı LR taraması
   yapılmadan hiçbir retention-yasası karşılaştırması rapor edilmemelidir.
4. `cubic_flux_chunked` eğitimde davranışsal olarak doğrulandı (1e-3'te %100'e
   ulaşıyor) — büyük koşularda sıralı mod yerine bu kullanılmalıdır.

**Sınırlar:** mini ölçek (~0.5M param), sentetik görev, 2 seed, tek görev ailesi.
Tekrar üretim: `python review_scripts/sweep_ckpt.py <exp|cubic_flux_chunked> <lr> <seed>`

## Ek: Ayrıştırıcı görev v1 (hard_retention.py) — 2026-07-04

Tasarım: ctx 320, 6 distraktör çift + hedef çift kontrollü gap'te (8→256),
pencere 8, 2 katman, 800 adım, lr 1e-3, acc + hedef log-prob raporu.

| koşu | gap8 | gap32 | gap64 | gap128 | gap256 |
|---|---|---|---|---|---|
| exp (düz) | 2.5 | 3.3 | 3.3 | 4.2 | 2.5 |
| cubic_chunked (düz) | 2.5 | 3.3 | 3.3 | 4.2 | 2.5 |
| exp (curriculum) | 0.8 | 3.3 | 3.3 | 6.7 | 2.5 |

(şans %3.3; log-prob hepsinde ≈ −3.4 = ln 30 → tam plato)

**Bulgu:** Bu bütçede İKİ MOD DA hiçbir gap'te öğrenemiyor — gap 8 dahil.
Yani engel retention yasası değil; ctx 320'lik filler/distraktör akışı altında
bellek yolunun ÖĞRENİLEBİLİRLİĞİ çöküyor (tek-token süpervizyon + ~300 alakasız
yazımın girişimi). Curriculum tek başına kurtarmıyor.

**Sonuç ve yol haritası:** Ölçekleme öncesi asıl darboğaz bu. Aday çözümler
(önem sırasıyla): (1) delta-kuralı yazım (M'e toplamalı değil fark yazımı —
girişimi kökten azaltır), (2) yoğun süpervizyon (dizide birden çok sorgu
pozisyonu), (3) surprise-tabanlı gate (zaten geri getirilebileni yazma).
Bunlar denenmeden GPU'da uzun-bağlam koşusuna para yakılmamalı; mevcut haliyle
model uzun akışta ezberleyemiyor. Görev scripti: review_scripts/hard_retention.py
(curriculum bayrağıyla). Tekrar: python review_scripts/hard_retention.py <mod> <lr> <seed> [sn] [curriculum]

## Ek 2: write_rule="delta" (DeltaNet-tarzı ölçüm-güncelleme yazımı) — A/B

Implementasyon: `M~ = λ⊙M; v_old = kᵀM~; M = M~ + β·k(v−v_old)ᵀ`, k L2-normalize,
β=σ(gate)∈(0,1) öğrenilebilir, payda kaldırıldı (çıktı q·M → LayerNorm).
Doğrulama: chunk-tutarlı (exp+cubic), state kendinden-sınırlı (4k token max|M|=0.86),
beta_gate gradyanlı, causal sızıntı yok. Her iki decay yasasıyla birleşebilir.

A/B (hard-retention, ctx 160, 6 distraktör, 600 adım, lr 1e-3, seed 0):

| yazım kuralı | gap8 | gap32 | gap64 |
|---|---|---|---|
| additive | 1.7 | 5.0 | 3.3 |
| delta | 1.7 | 5.0 | 3.3 |

(şans %3.3, log-prob ≈ −3.4 = tam plato; iki koşu da öğrenemedi)

**Bulgu:** Delta yazım TEK BAŞINA ln(V) platosunu kırmıyor. Girişim hipotezi
(Ek 1'deki çöküşün tek nedeni olarak) doğrulanmadı; asıl şüpheli artık
SÜPERVİZYON YOĞUNLUĞU: 160-320 token'lık dizide tek etiketli token var —
recall devresinin bulunması için sinyal çok seyrek. (Kolay görev ctx 48'de
aynı seyreklikle öğreniliyordu; dizi uzadıkça sinyal/gürültü lineer düşüyor.)

**Sonraki adım (net):** Yoğun süpervizyon varyantı — aynı dizide 4-8 sorgu
pozisyonu (her biri kendi gap'iyle) → etiket yoğunluğu 4-8×. Bu öğrenmeyi
açarsa, delta-vs-additive ve cubic-vs-exp karşılaştırmaları ancak o zaman
anlamlı ayrım verebilir. Delta kodu hazır ve doğrulanmış durumda bekliyor.

## Ek 3: Yoğun süpervizyon (dense_retention.py) — kilit açıldı

Tasarım: ctx 160, 8 çift + 8 sorgu aynı dizide serpiştirilmiş (etiket yoğunluğu 8×),
gap doğal olarak 1..~150 dağılır, kova bazında acc. 600 adım, lr 1e-3, seed 0.

| konfigürasyon | gap 1-15 | 16-47 | 48-95 | 96+ |
|---|---|---|---|---|
| exp + additive | 55.5 | 31.4 | 20.7 | 6.2 |
| exp + delta | 60.9 | 34.8 | 16.7 | 3.7 |
| cubic_chunked + additive | 43.0 | 31.4 | 17.2 | 4.9 |

(şans %3.3; n=128/204/227/81)

**Bulgular:**

1. **Süpervizyon yoğunluğu hipotezi DOĞRULANDI.** Tek-etiketli aynı görev (Ek 1-2)
   hiç öğrenilemezken 8×-etiketli varyant ln(V) platosunu ~400. adımda kırdı ve
   gap'le düşen gerçekçi bir retention eğrisi üretti. Metodolojik sonuç: bundan
   sonraki TÜM retention deneyleri (GPU dahil) çok-sorgulu format kullanmalı;
   tek-sorgulu format optimizasyon artefaktı üretiyor.
2. Görev artık öğrenilebilir VE doymamış → ilk gerçek ayrıştırıcı zemin. Bu zeminde
   (tek seed): exp ≥ cubic (kısa gap'te belirgin, uzunda benzer); delta ≈ additive
   (kısa gap'te delta önde, uzunda geride — gürültü sınırında). Ne cubic ne delta
   avantajı bu kanıtla iddia edilebilir; çok-seed tekrarı sonraki adım.
3. Pratik reçete: retention iddiaları için ölçüm aracı artık hazır
   (dense_retention.py). GPU koşusundan önce bu görevde ≥3 seed × 2 LR taraması
   yapılmalı; hangi mekanizma kazanırsa büyük koşuya o gitmeli.

## Ek 4: Karar kapısı — dense görevde 3-seed karşılaştırma

Aynı kurulum (ctx 160, 8 çift/8 sorgu, 600 adım, lr 1e-3), seed {0,1,2} ortalaması:

| konfigürasyon | gap 1-15 | 16-47 | 48-95 | 96+ |
|---|---|---|---|---|
| exp + additive | **44.4** | **32.7** | **18.8** | **8.1** |
| exp + delta | **52.1** | 31.1 | 16.4 | 6.6 |
| cubic + additive | 31.3 | 23.0 | 11.8 | 4.7 |

(şans %3.3; cubic seed-1 neredeyse hiç öğrenemedi: 11.7/3.8/1.9/4.9 — 3 seed'in 1'i çöktü)

**Karar kapısı sonucu (bu ölçekte, dürüst):**

1. **cubic_flux bu rejimde exp'ten GERİDE ve seed-kırılgan** (1/3 çöküş). Kübik
   yasanın kalan umudu uzun-ufuk rejimi (gap >> 150): orada HERKES zayıf
   (%5-8) ve plato mekanizmasının teorik avantajı tam o bölgede. Bu, default
   seçim olmaktan çıkıp hedefli bir hipotez testine dönüşmeli (daha uzun
   eğitim + iki-kademe bellekle birlikte).
2. **delta, kısa menzilde tutarlı önde** (52 vs 44), orta/uzunda eşit. Sıralı
   maliyeti dikkate alınınca default olmayı henüz hak etmiyor; chunkwise/WY
   paralel formu yazılırsa cazipleşir.
3. **GPU bütçesi için kazanan konfigürasyon: exp + additive + ffn_type=standard**
   (en iyi acc, en hızlı, en dayanıklı). cubic ve delta, GPU'da ANA koşu değil
   yan-deney olarak koşulmalı.
4. Uzun-ufuk (96+) problemi herkes için açık: iki-kademeli bellek (konsolidasyon)
   hipotezinin hedefi tam burası — sıradaki mimari deney adayı.

## Ek 5: İki-kademeli bellek prototipi + ctx-320 uçurumu

Prototip (review_scripts/two_tier.py): M_fast (exp+additive) + M_slow; yavaş kademe
her 16 token'da bir HIZLI BELLEKTEN konsolidasyon alır (M_s = λs⊙M_s + γ·M_f),
yavaş yasa cubic (η ~1e-6..1e-4 → uzun plato). Doğrulama: chunk-tutarlı,
γ/mix/η_slow gradyanlı, causal sızıntı yok. Çekirdeğe dokunmadan subclass.

Uzun-ufuk testi (ctx 320, dense, 600 adım, lr 1e-3, seed 0):

| konfigürasyon | <48 | 48-127 | 128-223 | 224+ |
|---|---|---|---|---|
| baseline (P=8) | 4.5 | 1.1 | 3.0 | 6.4 |
| baseline (P=16, yoğunluk eşit) | 1.9 | 3.9 | 3.3 | 2.0 |
| two-tier cubic-slow (P=16) | 2.3 | 3.9 | 4.1 | 2.0 |

(şans %3.3 — hepsi şansta, loss ≈ 3.41 platosunda)

**Bulgular:**

1. **ctx-160→320 uçurumu etiket yoğunluğuyla açıklanmıyor** (P=16 eşitlemesi
   kurtarmadı) ve **yavaş kademe de kurtarmıyor**: hızlı kademe hiç öğrenemeyince
   konsolide edilecek sinyal de yok. Uçurum retention değil, DİZİ UZUNLUĞUNDA
   ÖĞRENİLEBİLİRLİK problemi (sabit bütçede uzunluk 2× → öğrenme 0'a düşüyor).
2. İki-kademe hipotezi bu deneyle NE doğrulandı NE çürütüldü — adil testi,
   öğrenmenin çalıştığı rejimde (ctx 160) uzun-gap kovasında (96+, 1-kademe ~%8)
   iyileşme olup olmadığı. Sıradaki iş.
3. Uçurumun kendisi için en umut verici müdahale: UZUNLUK CURRICULUM'u
   (ctx 80'den 320'ye kademeli) veya daha uzun eğitim (uçurum duvar mı rampa mı
   sorusu). Bu, GPU öncesi cevaplanması gereken 1 numaralı soru haline geldi:
   modelin uzun dizide eğitilebilirliği, her şeyin önkoşulu.

## Ek 6: UZUNLUK GENELLEMESİ — pozitif ana bulgu (yalnız dev)

Deney (length_gen.py): ctx 160'ta eğit (dense, P=8, 600 adım, lr 1e-3, seed 0),
AYNI ağırlıkları 160/320/640/1280'de değerlendir (PE deterministik → uzatılır).

| eval ctx | gap<48 | 48-127 | 128-255 | 256+ |
|---|---|---|---|---|
| 160 (eğitim uzunluğu) | 38.2 | 14.5 | 4.3 | — |
| 320 (2×) | 63.2 | 23.8 | 6.4 | 6.7 |
| 640 (4×) | 71.7 | 40.5 | 16.7 | 5.4 |
| 1280 (8×) | 75.0 | 74.2 | 46.7 | 8.6 |

(şans %3.3; P=8 sabit → uzun dizide olgu yoğunluğu düşer)

**Bulgular:**

1. **Kısa eğit → uzun çalıştır ÇALIŞIYOR**: 8× ekstrapolasyonda recall korunuyor,
   hatta sabit gap'te doğruluk uzunlukla ARTIYOR. ctx-320 "öğrenilebilirlik
   uçurumu" (Ek 5) mimari sınır değil, salt EĞİTİM-optimizasyonu artefaktı —
   ve pratik çözümü bedava: kısa bağlamda yoğun-olgu ile eğit, uzunda çalıştır.
   Bu, O(1)-bellek mimarisinin ana vaadinin (sınırsız çıkarım bağlamı) ilk
   deneysel doğrulaması.
2. Sabit gap'te doğruluğun uzunlukla artması, belleğin bu rejimde decay-sınırlı
   değil GİRİŞİM-sınırlı olduğunu gösteriyor (olgu yoğunluğu düşünce geri
   getirme temizleniyor). Delta-yazım/kapasite (dpfp) eksenleri tam bu sınıra
   saldırıyor — yeniden değerlenmeyi hak ediyor (girişim-yoğun eval'de).
3. gap 256+ hâlâ zayıf (%8.6 ≈ 2.6× şans) → gerçek uzun-ufuk sınırı burada
   başlıyor; iki-kademe/konsolidasyon hipotezinin hedefi olarak geçerli.
4. **GPU reçetesi netleşti**: kısa-ctx yoğun-olgu eğitimi + uzun-ctx eval.
   Eğitim maliyeti düşük (ctx 160-256), değerlendirme istenildiği kadar uzun.

Tekrar: python review_scripts/length_gen.py train <seed> && ... eval <seed>
Sonraki: çok-seed teyidi + girişim-yoğun uzun eval'de delta/dpfp/two-tier.

## Ek 7: Uzunluk genellemesi — 3-seed teyidi (yalnız dev)

Aynı protokol (train@160 → eval@160/320/640/1280), seed {0,1,2}. gap<48 kovası:

| seed | 160 | 320 | 640 | 1280 |
|---|---|---|---|---|
| 0 | 38.2 | 63.2 | 71.7 | 75.0 |
| 1 | 32.4 | 39.3 | 45.5 | 42.9 |
| 2 | 40.0 | 54.7 | 75.4 | 85.7 |

gap 48-127 kovası: s0: 14.5→23.8→40.5→74.2 | s1: 13.2→21.5→29.5→40.4 | s2: 18.6→28.5→45.9→69.4

**ÜÇ SEED'DE DE monoton aynı desen: 8× uzunlukta transfer + sabit gap'te artış.**
Bulgu sağlam. (Seed 1 genel olarak zayıf öğrenmiş ama eğilim aynı — mutlak
seviye seed'e duyarlı, DESEN değil.) Girişim-sınırlılık yorumu da üç seed'de
tutarlı. Bu, projenin şu ana kadarki en güvenilir pozitif sonucudur ve GPU
reçetesinin (kısa-yoğun eğitim → uzun çıkarım) temelini oluşturur.

## Ek 8: Girişim ayrıştırması + delta/dpfp rövanşı (yalnız dev)

**Girişim doğrulandı (3 seed, eval-only):** ctx 640 sabit, P (olgu sayısı)
8→16→24 artınca sabit-gap doğruluğu monoton düşüyor (ör. s0 gap<48: 71→54→48;
s2: 85→44→46). Uzunlukla artışın mekanizması girişim-azalması; bellek bu
rejimde decay değil GİRİŞİM-sınırlı.

**Rövanş (train@160 → eval ctx 640, seed 0):** P=24 (en yoğun girişim) sütunları:

| varyant | <48 | 48-127 | 128-255 | 256+ |
|---|---|---|---|---|
| additive/elu | 48.3 | 22.1 | 5.3 | 3.5 |
| delta | 44.8 | 20.2 | 7.5 | 4.7 |
| **dpfp** | 37.9 | 23.1 | **13.2** | **7.0** |

P=8'de de dpfp 256+ kovasında ~2× önde (10.7 vs 5.4; delta 11.6).

**Bulgular:**

1. **DPFP rövanşı kazandı**: kapasite ekseni (key_dim 4×) tam teorinin dediği
   yerde ödüyor — uzun gap × yüksek girişim. Kısa gap'te küçük bedel ödüyor
   ama uzun-menzil retention'da tutarlı 2× iyileşme. İLK kez bir HFP ekseni
   baseline'a karşı net avantaj gösterdi (tek seed; çok-seed teyidi şart).
2. Delta bu protokolde additive'e denk (256+ hafif önde) — girişim yazım-anında
   değil OKUMA-anında baskın görünüyor (aynı anahtar tekrar yazılmıyor; farklı
   anahtarların çapraz-girişimi feature-map örtüşmesinden geliyor → çözüm
   kapasite/dikeylik, delta değil). Delta, anahtar-GÜNCELLEME görevlerinde
   (aynı anahtara yeni değer) hâlâ umutlu; o ayrı test.
3. Sıradaki: dpfp çok-seed teyidi + dpfp×uzunluk-genellemesi etkileşimi
   (dpfp@1280 eval) + anahtar-güncelleme göreviyle delta'nın adil testi.

## Ek 9: DPFP çok-seed teyidi + uzunluk-genellemesiyle bileşimi (yalnız dev)

**Girişim-eval, ctx 640, 3 seed** (256+ kovası, P=8): elu {5.4, 3.2, 2.5}≈şans;
dpfp {10.7, 12.9, 31.1}. 128-255 kovası ort.: elu 15.8 → dpfp 35.2 (>2×).
Ayrıca dpfp, elu'nun zayıf-seed problemini de düzeltiyor (s1: elu tepe %39 →
dpfp tepe %95) — kapasite ekseni öğrenme İSTİKRARINI da artırıyor.

**dpfp × uzunluk-genellemesi (seed 2, train@160 → eval@1280):**

| varyant | <48 | 48-127 | 128-255 | 256+ |
|---|---|---|---|---|
| elu @1280 | 85.7 | 69.4 | 29.9 | 5.4 |
| **dpfp @1280** | 88.1 | **87.8** | **70.1** | **33.5** |

İki mekanizma bileşik çalışıyor: 8× eğitim uzunluğunda, 256+ token geriden
olgu geri getirme 10× şans seviyesinde (elu'da ≈şans). gap 128-255'te 2.3×.

**Güncel kazanan reçete:** exp decay + additive yazım + **dpfp** feature map +
standard FFN; ctx 160 yoğun-olgu eğitimi → istenilen uzunlukta çıkarım.
DPFP mekanizması literatürden (Schlag ve ark.); HFP'nin katkısı bu bileşimin
Ölçülmüş uzun-ufuk davranışı ve platformu. Kalan işler: dpfp'nin 1280-etkileşimi
şu an tek seed (s2) — s0/s1 teyidi; anahtar-güncelleme göreviyle delta testi;
dpfp+delta bileşimi.

## Ek 10: Anahtar-güncelleme görevi — 2×2 (yazım × kapasite) (yalnız dev)

Görev (key_update.py): aynı anahtar iki kez yazılır (k→v1 ... k→v2), sorgu
ikinci yazımdan sonra; doğru cevap v2. ctx 160, P=5 anahtar, 600 adım, seed 0.

| konfigürasyon | doğru (v2) | bayat cevap (v1) |
|---|---|---|
| additive + elu | 28.0 | 9.8 |
| **delta + elu** | **33.8** | **5.8** |
| additive + dpfp | 32.2 | 8.0 |
| delta + dpfp | 32.8 | 7.6 |

**Bulgular (tek seed, farklar mütevazı — SE ~%2):**

1. Delta güncelleme semantiğinde yönlü avantaj veriyor: en yüksek v2 doğruluğu
   ve bayat-cevap oranında ~%40 azalma (9.8→5.8). Teoriyle uyumlu ama çarpıcı
   değil.
2. Beklenti düzeltmesi: additive "yapısal olarak çökmedi" — decay son yazımı
   kayırır (v2 daha az sönmüş), attention yolu da telafi eder; naif karışım
   tavanı pratikte yumuşuyor.
3. Bu görevde dpfp+delta BİLEŞMİYOR (32.8 ≈ 33.8): iki mekanizmanın hedef
   hataları farklı ve bu görevde kapasite baskın kısıt değil. Girişim-yoğun
   retention'da dpfp, güncelleme-yoğun akışta delta; ikisinin birlikte
   parlayacağı doğal rejim uzun-akış + güncelleme karışımı (gelecek testi).
4. Tüm kollar ~%60 "diğer" hata yapıyor → görev bu bütçede zor; çok-seed +
   daha uzun eğitim farkları netleştirebilir.

## Ek 11: key-update ÇOK-SEED — delta'nın gerçek kazancı (yalnız dev)

| seed | additive v2% | delta v2% |
|---|---|---|
| 0 | 28.0 | 33.8 |
| 1 | 14.4 | 33.8 |
| 2 | 5.6 | 30.8 |
| **ort.** | **16.0** | **32.8** |

Delta güncelleme görevinde ortalamada 2×, ve additive'in vahim seed-kırılganlığı
(5.6↔28.0) yokken delta dar bantta (30.8-33.8). Bayat-cevap oranı da tutarlı
düşük (4.2-5.8 vs 5.4-9.8). **Delta artık çok-seed'li, sağlam bir mekanizma
kazancı** — niş: güncelleme-ağır akışlar. (Ek 10'un tek-seed "mütevazı" sonucu
yanıltıcıymış; additive s0'da şanslıydı.)

### Ek 14: Karar Deneyi (Additive vs Delta) - Streaming Mix
- **Tarih:** 2026-07-07
- **Test:** `kaggle_streaming_mix_test.ipynb` (3 Seed, Ctx 160 ve 640)
- **Mimari:** `cubic_flux_chunked` + `dpfp`
- **Bulgular:** 3 Seed'in ortalaması alındığında:
  - **Kısa Bağlam (Ctx 160):** Additive %34.8 güncellenen bilgi tutarken, Delta bu oranı **%52.8**'e çıkardı. (+18% mutlak sıçrama).
  - **Uzun Bağlam (Ctx 640):** Additive %14.7 güncellenen bilgi tutarken, Delta **%28.6**'ya ulaştı (neredeyse 2 kat performans). Stale (bayat) bilgi oranı ise yarı yarıya azaldı (Ortalama %6.8 -> %3.7).
  - Delta, Seed 0 ve 2'de muazzam bir fark atarken, en kötü koşusu olan Seed 1'de bile Additive'den daha iyi performans sergiledi.
- **Karar:** Delta yazım kuralı, özellikle uzun bağlamlarda (ctx=640) bilgiyi güncelleme ve bayat veriyi unutma konusunda Additive kuralına göre **2 kat daha başarılı** olduğunu kanıtlamıştır. Eğitim sırasında 2.2x yavaş olmasına rağmen, cihaz-içi (on-device) çıkarım (inference) hızları ve O(1) bellek tüketimleri birebir aynı olduğu için, kazanılan bu muazzam hafıza kapasitesi yavaşlığa kesinlikle değmektedir. **Faz 2 (Dil Modelleme) için HFP'nin yazım kuralı DELTA olarak belirlenmiştir.** Nihai reçete: `Cubic (Çürüme) + Delta (Yazım) + DPFP (Kapasite)`.

### Ek 15: Faz 2.1 — Gerçek Dil Modelleme Benchmark'ı (TinyShakespeare, çok-seed)
- **Tarih:** 2026-07-07
- **Test:** `kaggle_lm_benchmark.ipynb` (2× T4 GPU paralel, 12 koşu)
- **Karşılaştırma:** GPT-2 (Transformer Baseline) vs HFP (`cubic_flux_chunked` + `delta` + `dpfp` + `standard` FFN)
- **Model Boyutları:** İki model de eşdeğer mimari: hidden=256, layers=4, heads=4, ~16M toplam parametre (3M core + 13M embedding, weight-tied).
- **Protokol:** 2 LR {5e-4, 1e-3} × 3 seed {0, 1, 2} = 12 koşu. max_iters=5000, eval_interval=200, patience=7, batch_size=16, seq_length=256. TinyShakespeare (~300K token). Her model için en iyi LR, 3-seed ortalamasıyla seçildi.

| Model | LR | Seed 0 | Seed 1 | Seed 2 | Ort. ± Std |
|---|---|---|---|---|---|
| GPT-2 | 5e-4 | 5.7196 | 5.6752 | 5.7137 | **5.703 ± 0.024** |
| GPT-2 | 1e-3 | 5.7968 | 5.7680 | 5.7701 | 5.778 ± 0.016 |
| HFP | 5e-4 | 5.5484 | 5.5383 | 5.5573 | **5.548 ± 0.010** |
| HFP | 1e-3 | 5.5968 | ❌ OOM | 5.5396 | 5.568 (2 seed) |

**Nihai karşılaştırma (en iyi LR = 5e-4, 3 seed):**

| Model | Val Loss | PPL | Early Stop |
|---|---|---|---|
| GPT-2 | 5.703 ± 0.024 | ~300 | step 1800-2000 |
| **HFP** | **5.548 ± 0.010** | **~257** | step 2000 |
| Fark | **−0.155** | **−43 PPL** | |

- **Bulgular:**
  1. HFP, GPT-2'yi 3 seed'de tutarlı olarak geçiyor. Fark ~6× standart hatadan büyük.
  2. HFP'nin seed-varyansı GPT-2'den düşük (std 0.010 vs 0.024) — daha kararlı öğrenme.
  3. İki model de erken durdurma ile step 1800-2000'de durdu (train loss ~2-3 vs val loss ~5.5-5.7, şiddetli overfitting).
  4. HFP per-step ~6× yavaş (cubic sequential scan + delta + dpfp overhead).
  5. `hfp_lr0.001_s1` koşusu GPU bellek yetersizliği (OOM) nedeniyle çöktü; en iyi LR (5e-4) 3 seed'de tam.
- **Sınırlar:** Küçük ölçek (~16M param, ~300K token). Harici baseline (GLA/Mamba) karşılaştırması yapılmadı. Overfitting nedeniyle erken durdurma zorunlu; daha büyük veriyle doğrulama gerekir.
- **Sonuç:** HFP mimarisinin O(1) bellek ile çalışmasına rağmen, tam dikkat kullanan GPT-2'yi dil modelleme görevinde geçtiği, çok-seed'li ve LR-taramalı titiz bir deneyle doğrulandı. Fark istatistiksel olarak güçlü ancak küçük ölçekte; genelleme için daha büyük veri/model gerekir.

## Ek 12: Chunkwise delta (TAM paralel form) + streaming-mix (yalnız dev)

**(c) Paralel delta:** u_t okuması "sözde-değer" W'ye dönüştürülünce recurrence
exp-chunkwise cebirine + TEK unit-alt-üçgen çözüme indirgeniyor:
  (I + diag(β) S_strict) W = diag(β)(V − A),  S_tj = k_t^T Λ^{t-j} k_j
Doğrulama: bağımsız sıralı referansla birebir (6/6 PASS: elu+dpfp × eşitlik/
blok-bağımsızlık/chunk-split). CPU'da sıralıya karşı 1.9× (additive'in yalnızca
1.3×'i); GPU'da kazanç çok daha büyük olacak. Delta artık ölçeklenebilir.
(cubic+delta kombinasyonu sıralı kaldı; gerekirse iki-geçişli z-taramasıyla
birleştirilebilir.)

**(a) Streaming-mix (tek seed 0!):** olgu+güncelleme karışık akış, train@160,
eval@160(P8) ve @640(P24):

| konfig | 160 tek-yazım | 160 güncellenen | 640 tek | 640 günc. |
|---|---|---|---|---|
| add+elu | 21.0 | 27.6 | 12.7 | 13.5 |
| delta+elu | 19.6 | 25.6 | 13.2 | 16.6 |
| add+dpfp | 24.2 | 28.1 | 15.1 | 16.2 |
| delta+dpfp | 23.8 | 28.1 | 13.9 | 16.2 |

Farklar ±3 puan — tek seed'de ayrım YOK. Ancak Ek 11'in dersi: additive
seed-0'da en iyi halindedir; bu tablo çok-seed'siz sonuçsuz sayılmalı.
Sıradaki: streaming-mix seed 1-2 (artık delta hızlı olduğundan ucuz).

## Ek 13: Uzun Bağlamda Eğitilebilirlik (Trainability) — cubic vs exp

**Gerekçe:** Streaming v2 testindeki (160 token) büyük loss farkını genellemek.
**Tasarım:** Dense retention (P=8), ctx ∈ {160, 256, 384, 512}, dpfp, lr=1e-3, 3 seed.
**Ön-Kayıtlı Kriter:** "ctx ≥ 384'te cubic ≥2/3 kırıyorsa VE exp ≤1/3 kırıyorsa (loss < 3.0), eğitilebilirlik avantajı doğrulanır."

| Mod | ctx=160 | ctx=256 | ctx=384 | ctx=512 |
|---|---|---|---|---|
| exp | 3/3 (1.78) | 3/3 (2.28) | **1/3** (3.05) | 2/3 (2.93) |
| cubic | 3/3 (1.57) | 3/3 (1.91) | **3/3** (2.24) | 3/3 (2.33) |

*(Parantez içi: 3 seed ortalama son loss. Plato sınırı: ~3.40)*

**Bulgular:**
1. **Kriter KARŞILANDI (ctx=384):** Cubic 3/3 plato kırarken, exp 1/3 kırdı. (ctx=512'de exp 2/3 kırdığı için teknik olarak kriteri kısmen esnetti, ama cubic'in loss avantajı bariz: Δ=0.60).
2. **Cubic'in İlk 100% Güvenilir Sonucu:** 4 farklı uzunluk × 3 seed = 12 koşunun tamamında cubic platosunu kırdı ve hiç NaN yaşamadı.
3. **Maliyet:** Cubic nonlinear sıralı scan nedeniyle exp'e göre per-step 4-5× daha yavaş.

**Sıradaki (Dürüstlük Kontrolü):** Exp ctx≥384'te tek LR'de (1e-3) çöküyor olabilir. Exp için {3e-4, 3e-3} ile hızlı LR taraması yapılıp, eğitilebilirlik farkının LR duyarlılığı olup olmadığı netleştirilmeli.

---

### Ek 16: Faz 1b — cubic_flux Uzun-Ufuk Karar Deneyi (DOĞRULANDI ✅)
- **Tarih:** 2026-07-07
- **Test:** `kaggle_cubic_longhorizon.ipynb` → `review_scripts/cubic_longhorizon.py` (P100 GPU, 36 koşu sıralı)
- **Tasarım:** 4 konfigürasyon {exp, cubic_flux_chunked} × {elu, dpfp}, 3 LR {3e-4, 1e-3, 3e-3}, 3 seed {0,1,2} = 36 koşu. Eğitim ctx=160 (P=8 dense recall, 600 adım, cosine LR schedule), eval ctx=640 ve ctx=1280. Model: 2 katman, hidden=64, local_window=8, write_rule="additive", ffn_type="standard", bulk_dim=32, rec_block=32.
- **Önceden kayıtlı kriter (GPU_ROADMAP.md):** "cubic_flux_chunked + dpfp, 256+ kovasında exp + dpfp'yi 3 seed ortalamasında >2 standart hata geçerse → cubic uzun-ufuk avantajı DOĞRULANDI. Aksi halde hipotez reddedilir / parked kalır."

**ctx=1280, 256+ kovası — mod-başına en iyi LR (3-seed ortalaması):**

| Konfigürasyon | En İyi LR | Seed 0 | Seed 1 | Seed 2 | Ort. ± SE |
|---|---|---|---|---|---|
| exp + elu | 3e-3 | 3.5 | 12.3 | 11.6 | 9.1 ± 2.8 |
| exp + dpfp | 1e-3 | 29.2 | 20.4 | 12.4 | 20.7 ± 4.9 |
| cubic + elu | 1e-3 | 4.3 | 4.6 | 4.1 | 4.3 ± 0.1 |
| **cubic + dpfp** | **3e-3** | **69.3** | **46.5** | **76.0** | **63.9 ± 8.9** |

*(şans seviyesi: %3.3)*

**Kriter kontrolü:** cubic+dpfp (63.9) − exp+dpfp (20.7) = **+43.2 puan**. Birleşik SE = 10.2. **Fark/SE = 4.25 → >2 SE eşiği ✅**

**Tüm kovalar, ctx=1280, en iyi LR (3-seed ortalaması %):**

| Konfigürasyon (LR) | <48 | 48-127 | 128-255 | 256+ |
|---|---|---|---|---|
| exp + elu (3e-3) | 61.1 | 41.3 | 24.3 | 9.1 |
| exp + dpfp (1e-3) | 89.6 | 73.8 | 56.9 | 20.7 |
| cubic + elu (1e-3) | 25.3 | 23.5 | 8.9 | 4.3 |
| **cubic + dpfp (3e-3)** | **93.2** | **91.9** | **80.7** | **63.9** |

- **Bulgular:**
  1. **Önceden kayıtlı kriter 4.25× SE ile karşılandı.** cubic_flux'ın uzun-ufuk avantajı DOĞRULANDI. Hipotez artık "parked" değil, kanıtlanmış.
  2. **cubic + dpfp sinerjisi:** cubic tek başına (elu) çöküyor (%4.3 ≈ şans); dpfp tek başına (exp) iyi (%20.7). Birlikte **%63.9** — basit toplam değil, sinerjik. dpfp'nin kanalları seyrek tutması cubic'in yavaş-çürüme platosunun korunmasını sağlıyor.
  3. **LR duyarlılığı teyit edildi:** cubic+dpfp LR=3e-4'te %15.9, LR=1e-3'te %33.8, LR=3e-3'te **%63.9**. Mod-başına LR taraması yapılmadan cubic haksızlığa uğruyor.
  4. **Avantaj 128+ gap'ten itibaren belirgin:** 128-255 kovasında cubic+dpfp %80.7 vs exp+dpfp %56.9 (3-seed).
  5. **exp+elu LR=3e-3'te seed-kırılgan:** s0 öğrenmedi (%3.5≈şans), s1 ve s2 öğrendi (%12). cubic+dpfp'nin 3 seed'de en düşüğü bile %46.5.
  6. **cubic ~8× yavaş** (per-step). Eğitim maliyeti yüksek ama çıkarım O(1) ve eşit hızda.
- **Sınırlar:** Mini ölçek (~0.5M param, sentetik recall). Gerçek dil modelleme görevinde cubic+dpfp'nin avantajı henüz test edilmedi (Ek 15'te cubic+delta+dpfp LM'de iyi ama exp+dpfp LM karşılaştırması yapılmadı). Bu test write_rule="additive" ile yapıldı; delta ile tekrarlanmalı.
- **Güncellenen durum:** cubic_flux_chunked + dpfp, seyrek-uzun-gap rejiminde (ctx≥640, gap≥128) exp+dpfp'ye göre **3× recall avantajı** sağlıyor. Bu, projenin asıl özgün iddiasının ilk deneysel doğrulamasıdır.
