# AGENTS.md — HFP (Hyper-Flux Projection)

Bu dosya jcode ve diğer kodlama ajanları için proje rehberidir. jcode her oturum
başında repo kökündeki `AGENTS.md`'yi otomatik okur. Bir işe başlamadan önce bu
dosyayı ve `README.md` + `NASIL_CALISTIRILIR.md`'yi dikkate al.

> ⚠️ **EN ÖNEMLİ İKİ KURAL — asla ödün verilmez:**
> 1. **BİLİMSEL DÜRÜSTLÜK** (aşağıdaki bölüm). Hız ya da "işi bitirmiş görünmek"
>    uğruna sonuç uydurulmaz, test hilesi yapılmaz. Emin değilsen dur ve söyle.
> 2. **LİSANS & FİKRİ MÜLKİYET KORUMASI (AGPL-3.0).** Bu proje **AGPL-3.0**
>    lisanslıdır ve bu kritiktir. Aşağıdaki "Lisans" bölümüne harfiyen uy;
>    lisans başlıkları, telif ve copyleft yükümlülükleri asla kaldırılmaz veya
>    zayıflatılmaz. Fikir/kod çalınmasına karşı azami dikkat.

---

## Proje nedir

**HFP — Hyper-Flux Projection**: deneysel bir causal language-model mimarisi.
Pencereli yerel attention (`windowed local attention`) ile katman-başına
**recurrent bellek** (`M ∈ ℝ^{H×H}`, `z ∈ ℝ^H`) birleştirir. Çıkarım-zamanı state
bağlam uzunluğundan **bağımsız (O(1) bellek)** — büyüyen KV-cache yerine sabit
boyutlu state. Uzun menzilli bilgi bu recurrent bellekten akmak zorundadır.

Ayırt edici özellik **retention law** (unutma yasası): standart üstel (`exp`)
decay'in yanında, fizik makalelerinden türetilen **cubic-plateau decay**
(`cubic_flux`) uygulanır ve tek bir flag ile geçiş yapılır.

**Bu bir araştırma/yayın projesidir** (bkz. `docs/osf_companion.pdf`,
`docs/paper3_ml_architecture.tex`, `RESULTS.md`). Kod, çalıştırılıp doğrulanacak
ML iddiaları üretir; süslü değil, ölçülebilir sonuç önemlidir.

## Vizyon & yön (tasarım kararlarını etkiler)

Aşağıdakiler uzun vadeli hedeflerdir; tasarım/uygulama tercihlerinde bunları
gözet (ama bir **hedef**tir, kanıtlanmış bir iddia değildir — abartma):

- **Cihaz-içi (on-device) entegrasyon KRİTİK.** HFP'nin O(1) sabit-bellek avantajı
  tam da uç/gömülü ve cihaz-içi çalıştırma içindir. Bu yüzden: düşük RAM/VRAM ayak
  izini koru, ağır/gereksiz bağımlılık ekleme, quantization ve dışa aktarıma
  (ONNX/TorchScript/GGUF vb.) uygun, temiz ve taşınabilir inference yolu hedefle.
  Bir değişiklik cihaz-içi çalıştırmayı zorlaştırıyorsa bunu açıkça belirt.
- **İleride API hizmeti olasılığı.** Eğitimli modellerin entegrasyonu bitince
  doğrudan bir API hizmeti sunulması düşünülüyor. Bu yüzden inference arayüzünü
  net, kararlı ve iyi belgelenmiş tut; yapılandırma (config) ve model
  yükleme/servis yollarını temiz ayır. (Not: Bu ticari yön AGPL-3.0 copyleft
  yükümlülükleriyle uyumlu planlanmalı — bkz. "Lisans" bölümü; şüphede kullanıcıya sor.)

Bu hedefler mevcut deneysel sonuçları değiştirmez; yalnızca yol ayrımlarında
tercih önceliğidir.

---

## Ortam / Kurulum

- **Python 3.10** (CI ve `.pyc` cache 3.10 ile üretiliyor — bu sürümü hedefle).
- Bağımlılıklar `requirements.txt`'te; şununla kur:

```bash
pip install -r requirements.txt
```

- CPU-only torch (GPU yoksa, CI'daki gibi):
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- GPU'lu ağır deneyler için `notebooks/` altındaki Colab/Kaggle defterleri kullanılır.

---

## İLK İŞ HER ZAMAN: smoke test

Herhangi bir değişiklikten önce ve sonra çalıştır:

```bash
python smoke_test.py            # CPU, ~1 dk
```

Bu, dört değişmezi mühürler: bellek parametreleri gradyan alıyor (T2), MQAR
loss'u sonlu (T3), recurrence matematiği chunk-bağımsız (T4). **T2/T4 geçmeden
hiçbir benchmark sonucuna güvenme.**

Daha derin doğrulama (chunk-tutarlılık, causal doğruluk, kübik ODE kesinliği):

```bash
python review_scripts/verify_claims.py
```

CI (`.github/workflows`) push/PR'da `smoke_test.py` + `verify_claims.py` +
tüm kaynakların byte-compile'ını çalıştırır. **Bu ikisi geçmeden işi bitmiş sayma.**

---

## Sık kullanılan komutlar

Associative recall — asıl bellek yeteneği testi (MQAR):
```bash
python run_experiment.py --task recall --steps 1500 --context 128 --pairs 8      # CPU
python run_experiment.py --task recall --steps 3000 --context 512 --pairs 16     # GPU
```

Gerçek perplexity (TinyShakespeare LM):
```bash
python run_experiment.py --task lm --steps 3000 --seq 256
```

Standart eğitim (checkpoint üretir):
```bash
python train.py --model hfp --optimizer adamw --max_iters 3000
python train.py --model hfp --optimizer thermodynamic --max_iters 3000   # opsiyonel A/B
```

VRAM O(1) kanıtı:
```bash
python eval_memory_scaling.py
```

Önemli flag'ler: `--decay_mode {exp,cubic_flux,cubic_flux_chunked}`,
`--key_feature_map {elu,dpfp}`, `--local_window` (recall testinde 32 kritik —
bilgi yalnızca bellekten aksın diye), `--seed`.

---

## Repo yapısı

- `hfp/` — asıl paket. `hfp/core/` (state, config, utils, physics_optimizers),
  `hfp/models/` (modeling_hfp, configuration_hfp, grafting).
- `run_experiment.py`, `train.py`, `smoke_test.py`, `eval_*.py` — giriş noktaları.
- `review_scripts/` — bağımsız doğrulama/analiz scriptleri.
- `notebooks/` — Colab/Kaggle deney defterleri.
- `hf_upload/` — HuggingFace yayın paketi (`modeling_hfp.py` vb. buraya kopyalanır).
- `docs/` — `docs/tr`, `docs/internal_tr` (Türkçe notlar), `.tex` makale.
- `_legacy_reference/` — eski/terk edilmiş kod; **referans amaçlı, import etme.**

---

## Bilimsel dürüstlük (ZORUNLU — pazarlık konusu değil)

Bu projede amaç "modelin iyi görünmesi" değil, **doğru olanı bulmaktır**. Olumsuz
ya da sıfır sonuç da tam bir sonuçtur ve aynı özenle raporlanır. Ajanlar aşağıdaki
kurallara istisnasız uyar:

**1. Veri ve sonuç uydurma yok.**
- Hiçbir sayı, metrik, grafik ya da tablo çalıştırılmadan yazılmaz. Her rapor
  edilen değerin ardında, o oturumda gerçekten koşturulmuş bir komut olmalı.
- Beklenen/tahmini sayıyı gerçek sonuçmuş gibi sunma. Tahminse "tahmin" de.
- Log, çıktı ya da checkpoint'i elle düzenleyip "sonuç buymuş gibi" gösterme.

**2. Testlerde hile yok.**
- Testi geçirmek için testi zayıflatma, atlama (`skip`), eşiği gevşetme ya da
  assert'i silme. Test kırmızıysa **kod düzeltilir, test değil.**
- `smoke_test.py` ve `review_scripts/verify_claims.py` gerçek değişmezleri
  koruyan bekçilerdir; bunları by-pass etmek, hardcode'lu "beklenen" değer
  gömmek ya da rastgeleliği sonucu sabitleyecek şekilde ayarlamak yasaktır.
- Seed'i yalnızca tekrarlanabilirlik için sabitle; "iyi sonuç veren seed'i seçmek"
  (seed-hacking / cherry-picking) manipülasyondur.

**3. Sonuç manipülasyonu yok.**
- Metriği güzelleştirmek için değerlendirme kümesini eğitim kümesine sızdırma,
  görevi kolaylaştıracak şekilde gizlice ayar kaçırma yok. Özellikle recall
  testinde `--local_window` kısıtı korunmalı — bilgi yalnızca recurrent bellekten
  akmalı; attention'a sızıntı sonucu şişirir ve bu geçersizdir.
- `exp` vs `cubic_flux` karşılaştırmasında **iki mod dışındaki her şey birebir
  aynı** tutulur. Bir tarafa avantaj sağlayacak farklı hiperparametre = geçersiz
  karşılaştırma.
- Aykırı/uygunsuz sonucu sessizce atma. Anomali varsa raporla ve araştır.

**4. Belirsizlik ve sınırlar açıkça belirtilir.**
- Bir iddia kanıtlanmadıysa "kanıtlandı" deme. `README.md`'deki "Honesty note"
  ile aynı çizgi: fizik analojisi modeli kanıtlamaz, model fiziği kanıtlamaz.
- Sonucu `NASIL_CALISTIRILIR.md`'deki dürüst-okuma kurallarıyla yorumla:
  "chunked ≫ şans VE reset ≈ şans" değilse, bunu olumlu sonuç gibi sunma.
- Örneklem küçükse, tek koşuysa, varyans yüksekse — bunu açıkça yaz.

**5. Şeffaflık ve tekrarlanabilirlik.**
- Her sonucu üreten tam komutu (flag'ler, seed, adım sayısı dahil) sonuçla
  birlikte ver ki başkası aynısını koşabilsin.
- Ne yaptığını, neyi değiştirdiğini ve neden değiştirdiğini açıkça anlat; sessiz
  "düzeltmeler" yapma.
- Emin olmadığın, doğrulayamadığın ya da başarısız olan şeyi gizleme — açıkça
  bildir. Başarısızlığı raporlamak, gizlemekten her zaman iyidir.

**6. Disiplin.**
- Bellek yolunu ya da matematiği değiştiren her değişiklikten sonra
  `smoke_test.py` + `verify_claims.py` yeniden koşulur; ikisi de yeşil değilse iş
  bitmemiştir.
- İstenmeyen kapsam genişletme yok; talep edileni yap, yan etkileri bildir.

> Bu kurallardan biriyle "işi tamamlamak" arasında çatışma olursa, **kurala uy ve
> durumu kullanıcıya bildir.** Hilesiz eksik bir sonuç, hileli tam bir sonuçtan
> üstündür.

## Diğer konvansiyonlar

- Kod değişikliğini repo dilinde yorumla (Türkçe teknik terimler + İngilizce kod
  standart). Mevcut stile uy; yeni bağımlılık eklemeden önce gerekçelendir.
- `hf_upload/` ve `hfp/` arasında kopyalanan dosyalar (örn. `modeling_hfp.py`,
  `configuration_hfp.py`) senkron tutulmalı — birini değiştirdiysen diğerini kontrol et.
- `_legacy_reference/` yalnızca referans; import etme, sonuç üretmede kullanma.

---

## Lisans & Fikri Mülkiyet (AGPL-3.0 — KRİTİK)

Bu proje **GNU AGPL-3.0** altında lisanslıdır (bkz. `LICENSE`, README başlığındaki
`license: agpl-3.0`). Bu, fikri korumanın ana aracıdır; ajanlar buna istisnasız uyar:

- **Lisansı asla kaldırma/zayıflatma.** Dosyalardaki lisans başlıkları, telif
  bildirimleri ve `LICENSE` dosyası korunur; silinmez, "MIT/BSD" gibi daha gevşek
  bir lisansla değiştirilmez, "public domain" denmez.
- **Yeni dosyalar da AGPL-3.0'dır.** Oluşturulan her kaynak dosya projenin lisansı
  altındadır; aksini ima eden bir başlık ekleme.
- **AGPL copyleft'ine dikkat.** AGPL, ağ üzerinden erişilen türevlerde bile
  kaynak-paylaşımını zorunlu kılar. Kod tabanına eklenen bağımlılıklar lisans
  uyumlu olmalı; **AGPL ile bağdaşmayan (ör. tescilli/kapalı) kod kopyalanıp
  eklenmez.** Yeni bağımlılık önerirken lisansını belirt.
- **Dış kod kopyalama = kaynak belirt.** Başka bir yerden alınan hiçbir kod
  kaynağı ve lisansı belirtilmeden gömülmez; lisansı belirsizse ekleme, sor.
- **Fikri sızıntıya karşı disiplin.** Projeye özgü yöntemler, henüz yayınlanmamış
  fikirler, iç planlama notları (`docs/internal_tr/`) dışarı sızdırılmaz; harici
  servislere/dosyalara özgün algoritma detayları gereksiz yere kopyalanmaz.
- **Atıf zinciri korunur.** `README.md`'deki "Honesty note" ve fizik-makale
  atıfları gibi köken/atıf ifadeleri değiştirilmeden korunur.
- Şüphe varsa (bir kodun lisansı, bir bağımlılığın uyumu, bir metnin kökeni)
  **dur ve kullanıcıya sor** — yanlış lisanslama geri dönülmesi zor bir hatadır.
- **KARAR KAPILARI:** Şu üç durumdan biri oluşursa İŞE BAŞLAMADAN ÖNCE
  `docs/internal_tr/LISANS_KARAR_REHBERI.md` açılır ve checklist uygulanır:
  (1) dışarıdan bir PR/katkı geliyorsa → CLA'sız merge YASAK;
  (2) HF'ye model ağırlığı yükleniyorsa → ağırlık lisansı + Qwen atfı kontrolü;
  (3) API/ticarileşme veya büyük yayın hazırlığı → patent/çift-lisans kararı.
  Bu rehber projenin en kritik hatırlatıcısıdır; kullanıcıya da hatırlat.

## Git

- Remote: `https://github.com/kayra-hn/HFP.git` — ana dal `main`.
- CI `main`'e push ve PR'da çalışır. Yeşil CI olmadan merge etme.

---

## Ajan için hızlı başlangıç yordamı

1. `AGENTS.md` + `README.md` + `NASIL_CALISTIRILIR.md`'yi oku.
2. `python smoke_test.py` çalıştırıp temiz taban çizgisini doğrula.
3. İstenen değişikliği yap; ilgili `run_experiment.py` / `train.py` komutunu koş.
4. `python smoke_test.py` ve `python review_scripts/verify_claims.py` ile mühürle.
5. Ürettiğin sayıları dürüstçe, çalıştırdığın komutla birlikte raporla.
