# Ajan brifingi — denetim ajanı, 3. tur

*Hazırlanma: 2026-08-02, HEAD `fe35457`. Ajan bu repoyu **proje kökü** olarak
açıyor; tüm yollar repo köküne görelidir.*

---

## Prompt (kopyala-yapıştır)

Bu repoda teknik denetim ve bakım yapıyorsun. Proje kökündeki `AGENTS.md`'yi
oku — **iki kural ödünsüz:** bilimsel dürüstlük ve lisans/IP koruması.

Bu senin **üçüncü turun**. İki tur rapor verdin; ikisinde de değerli iş çıktı ve
ikisinde de doğrulanmamış bir iddia geçti. Aşağıdakiler kanıtıyla yazılı, suçlama
için değil kural çıkarımı için.

**Bu bir araştırma reposu.** Negatif sonuçlar, erratum'lar ve dürüstlük uyarıları
kasıtlıdır; "temizlenmez". İki ajan daha aynı repoda çalışıyor.

---

## ⚠ BU BÖLÜM GERİ ÇEKİLDİ (2026-08-02) — hata bu brief'i yazandaydı

> **Aşağıdaki iddia YANLIŞTI ve 3. tur görevlerinin bir kısmını gereksiz yere
> tetikledi. Kayıt için bırakılıyor, uygulanmıyor.**
>
> Bu brief, 2. turun `hf_upload` analizini "karşılaştırma çiftleri karışmış"
> diye işaretledi ve `bulk_trigger_decoder` için 528, `hfp_utils` için 232
> satır sapma olduğunu iddia etti. **Bu sayılar CRLF/LF artefaktıydı.**
>
> `hfp/core/bulk_trigger_decoder.py` ve `hfp/core/hfp_utils.py` **CRLF**,
> `hf_upload/` kopyaları **LF** satır sonu kullanıyor. `diff` bu durumda bütün
> dosyayı değişmiş sayar: 264×2 = 528, 116×2 = 232 — tam olarak dosya
> uzunluğunun iki katı. Ölçen taraf satır sonlarını kontrol etmedi.
>
> **Satır sonu normalize edilince gerçek matris (md5 ile de doğrulandı):**
>
> ```
>                      hf_upload↔hfp/core   hf_release↔hfp/core
> bulk_trigger_decoder        0                    2
> hfp_utils                   0                    0
> hfp_bulk_state              0                  210
> hfp_config                  0                   15
> ```
>
> Yani 2. turun "1 satır kozmetik fark" ve "`hfp_utils` bit-bit aynı" ifadeleri
> **doğruydu.** GÖREV 1 (raporu düzelt) hatalı bir gerekçeyle verildi; o görevle
> `DENETIM_2026-08.md`'ye yazılan düzeltme notu **geri alınmalıdır** —
> doğru bir ifadeyi "düzeltiyor".
>
> **Ayakta kalan bulgu:** `hf_release/hfp_bulk_state.py` ve `hfp_config.py`
> gerçekten 210 ve 15 satır geride. 1. turun `hf_upload/` güncellemesi yayın
> paketinin iki kopyasını birbirinden ayırdı.
>
> **Ve yeni gerçek bulgu:** `hfp/core/` içinde satır sonları karışık
> (`bulk_trigger_decoder.py`, `hfp_utils.py` CRLF; `hfp_bulk_state.py` LF).
> Bütün karışıklığın kaynağı bu. `.gitattributes` ile normalize edilmeli.

**Kural (geçerliliğini koruyor):** bir fark rapor ederken **hangi iki yol
arasında** olduğunu yaz, ve **satır sonu farkını içerik farkından ayır.**
`A ↔ B: N satır (satır sonu normalize)` formatı dışında fark rapor etme.

---

## GÖREV 1 — Raporu düzelt

`DENETIM_2026-08.md` 1b'yi yukarıdaki matrisle düzelt, ve "birinci turun
kopyalaması `hf_upload/` ile `hf_release/` arasında yeni sapma yarattı" bulgusunu
ekle. Eski yanlış metni silme — **düzeltme notu olarak bırak** (bu projede
erratum silinmez, yazılır).

---

## GÖREV 2 — Üç kopya sorununu kökten çöz

**Durum:** aynı kodun üç kopyası var — `hfp/core/` (kanonik), `hf_upload/`,
`hf_upload/hf_release/` — ve üçü de birbirinden farklı. Elle senkron iki turda
denendi, ikinci turda daha kötü hâle geldi. Bu bir sınıf sorunu, tek tek dosya
sorunu değil.

**2a — Önce farkı ANLA.** 528 ve 232 satırlık farkın **ne olduğunu** çıkar.
Kritik soru: bu fark salt sürüklenme mi (kanonik ilerledi, kopya geride kaldı),
yoksa içinde **HF paketine özel kasıtlı uyarlamalar** var mı? (Örnek: HF'nin
`trust_remote_code` yoluna özgü import düzeni, `hfp` paketine bağımlılığı kesen
düzenlemeler, model kartıyla uyumlu isimlendirme.) Fark tipine göre sınıflandır:
**SÜRÜKLENME** / **KASITLI UYARLAMA** / **BELİRSİZ**.

Çıktı: `docs/internal_tr/HF_UPLOAD_ANALIZI.md` — dosya bazında tablo, her fark
öbeği için sınıf ve gerekçe.

**2b — Kararı buna göre ver.**

- Hiç **KASITLI UYARLAMA** yoksa → `scripts/build_hf_release.py` yaz: yayın
  paketini `hfp/`'den üretir, kopyalar silinir, sapma matematiksel olarak
  imkânsız hale gelir. Sonra `hf_upload/` ve `hf_release/` kopyalarını **silme
  önerisi** getir, ama **silme** — o kullanıcının onayına bağlı.
- **KASITLI UYARLAMA varsa** → build script'i o uyarlamaları uygulayacak şekilde
  tasarlanmalı; uyarlamaları önce belgele. Anlamadığın bir uyarlama varsa
  **dur ve sor.**
- **BELİRSİZ** kalan varsa → o dosya için karar ertelenir, raporla.

`hf_upload/` `.gitignore:58`'de — **takipsiz.** Oraya yazma. Build script'i
`scripts/` altına yaz (takipli).

---

## GÖREV 3 — `load_checkpoint_safe`'i gerçekten güvenli yap

`review_scripts/verify_graft.py` içindeki `load_checkpoint_safe` şu an yalnızca
**isim kesişimine** bakıyor (`m_keys ∩ s_keys`) ve 0 ise hata veriyor. Bu,
kaynağı olan `notebooks/memory_capability_v30.ipynb` hücre 7'den zayıf. Orada
korkulan üç senaryo vardı ve ikisi hâlâ yakalanmıyor:

1. isimler eşleşmiyor → hiçbir şey yüklenmiyor  ✔ yakalanıyor
2. isimler eşleşiyor ama **şekiller tutmuyor** → o tensörler sessizce atlanıyor  ✘
3. yükleme "başarılı" görünüyor ama değerler modele **geçmemiş**  ✘

Ek olarak §30 bir **eğitim izi** kontrolü yapıyordu: `out_gain` init 0.1'den
sapmış mı? Sapmadıysa ağırlıklar eğitimsizdir ve "hafıza yok" sonucu geçersizdir.

Yap: `load_checkpoint_safe`'i şunları döndürecek ve gerekirse hata verecek
şekilde genişlet — isim eşleşmesi, **şekil eşleşmesi**, **`torch.equal` ile
bit-bit değer doğrulaması**, ve opsiyonel bir eğitim-izi kontrolü (parametre adı
+ beklenen init değeri parametre olarak alınsın; modüle özel isim gömme).
`verify_graft.py`'ye bu üç senaryoyu ayrı ayrı sınayan testler ekle — özellikle
**şekil uyuşmazlığının yakalandığını** gösteren bir test.

---

## GÖREV 4 — Tekrarlanabilirlik raporunu çalışan bir şeye çevir

`docs/internal_tr/TEKRARLANABILIRLIK_2026-08.md` 17 bölümü KOŞULABİLİR diye
işaretledi. Rapor iyi ama bir belge; asıl değer koşan bir şey olması.

`reproduce.py` yaz (repo kökü). O 17 bölümün her biri için **çok kısa
parametrelerle** üretim yolunu çalıştırır ve yolun ayakta olduğunu raporlar.

- Bu bir **yol** kontrolü, **sonuç** kontrolü değil. Sayıları doğrulama,
  eşiklerle karşılaştırma, `RESULTS.md`'deki değerlerle kıyaslama **yapma.**
  Amaç: "bu komut bugün hâlâ çalışıyor mu?"
- Her bölüm için: komut, çıkış kodu, süre. Tablo bas.
- `--section 12` gibi tek bölüm koşma yolu olsun; `--list` yolları listelesin.
- Toplam süre CPU'da makul kalsın; kalmıyorsa hangi bölümlerin ağır olduğunu
  raporla ve varsayılan dışında bırak.
- Koşmadan "çalışıyor" yazma. Kırılan varsa **düzeltme, raporla** — kırık bir
  üretim yolu Sınıf 2 olabilir (sonucun kendisini ilgilendirir).

CI'a **ekleme** — 30 dakika limiti var ve bu iş ona sığmaz.

---

## Sınıf ayrımı (aynen geçerli)

- **SINIF 1 mekanik** → uygula. Objektif, doğrulanabilir, yargı gerektirmez.
- **SINIF 2 iddia/bilim** → raporla, **uygulama.** Bir cümlenin `RESULTS.md`'deki
  ölçümü aşması, bir sayının kaynağının bulunamaması, bir üretim yolunun kırık
  olması.
- **SINIF 3 tasarım** → öner, **uygulama.** Davranışı ya da deney semantiğini
  değiştiren her şey.
- **Emin değilsen Sınıf 2 say ve uygulama.**

## Geçmiş turlardan gelen sert kurallar

- **Yeşil mühür, kapsamadığı şey hakkında hiçbir şey söylemez.** Uyguladığın her
  değişiklik için "bunu hangi doğrulama kapsıyor" sorusunu cevapla; kapsayan yoksa
  öyle yaz.
- **Takipsiz (gitignored) bir yolda yapılan değişiklik "uygulandı" sayılmaz.**
  (`hf_upload/`, `_legacy_reference/`, `_archive_old/`, `checkpoints/`)
- **Bir Sınıf 1 partisi bitince kombinasyonu doğrula**, tek tek değil.
  (1. turda iki commit birbirini kırmıştı.)
- **Bir fark rapor ederken hangi iki yol arasında olduğunu yaz.**
- **Bir değişmez tutmuyorsa kodu düzeltip geçme** — dur, raporla. Tutmayan bir
  değişmez yayınlanmış sonuçların geçerliliğini ilgilendirir.

## Dosya sahipliği — İHLAL ETME

| yol | sahip |
|---|---|
| `hfp/`, kök scriptler, `.github/`, `pyproject.toml`, `requirements.txt` | **SEN** |
| `review_scripts/verify_graft.py`, `reproduce.py`, `scripts/` | **SEN** |
| `docs/internal_tr/DENETIM_*.md`, `TEKRARLANABILIRLIK_*.md`, `HF_UPLOAD_ANALIZI.md` | **SEN** |
| `README.md`, `NASIL_CALISTIRILIR.md` | **SEN** (yalnız Sınıf 1) |
| `RESULTS.md` | GPU oturumu — **DOKUNMA**, hata bulursan raporla |
| `notebooks/` | GPU oturumu — **DOKUNMA** (okuyabilirsin) |
| `review_scripts/` (diğer dosyalar) | 2. ajan — **DOKUNMA** (okuyabilirsin) |
| `docs/` (yukarıda sayılmayanlar) | 2. ajan — **DOKUNMA** |
| `hf_upload/` | **hiç kimse** — takipsiz; yalnız oku ve raporla |

## Yapmayacakların

- `RESULTS.md`'yi düzenleme. Otorite deney kaydı.
- Takipsiz yollarda değişiklik uygulama.
- Negatif sonuçları, erratum'ları, dürüstlük uyarılarını "temizleme".
- Testi zayıflatma: eşik gevşetme, `skip`, assert silme, hardcode'lu "beklenen"
  değer gömme **yasak**. Test kırmızıysa kod düzeltilir.
- GPU/compute işi başlatma. Kota §36'ya ayrılmış. CPU serbest.
- `git push` yapma. Commit at, push kullanıcıya ait.
- Checkpoint (`.pt`) commit etme — `docs/internal_tr/LISANS_KARAR_REHBERI.md` Kapı 2.
- Yeni bağımlılık ekleme — öner, lisansını belirt, uygulama.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

## Bitirmeden önce

`python smoke_test.py` + `python review_scripts/verify_claims.py` +
`python review_scripts/verify_graft.py` — üçü de yeşil olmalı.

## Raporun başına

Kaç bulgu, sınıf dağılımı, **hangi alanlara hiç bakmadığın**, ve uyguladığın her
değişiklik için **onu hangi doğrulamanın kapsadığı**. Bakmadığın yeri "temiz"
diye raporlama.

## Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor).
Ne yapıldı + neden + hangi sınıf. Küçük ve ayrı commit'ler.

## Belirsizlikte

Dur ve sor. Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
