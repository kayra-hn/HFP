# Ajan brifingi — repo geneli teknik denetim (3. ajan)

*Hazırlanma: 2026-08-01, HEAD `8b95a97`. Bu, **üçüncü** paralel ajandır.
Diğer ikisi: GPU oturumu (§36, `RESULTS.md` + `notebooks/`) ve yayın/araç ajanı
(`review_scripts/`, `docs/`, ayrı `cache-audit` reposu).*

---

## Prompt (kopyala-yapıştır)

Sen HFP projesinde teknik denetim yapıyorsun: `C:\Users\yilma\Documents\HFP_Project`.
Önceden eğitilmiş bir LLM'in (Qwen2.5-1.5B) attention katmanlarının bir kısmını
O(1)-bellekli recurrent bir mekanizmayla değiştiren bir **araştırma** projesi.
AGPL-3.0. Sahibi Kayrahan Yılmaz.

**Bu bir araştırma reposu, ürün reposu değil.** Buradaki birçok "tuhaflık"
kasıtlıdır: negatif sonuçlar silinmez, erratum'lar durur, dürüst uyarılar
kalır. Bunları "temizleme".

**İki ajan daha aynı repoda çalışıyor.** Dosya sahipliğine harfiyen uy.

### Önce oku

1. `AGENTS.md` — **iki kural ödünsüz:** bilimsel dürüstlük, lisans/IP koruması.
2. `docs/internal_tr/BEKLEYEN_ISLER.md` — bilinen açık işler. #9, #10, #11
   senin kapsamında.
3. `README.md` + `NASIL_CALISTIRILIR.md` — iddia edilen davranış.
4. `RESULTS.md` — **okumak için, düzenlemek için değil.** Bir iddianın dayanağını
   ararken buraya bakacaksın.

---

## Bulguları ÜÇ SINIFA ayır. Sınıf ayrımı işin en önemli kısmı.

### SINIF 1 — MEKANİK (uygula)

Objektif, doğrulanabilir, yargı gerektirmeyen. Örnekler:

- Kırık/ölü kod, erişilemeyen dal, kullanılmayan import, yanlış yol
- **Docstring ile kod uyuşmazlığı** — özellikle env değişkeni adları ve
  varsayılanları (`CC_*`, `MP_*`, `LG_*`, `HFP_*`). Bu projede stdout'tan
  regex'le sayı kazıyan bir notebook, print formatı değişince sessizce
  bozulmuştu; benzerlerini ara.
- **`hf_upload/` ↔ `hfp/` senkron sapması** (BEKLEYEN #9). `hfp_bulk_state`,
  `bulk_trigger_decoder`, `hfp_config`, `hfp_utils` kanonikten sapmış (≤960
  satır diff). Karşılaştır, **hangi tarafın kanonik olduğunu belirle**, eşitle.
  Yayınlanan paketin kanonikten sapması gerçek bir doğruluk riski.
- `.gitignore` hijyeni; `tinyshakespeare.txt`'nin git'ten çıkarılması
  (BEKLEYEN #10: `git rm --cached`, dosya yerelde kalır)
- `_legacy_reference/` içinden **import edilip edilmediğini** kontrol et —
  `AGENTS.md` bunu yasaklıyor. Ediliyorsa raporla.
- CI kapsam boşlukları (`.github/workflows/ci.yml`): hangi giriş noktaları
  hiç çalıştırılmıyor? Öneri getir, ama CI'ı ağırlaştırma (30 dk limiti var).
- `pyproject.toml` / `requirements.txt` tutarsızlıkları, Python 3.10 hedefi

**Her Sınıf 1 düzeltmesinden sonra:** `python smoke_test.py` +
`python review_scripts/verify_claims.py`. İkisi yeşil değilse geri al.

### SINIF 2 — İDDİA / BİLİM (RAPORLA, UYGULAMA)

Bir cümlenin `RESULTS.md`'deki ölçümü aşması, bir çerçevenin abartılı olması,
bir sayının kaynağının bulunamaması. **Bunlara dokunma.** Her biri yargı
gerektirir ve yanlış bir "düzeltme" otorite kaydı bozar.

Raporla, düzeltme öner, **uygulama.** Dayanak bulamıyorsan "DESTEKSİZ" yaz;
"muhtemelen §X" yazma.

### SINIF 3 — TASARIM (ÖNER, UYGULAMA)

Davranışı ya da deney semantiğini değiştiren her şey. Örnek: bir eğitim
kaybının terimlerini ayırmak, bir sondanın üretimini değiştirmek, bir
varsayılanı değiştirmek. Bunlar **tarihsel karşılaştırılabilirliği kırar** —
§26-§29 arası sonuçlar aynı araçla üretildi.

Öneriyi gerekçesiyle yaz, **uygulama.**

---

## Çıktı

**1.** `docs/internal_tr/DENETIM_2026-08.md` — üç bölümlü rapor:

| # | sınıf | dosya:satır | bulgu | kanıt | eylem |
|---|---|---|---|---|---|

`kanıt` = koştuğun komut ve çıktısı, ya da diff, ya da `RESULTS.md` referansı.
Kanıtı olmayan bulgu rapora girmez.

**2.** Sınıf 1 düzeltmeleri uygulanmış hâlde, **küçük ve ayrı commit'ler**
(tek büyük commit değil — geri alınabilir olsun).

**3.** Raporun başına: kaç bulgu, sınıf dağılımı, ve **hangi alanlara hiç
bakmadığın**. Bakmadığın yeri "temiz" diye raporlama.

---

## Dosya sahipliği — İHLAL ETME

| dosya | sahip |
|---|---|
| `hfp/`, `hf_upload/` | **SEN** |
| `train.py`, `run_experiment.py`, `smoke_test.py`, `eval_*.py` | **SEN** |
| `.github/`, `.gitignore`, `pyproject.toml`, `requirements.txt` | **SEN** |
| `README.md`, `NASIL_CALISTIRILIR.md` | **SEN** (yalnız Sınıf 1) |
| `RESULTS.md` | GPU oturumu — **DOKUNMA**, hata bulursan raporla |
| `notebooks/` | GPU oturumu — **DOKUNMA** (okuyabilirsin) |
| `review_scripts/` | 2. ajan — **DOKUNMA** (okuyabilirsin) |
| `docs/` (`DENETIM_2026-08.md` hariç) | 2. ajan — **DOKUNMA** |

Sahibi olmayan bir dosyada hata bulursan: **raporla, düzeltme.**

## Yapmayacakların

- **`RESULTS.md`'yi düzenleme.** Otorite deney kaydı.
- **Negatif sonuçları, erratum'ları, dürüstlük uyarılarını "temizleme".**
  Bunlar kasıtlı. `README.md`'deki "Honesty note" ve atıf ifadeleri korunur.
- **Testi zayıflatma.** `smoke_test.py` ve `verify_claims.py` gerçek
  değişmezleri koruyan bekçilerdir; eşik gevşetme, `skip`, assert silme,
  hardcode'lu "beklenen" değer gömme **yasak**. Test kırmızıysa kod düzeltilir.
- **`_legacy_reference/`'ı düzeltme.** Terk edilmiş referans kod; import
  edilmediğini doğrula, içeriğine dokunma.
- **GPU/compute işi başlatma.** Kota §36'ya ayrılmış. CPU serbest.
- **`git push` yapma.** Commit at, push kullanıcıya ait.
- **Checkpoint (`.pt`) commit etme.** Ağırlık yayınlamak
  `docs/internal_tr/LISANS_KARAR_REHBERI.md` Kapı 2'yi tetikler.
- **Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.**
- **Yeni bağımlılık ekleme** — gerekiyorsa öner, lisansını belirt, uygulama.

## Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor).
Ne yapıldı + neden + hangi sınıf.

## Belirsizlikte

Bir bulgunun Sınıf 1 mi Sınıf 2 mi olduğundan emin değilsen, **Sınıf 2 say ve
uygulama.** Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
