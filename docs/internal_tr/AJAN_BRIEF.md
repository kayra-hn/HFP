# Ajan brifingi — confound teşhis aracı + yayın hattı

*İkinci ajana verilecek görev tanımı. GPU oturumu paralel olarak §36'yı koşuyor.
Hazırlanma: 2026-08-01, repo HEAD `e6dfcac`.*

---

## Prompt (kopyala-yapıştır)

Sen HFP projesinde çalışıyorsun: `C:\Users\yilma\Documents\HFP_Project`.
Önceden eğitilmiş bir LLM'in (Qwen2.5-1.5B) attention katmanlarının bir kısmını
O(1)-bellekli recurrent bir mekanizmayla değiştiren bir araştırma projesi.
Sahibi Kayrahan Yılmaz, tek başına, bedava GPU kotalarıyla çalışıyor.

**Paralel bir oturum aynı repoda GPU deneyi (§36) koşuyor.** Dosya sahipliği
aşağıda; ona harfiyen uy.

### Önce oku (bu sırayla)

1. `AGENTS.md` — çalışma kuralları. **İki kural ödünsüz:** bilimsel dürüstlük,
   lisans/IP koruması (AGPL-3.0).
2. `RESULTS.md` — **otorite deney kaydı.** Başındaki ERRATUM'u mutlaka oku,
   sonra **§30, §30a-c, §31, §33, §34** (hafıza organı hattı) ve **§36**
   (ön-kayıt). Asıl işin bunlardan çıkıyor.
3. `notebooks/memory_capability_v30.ipynb` — mevcut cache-reset makinesi.
   Özellikle hücre 2 (model+checkpoint kurulumu) ve **hücre 7** (yükleme
   doğrulaması). Aracın çekirdeği burada, dağınık hâlde.
4. `notebooks/retention_lm_scale_v36.ipynb` — aynı protokolün LM-kaybı versiyonu.

---

## GÖREV A — Confound teşhis aracını yaz (BİRİNCİL)

### Bulgu

Hibrit attention/O(1) modellerde ölçülen uzun menzilli geri getirme, **artık
attention katmanlarının KV-cache'i tarafından taşınıyor olabilir** — ve alanda
bu genelde cache açıkken ölçülüyor. Biz kendi modelimizde yakaladık: needle
sonuçları 512-16384 token mesafede "BULDU" diyordu; cache chunk başına
sıfırlanınca **her mesafede %0**, tek bir chunk sınırında bile (§30a-c).
Sebep aranırken üç eğitim müdahalesi denendi, hiçbiri değiştirmedi (§31, §33,
§34). Bulgu erratum'la kayda geçti.

### Yazacağın şey

`review_scripts/cache_confound_audit.py` — **başkalarının da kendi modellerine
uygulayabileceği**, tek başına çalışan bir teşhis aracı.

Üç koşulu koşar ve tabloyu basar:

| koşul | ne yapar | ne söyler |
|---|---|---|
| **A — cache açık** | standart değerlendirme | alanın genelde raporladığı sayı |
| **B — cache chunk başına sıfır** | yalnız recurrent state taşır | O(1) state'in gerçekte taşıdığı |
| **C — üst sınır** | tam attention, aynı sonda | sonda geçerli mi, model çözebiliyor mu |

Hüküm mantığı (araç bunu kendisi basmalı, kullanıcı yorumlamak zorunda kalmasın):

- **C düşükse** → sonda geçersiz ya da model görevi zaten çözemiyor.
  **Diğer hiçbir satır yorumlanamaz.** Önce bunu düzelt.
- **A yüksek, B ≈ şans** → **geri getirme cache'den geliyor.** O(1) bellek
  iddiası bu ölçümle desteklenmiyor.
- **A yüksek, B yüksek** → state gerçekten taşıyor. İddia destekli.
- **A ≈ B ≈ şans** → hiç geri getirme yok; ayrı bir sorun.

### Zorunlu bileşenler

**1. Checkpoint yükleme doğrulaması.** `notebooks/memory_capability_v30.ipynb`
hücre 7'yi araca taşı. Gerekçe: `strict=False` isim uyuşmazlığında **sessizce
hiçbir şey yüklemez**, ve o durumda B=%0 "hafıza yok"u değil "ağırlıklar
eğitimsiz"i gösterir. Araç bunu ayırt etmeden hüküm basmamalı. Kontroller:
tensor isim/şekil eşleşme sayısı, bit-bit değer karşılaştırması, ve eğitim izi
(bizde `out_gain` init 0.1'den sapmış mı).

**2. Üst sınır kolu — dikkat, mevcut kod BOZUK.**
`review_scripts/matched_probe.py` içindeki `probe_unmatched`, K=0'da tam olarak
`[tgt_k, v, tgt_k]` dizisini üretiyor — üç token, `local_window=8` içinde.
Yani bugüne kadar "üst sınır %100" diye okunan şey sadece yerel attention'ın
çalıştığını gösteriyor. **Araçta geçerli bir üst sınır kolu yok, yazman gerek.**
Doğru üst sınır: aynı sonda, aynı mesafe, ama bilgi tam attention üzerinden
erişilebilir. Bu, aracın en kritik parçası — onsuz her "%0" yorumlanamaz.

**3. Kendi sonucumuzla kendini doğrulama.** Araç, §30a-c'nin sonuçlarını
yeniden üretebilmeli. Bir `--selftest` yolu koy: bizim checkpoint'le koşulunca
A yüksek / B %0 / C %100 çıkmalı. Çıkmıyorsa araç bozuktur.
Not: checkpoint'ler `.gitignore`'da, repoda yok — selftest dosya yoksa
**atlanmalı ve öyle raporlanmalı**, sessizce geçilmemeli.

### Arayüz ve kapsam — dürüst ol

Aracı "her hibrit modelde çalışır" diye pazarlama. Gerçekçi kapsam: **bizim
modelimizde çalışır + başkasının takabileceği belgelenmiş bir arayüz.** En az
şu ikisi dışarıdan verilebilmeli: (a) cache'i sıfırlama fonksiyonu, (b) sonda
üretici. README'sinde neyin test edildiğini ve neyin edilmediğini açıkça yaz.

Saf Python + torch. Yeni bağımlılık ekleme (scipy dahil); gerekiyorsa gerekçelendir.

### Lisans — karar sende değil

Yeni dosyalar `AGENTS.md` gereği **AGPL-3.0**. Bir teşhis aracının yayılması
isteniyorsa AGPL caydırıcı olabilir; bu gerçek bir gerilim ama **çözümü sen
seçme.** Aracı AGPL başlığıyla yaz, gerilimi kullanıcıya not olarak bildir ve
`docs/internal_tr/LISANS_KARAR_REHBERI.md` Kapı 3'e işaret et.

---

## GÖREV B — Confound yazısı (İKİNCİL, A ile birlikte anlamlı)

`docs/drafts/kv_cache_confound.md`. Yazı ve araç birbirini tamamlıyor: yazı
"sizde de olabilir" der, araç "buyur, kontrol et" der. Ayrı ayrı zayıflar.

- Somut sayılar `RESULTS.md`'den alınır, uydurulmaz.
- **Kendi hatamızı merkeze koy** (§22 needle → §30 cache-reset → erratum).
  Yazının en güçlü kısmı bu; başkasını suçlayan değil kendini düzelten bir metin.
- İlgili hattı (Mamba-in-Llama, LoLCATs, MOHAWK, LAWCAT) **atıf** olarak an.
  "Bunlar hatalı" iddiası **kurma** — o iddianın kanıtı bizde yok, ve kurarsan
  yazının güvenilirliği gider.
- Sonunda araca yönlendir.

---

## GÖREV C — İddia denetimi (ÜÇÜNCÜL, A ve B bitince)

`README.md`, `docs/PROJECT_SUMMARY.md`, `docs/paper3_ml_architecture.tex`
içindeki her ampirik iddiayı satır satır `RESULTS.md` ile karşılaştır.

**Bilinen canlı yanlışlık:** §28a'nın atfı §29b'de (2026-08-01) düzeltildi ama
dış belgeler bunu içermiyor. Özet: §28a kendi metriğinde replike oldu (+0.4080
nat, 14/16, p=0.0005) **ama o metrik karışıktı** — B-chunk etiketleri hem
chunk-içi çiftleri hem cross-chunk hedefini içeriyor, P=6'da denetlenen 7
tokenın 6'sı chunk-içi. Ayrıştırınca chunk-içi +0.4298 (p=0.0008), cross-chunk
+0.2770 (t p=0.1103). Sonuç: §28a'yı **"cross-chunk carry kazancı"** diye sunmak
abartılı. Etkilenen yerler: `README.md` ~69-77 ve ~120, `PROJECT_SUMMARY.md`
~45-50, ve `.tex` (hiç elden geçmedi, erratum öncesi olabilir).

Çıktı: `docs/internal_tr/IDDIA_DENETIMI_2026-08.md`, tablo:
`| belge:satır | iddia | RESULTS referansı | durum | önerilen metin |`
`durum` ∈ {DOĞRU, BAYAT, ABARTILI, DESTEKSİZ, ÇELİŞKİLİ}. Dayanak bulamıyorsan
DESTEKSİZ yaz, "muhtemelen §X" yazma. **Düzeltmeleri onaysız uygulama.**

---

## Bilmen gereken bağlam

**Strateji belgeleri kullanıcının değil.** `YAYIN_PLANI.md`, `YOL_HARITASI_*.md`,
`DEVIR.md` §6 üçüncü şahısta yazılmış **asistan çıkarımlarıdır**; kullanıcı
hedeflerinin farklı olduğunu açıkça söyledi. Kullanıcı niyeti olarak alma.
Olgusal envanterleri kullanılabilir. `RESULTS.md` istisnadır — ölçüm kaydıdır.

**Son bölümler:** §35a (geçersiz koşu kaydı), §29b (ön-kayıt + sonuç), §36
(ön-kayıt, koşuyor). `git log`'a bak.

---

## Dosya sahipliği (çakışma önleme)

| dosya | sahip |
|---|---|
| `RESULTS.md` | **GPU oturumu** — DÜZENLEME, hata bulursan raporla |
| `notebooks/` | **GPU oturumu** — dokunma (okuyabilirsin) |
| `review_scripts/` | **sen** — GPU oturumu bu süre boyunca dokunmayacak |
| `docs/drafts/`, `docs/internal_tr/` | **sen** (yeni dosyalar) |
| `README.md`, `PROJECT_SUMMARY.md`, `.tex` | **sen** — ama onaysız düzeltme yok |

## Yapmayacakların

- **GPU/compute işi başlatma.** Kotanın tamamı §36'ya ayrılmış. CPU testi serbest.
- **`git push` yapma.** Commit atabilirsin; push kullanıcıya aittir.
- **Checkpoint (`.pt`) commit etme.** `.gitignore` bunları dışarıda tutuyor ve
  ağırlık yayınlamak `LISANS_KARAR_REHBERI.md` Kapı 2'yi tetikler; ağırlık
  lisansı kararı henüz verilmedi, yayınlanan ağırlık geri alınamaz.
- **Testi zayıflatma.** `smoke_test.py` ve `verify_claims.py` bekçidir; eşik
  gevşetme, `skip`, assert silme yasak. Test kırmızıysa kod düzeltilir.
- **Yeni ampirik iddia üretme.** Ölçülmemiş sayı yazma; tahminse "tahmin" de.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

## Değişiklikten sonra

`python smoke_test.py` + `python review_scripts/verify_claims.py`.
İkisi yeşil değilse iş bitmemiştir.

## Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor).
Ne yapıldı + neden + hangi §.

## Belirsizlikte

Dur ve kullanıcıya sor. Özellikle: bir iddianın dayanağını bulamıyorsan,
aracın kapsamı hakkında bir söz vermen gerekiyorsa, ya da lisans/atıf ile ilgili
bir şey varsa. Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
