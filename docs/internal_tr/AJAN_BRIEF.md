# Ajan brifingi — bağımsız confound teşhis aracı + HFP yayın hattı

> ## ⏸ PARK EDİLDİ — 2026-08-02
>
> Kullanıcı bu hattı **~2 hafta ertelemeye** karar verdi; kendisi ilgilenecek.
> İptal değil. Başlatmadan önce aşağıdakileri bil:
>
> - **Bu ajan hiç başlatılmadı.** Brief'teki "2. ajan" dilim rezervasyonları
>   diğer brief'lerden **kaldırıldı**; yeniden başlatılırsa sahiplik tablosu
>   o günkü duruma göre yeniden yazılmalı.
> - **Görev B (confound yazısı) ve Görev C (iddia denetimi) bu brief'ten
>   ÇIKARILDI.** İkisi de Sınıf 2 iş — her satır `RESULTS.md`'ye karşı yargı
>   gerektiriyor — ve GPU oturumuna devredildi. Bu brief yeniden açılırsa
>   **yalnız Görev A (araç)** geçerlidir.
> - `C:\Users\yilma\Documents\cache-audit` **oluşturuldu**: Apache-2.0
>   `LICENSE` + `NOTICE` + iskelet `README.md`, tek commit (`7dd4665`).
>   GitHub remote'u **yok**. Durum notu: `cache-audit/DURUM.md`.
> - Lisans kararı ve bağlayıcı kısıtlar: `LISANS_KARAR_REHBERI.md`
>   karar defteri, 2026-08-01 kaydı. **Değişmedi, geçerli.**

*İkinci ajana verilecek görev tanımı. GPU oturumu paralel olarak §36'yı koşuyor.
Hazırlanma: 2026-08-01, HFP repo HEAD `2742dcd`.*

**İki klasörde çalışılacak.** Araç ayrı bir repoda, Apache-2.0 ile. Lisans kararı
ve gerekçesi `docs/internal_tr/LISANS_KARAR_REHBERI.md` karar defterinde
(2026-08-01 kaydı).

---

## Prompt (kopyala-yapıştır)

İki klasörde çalışıyorsun:

- `C:\Users\yilma\Documents\HFP_Project` — HFP araştırma projesi, **AGPL-3.0**
- `<YENİ_KLASÖR>` — yazacağın bağımsız teşhis aracı, **Apache-2.0**

HFP, önceden eğitilmiş bir LLM'in (Qwen2.5-1.5B) attention katmanlarının bir
kısmını O(1)-bellekli recurrent bir mekanizmayla değiştiren bir araştırma
projesi. Sahibi Kayrahan Yılmaz, tek başına, bedava GPU kotalarıyla çalışıyor.
Telif %100 kendisinde; projeye hiç dış katkı alınmadı.

**Paralel bir oturum HFP reposunda GPU deneyi (§36) koşuyor.** Dosya sahipliği
aşağıda; ona harfiyen uy.

### Önce oku (bu sırayla)

1. `HFP_Project/AGENTS.md` — çalışma kuralları. **İki kural ödünsüz:**
   bilimsel dürüstlük, lisans/IP koruması.
2. `HFP_Project/RESULTS.md` — **otorite deney kaydı.** Başındaki ERRATUM'u
   mutlaka oku, sonra **§30, §30a-c, §31, §33, §34** ve **§36** (ön-kayıt).
   Asıl işin bunlardan çıkıyor.
3. `HFP_Project/docs/internal_tr/LISANS_KARAR_REHBERI.md` — özellikle
   2026-08-01 karar defteri kaydı. Bağlayıcı kısıtlar orada.
4. `HFP_Project/notebooks/memory_capability_v30.ipynb` — mevcut cache-reset
   makinesi, dağınık hâlde. Özellikle hücre 2 ve **hücre 7** (yükleme
   doğrulaması).

---

## GÖREV A — Bağımsız teşhis aracını yaz (BİRİNCİL)

### Bulgu (aracın test edeceği şey)

Hibrit attention/O(1) modellerde ölçülen uzun menzilli geri getirme, **artık
attention katmanlarının KV-cache'i tarafından taşınıyor olabilir** — ve alanda
bu genelde cache açıkken ölçülüyor. HFP'de yakalandı: needle sonuçları
512-16384 token mesafede "BULDU" diyordu; cache chunk başına sıfırlanınca
**her mesafede %0**, tek bir chunk sınırında bile (§30a-c). Üç ayrı eğitim
müdahalesi denendi, hiçbiri değiştirmedi (§31, §33, §34). Erratum'la kayda geçti.

### Araç

Yeni klasörde, **bağımsız bir Python paketi.** HFP'ye bağımlı olmayacak.
Üç koşulu koşar ve **hükmü kendisi basar** — kullanıcı üç sayıya bakıp
yorumlamak zorunda kalmasın:

| koşul | ne yapar | ne söyler |
|---|---|---|
| **A — cache açık** | standart değerlendirme | alanın genelde raporladığı sayı |
| **B — cache chunk başına sıfır** | yalnız recurrent state taşır | state'in gerçekte taşıdığı |
| **C — üst sınır** | tam attention, aynı sonda | sonda geçerli mi, model çözebiliyor mu |

Hüküm mantığı:

- **C düşükse** → sonda geçersiz ya da model görevi zaten çözemiyor.
  **Diğer hiçbir satır yorumlanamaz.** Araç bunu en başta söylemeli.
- **A yüksek, B ≈ şans** → geri getirme **cache'den geliyor**; O(1) bellek
  iddiası bu ölçümle desteklenmiyor.
- **A yüksek, B yüksek** → state gerçekten taşıyor.
- **A ≈ B ≈ şans** → hiç geri getirme yok; ayrı bir sorun.

### Zorunlu bileşenler

**1. İKİ doğrulama vakası. Araç kendini kanıtlamadan yayınlanmaz.**

- **Negatif kontrol — düz transformer.** Recurrent state'i olmayan sıradan bir
  HF modelinde, cache sıfırlanınca geri getirme *tanım gereği* çöker: taşıyacak
  başka kanal yok. Araç bunu **%100 yakalamalı**. Yakalayamıyorsa araç bozuktur.
  Küçük bir model + CPU yeter, compute istemez. Bu, aracın mantığını HFP'den
  **bağımsız** olarak doğrular ve bağımsız yayının ön şartıdır.
- **Gerçek vaka — HFP.** `--selftest` benzeri bir yol: HFP checkpoint'iyle
  koşulunca §30a-c yeniden üretilmeli (A yüksek / B %0 / C %100). HFP
  checkpoint'leri `.gitignore`'da ve repoda **yok**; dosya yoksa selftest
  **atlanıp öyle raporlanmalı**, sessizce geçilmemeli.

**2. Checkpoint yükleme doğrulaması.** `memory_capability_v30.ipynb` hücre 7'nin
mantığını araca taşı. Gerekçe: `strict=False` isim uyuşmazlığında **sessizce
hiçbir şey yüklemez**, ve o durumda B=%0 "hafıza yok"u değil "ağırlıklar
eğitimsiz"i gösterir. Araç bu ikisini ayırt etmeden hüküm basmamalı. Kontroller:
tensor isim/şekil eşleşme sayısı, bit-bit değer karşılaştırması, eğitim izi.

**3. Üst sınır kolu — HFP'deki mevcut kod BOZUK, kopyalama.**
`HFP_Project/review_scripts/matched_probe.py` içindeki `probe_unmatched`, K=0'da
tam olarak `[tgt_k, v, tgt_k]` dizisini üretiyor — üç token, `local_window=8`
içinde. Yani bugüne kadar "üst sınır %100" diye okunan şey yalnızca yerel
attention'ın çalıştığını gösteriyor. **Geçerli bir üst sınır kolunu sıfırdan
yazman gerek:** aynı sonda, aynı mesafe, ama bilgi tam attention üzerinden
erişilebilir. Aracın en kritik parçası — onsuz her "%0" yorumlanamaz.

### Arayüz ve kapsam — dürüstlük şartı

Aracı "her hibrit modelde çalışır" diye pazarlama. Dışarıdan takılabilir olması
gereken en az iki şey: (a) cache sıfırlama fonksiyonu, (b) sonda üretici.
README'de **neyin test edildiğini ve neyin edilmediğini** açıkça yaz; hangi
model ailelerinde denendiğini ve hangilerinde denenmediğini listele.

Bağımlılık: `torch` + `transformers`. Başkasını ekleme (scipy dahil);
gerekiyorsa gerekçelendir.

### Lisans ve IP — bağlayıcı, tartışma yok

- Araç **Apache-2.0**. Her kaynak dosyaya Apache başlığı, kökte `LICENSE`
  (Apache-2.0 tam metni) ve `NOTICE`. Telif: Kayrahan Yılmaz.
- **HFP AGPL kalır.** Araç HFP'ye bağımlı olmaz; HFP'den kod taşıyacaksan
  yalnızca Kayrahan'ın kendi yazdığı kodu taşı ve hangi dosyadan geldiğini
  commit mesajında belirt.
- **Dış kod kontrolü zorunlu:** araca giren hiçbir satır başka bir projeden
  kopyalanmış olmayacak. Şüphen varsa **dur ve sor.**
- **HFP'nin yayınlanmamış mimari detayları araca GİRMEZ** — özellikle
  `cubic_flux` içselleri. Gerekçe: `LISANS_KARAR_REHBERI.md` Kapı 3, patent
  zamanlamasını cubic mekanizmasına özel işaretliyor; yayınlanan her şey prior
  art olur. Confound protokolü bilinçli olarak yayınlanıyor, o kapsam dışı.
- HFP reposunun README'sine araca **işaret eden bir satır** eklenebilir (ters
  yönde bağımlılık yok).

---

## GÖREV B — Confound yazısı (İKİNCİL, A ile birlikte anlamlı)

`HFP_Project/docs/drafts/kv_cache_confound.md`. Yazı ve araç birbirini
tamamlıyor: yazı "sizde de olabilir" der, araç "buyur, kontrol et" der.

- Somut sayılar `RESULTS.md`'den alınır, **uydurulmaz**.
- **Kendi hatamızı merkeze koy** (§22 needle → §30 cache-reset → erratum).
  Yazının en güçlü kısmı bu: başkasını suçlayan değil kendini düzelten bir metin.
- İlgili hattı (Mamba-in-Llama, LoLCATs, MOHAWK, LAWCAT) **atıf** olarak an.
  "Bunlar hatalı" iddiası **kurma** — o iddianın kanıtı yok; kurarsan yazının
  güvenilirliği gider.
- Sonunda araca yönlendir.

---

## GÖREV C — İddia denetimi (ÜÇÜNCÜL, A ve B bitince)

`README.md`, `docs/PROJECT_SUMMARY.md`, `docs/paper3_ml_architecture.tex`
içindeki her ampirik iddiayı satır satır `RESULTS.md` ile karşılaştır.

**Bilinen canlı yanlışlık:** §28a'nın atfı §29b'de (2026-08-01) düzeltildi ama
dış belgeler bunu içermiyor. §28a kendi metriğinde replike oldu (+0.4080 nat,
14/16, p=0.0005) **ama o metrik karışıktı** — B-chunk etiketleri hem chunk-içi
çiftleri hem cross-chunk hedefini içeriyor, P=6'da denetlenen 7 tokenın 6'sı
chunk-içi. Ayrıştırınca chunk-içi +0.4298 (p=0.0008), cross-chunk +0.2770
(t p=0.1103). Yani §28a'yı **"cross-chunk carry kazancı"** diye sunmak abartılı.
Etkilenen: `README.md` ~69-77 ve ~120, `PROJECT_SUMMARY.md` ~45-50, ve `.tex`
(hiç elden geçmedi, erratum öncesi olabilir).

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
| `HFP_Project/RESULTS.md` | **GPU oturumu** — DÜZENLEME, hata bulursan raporla |
| `HFP_Project/notebooks/` | **GPU oturumu** — dokunma (okuyabilirsin) |
| `HFP_Project/review_scripts/` | **sen** — GPU oturumu dokunmayacak |
| `HFP_Project/docs/drafts/`, `docs/internal_tr/` | **sen** (yeni dosyalar) |
| `HFP_Project/README.md`, `PROJECT_SUMMARY.md`, `.tex` | **sen** — onaysız düzeltme yok |
| yeni araç klasörü | **tamamen sen** |

## Yapmayacakların

- **GPU/compute işi başlatma.** Kotanın tamamı §36'ya ayrılmış. CPU serbest.
- **`git push` yapma** (iki repoda da). Commit atabilirsin; push kullanıcıya ait.
- **Checkpoint (`.pt`) commit etme.** Ağırlık yayınlamak Kapı 2'yi tetikler;
  ağırlık lisansı kararı henüz verilmedi ve yayınlanan ağırlık geri alınamaz.
- **Testi zayıflatma.** `smoke_test.py` ve `verify_claims.py` bekçidir; eşik
  gevşetme, `skip`, assert silme yasak. Test kırmızıysa kod düzeltilir.
- **Yeni ampirik iddia üretme.** Ölçülmemiş sayı yazma; tahminse "tahmin" de.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

## HFP'de değişiklikten sonra

`python smoke_test.py` + `python review_scripts/verify_claims.py`.
İkisi yeşil değilse iş bitmemiştir. (Araç reposu kendi testlerini taşır.)

## Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor).
Ne yapıldı + neden + hangi §.

## Belirsizlikte

Dur ve kullanıcıya sor. Özellikle: bir kodun kökeni/lisansı şüpheliyse, aracın
kapsamı hakkında bir söz vermen gerekiyorsa, bir iddianın dayanağını
bulamıyorsan. Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
