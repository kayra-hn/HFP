# Ajan brifingi — deney aracı sertleştirme + yayın/iddia denetimi

*İkinci ajana verilecek görev tanımı. GPU oturumu paralel olarak §36'yı koşuyor.
Hazırlanma: 2026-08-01, repo HEAD `9de9776`.*

---

## Prompt (kopyala-yapıştır)

Sen HFP projesinde çalışıyorsun: `C:\Users\yilma\Documents\HFP_Project`.
Önceden eğitilmiş bir LLM'in (Qwen2.5-1.5B) attention katmanlarının bir kısmını
O(1)-bellekli recurrent bir mekanizmayla değiştiren bir araştırma projesi.
Sahibi Kayrahan Yılmaz, tek başına, bedava GPU kotalarıyla çalışıyor.

**Paralel bir oturum aynı repoda GPU deneyi (§36) koşuyor.** Dosya sahipliği
aşağıda; ona harfiyen uy, yoksa otorite kayıt bozulur.

### Önce oku (bu sırayla)

1. `AGENTS.md` — çalışma kuralları. **İki kural ödünsüz:** bilimsel dürüstlük,
   lisans/IP koruması (AGPL-3.0).
2. `RESULTS.md` — **otorite deney kaydı.** Başındaki ERRATUM'u mutlaka oku.
   §1'den §36'ya; kayıtların ~2/3'ü negatif.
3. `review_scripts/carry_curriculum.py` ve `matched_probe.py` — asıl deney aracı.
4. `README.md`, `docs/PROJECT_SUMMARY.md`, `docs/paper3_ml_architecture.tex`.

### Kritik bağlam — bunları bilmeden başlama

**1. Strateji belgeleri kullanıcının değil, önceki asistanların yazımı.**
`docs/internal_tr/YAYIN_PLANI.md`, `YOL_HARITASI_*.md`, `DEVIR.md` §6 üçüncü
şahısta yazılmış asistan çıkarımlarıdır; kullanıcı hedeflerinin farklı olduğunu
açıkça söyledi. **Kullanıcı niyeti olarak alma.** Olgusal envanterleri
kullanılabilir; stratejik önermeleri sorulmadan varsayılamaz.
`RESULTS.md` istisnadır — ölçüm kaydıdır, otoritedir.

**2. §28a'nın atfı YENİ düzeltildi (§29b, 2026-08-01), dış belgelerde yok.**
- §28a kendi metriğinde replike oldu: +0.4080 nat, 14/16 seed, p=0.0005.
- Ama o metrik karışıktı: B-chunk etiketleri hem chunk-içi çiftleri hem
  cross-chunk hedefini içeriyor; P=6'da denetlenen 7 tokenın 6'sı chunk-içi.
- Ayrıştırınca chunk-içi **+0.4298** (p=0.0008), cross-chunk **+0.2770**
  (eşleşmiş t p=0.1103, SD 0.653). Sınır ötesi bileşen gerçek ama küçük ve
  gürültülü. §28a'yı "cross-chunk carry kazancı" diye sunmak abartılıdır.

**3. Erratum merkezi.** Hibritlerde görünen uzun menzilli geri getirme tamamen
artık KV-cache tarafından taşınabilir (§30). Kendi sonuçlarımızda yakalandı.

**4. Son bölümler:** §35a (geçersiz koşu kaydı), §29b (ön-kayıt + sonuç),
§36 (ön-kayıt, koşuyor). `git log` bak.

---

## GÖREV A — Deney aracını sertleştir (BİRİNCİL)

Bu oturumda araçta gerçek kusurlar bulundu ve **hiçbiri düzeltilmedi.** Deneyler
bunlar yüzünden yorumlanamıyor. Sırayla:

**A1 — Üst sınır kolu yok. En kritik eksik.**
`matched_probe.py`'nin `probe_unmatched` fonksiyonu K=0'da tam olarak
`[tgt_k, v, tgt_k]` dizisini üretiyor — üç token, `local_window=8` içinde. Yani
"ESKI sonda K=0'da %100" sadece yerel attention'ın çalıştığını gösteriyor,
taşıma hakkında sıfır bilgi veriyor. **Araçta geçerli bir üst sınır kolu yok.**
Sonuç: her "%0" ölçümü, "mekanizma başarısız" ile "görev zaten çözülemez"
arasında ayrım yapamıyor. `AGENTS.md` bu kolu zorunlu kılıyor.
Yap: eşleşmiş sahnenin çözülebilir olduğunu kanıtlayan bir kol tasarla ve ekle
(ör. tam attention ile aynı sahne, ya da sınır olmadan aynı mesafe). Ön-kayıt
gerektirmez — bu bir araç kontrolü, hipotez testi değil.

**A2 — Eğitim etiketleri karışık.** `carry_curriculum.py:126` (`dense_chunk`)
ve `:150` (`carry_example`): B chunk'ı hem chunk-içi çiftleri hem cross-chunk
hedefini denetliyor. Ölçüm tarafı düzeltildi (`val_cross`/`val_inchunk` ayrı
yazılıyor), **eğitim tarafı düzeltilmedi.** Cross-chunk hedefi kaybın 1/7'si;
model onu görmezden gelip chunk-içini optimize edebilir.
Yap: seçenekleri **analiz et ve öner** (ayrı ağırlık, ayrı loss terimi, maskeli
ikinci geçiş). **Tek başına değiştirme** — bu değişiklik §26-§29 arası bütün
tarihsel karşılaştırılabilirliği kırar. Öneriyi kullanıcıya sun.

**A3 — Araç bimodal sonuç üretiyor, ortalamalar bunu gizliyor.**
§29b'de `val_cross` dağılımı: `exp` 16/16 seed'de 3.25-3.65 arasında (görevi hiç
çözmüyor), `parallel_cubic` beş seed'de 3.0 altına iniyor (1.35, 2.43, 2.50,
2.56, 2.90). Ortalama + t-testi bu yapıyı görmüyor; §29b'de t-testi tek bir
aykırı yüzünden p=0.11 verirken işaret testi p=0.0042 verdi.
Yap: `review_scripts/report_stats.py` — importable bir raporlama modülü.
Her karşılaştırma için: seed-başı tablo, min/max, eşleşmiş t + işaret testi +
Wilcoxon (üçü birden), ve "çözdü/çözmedi" sayımı. Notebook'lar istatistiği elle
yazmayı bıraksın. Saf Python + numpy, scipy bağımlılığı ekleme.

**A4 — Ön-kayıt kusurları tekrarlanıyor.** Bu oturumda üç tanesi çıktı:
(i) §35 v2'de mutlak eşik, §26b'nin 6 seed'de bile ayıramadığı bir ölçümde
2 seed ile; (ii) §35 v3'te araç geçerlilik kapısı yok — araç eşiğin altında
kaldığı fark edilmeden "sonuç" üretildi; (iii) §29b'de "p<0.05" yazıp **hangi
test** olduğu yazılmadı, ve estimand kararsız paydalı bir orandı.
Yap: `docs/internal_tr/ON_KAYIT_KONTROL_LISTESI.md` — her yeni § için doldurulacak
kısa liste. En az şunları içermeli: birincil endpoint ve neden düşük varyanslı
olduğu; **hangi istatistiksel test** (önceden, isimle); araç geçerlilik kapısı ve
eşiği; güç kontrolü (iki kol da şanstaysa "sonuçsuz"); üst sınır kolu; durma
kuralı; bütçe tavanı. Her maddenin yanına bu projede hangi § ihlal ettiği
yazılsın — soyut kural değil, kendi kayıtlarından çıkarılmış ders olsun.

---

## GÖREV B — Yayın / iddia denetimi (İKİNCİL)

**B1 — İddia denetimi.** `README.md`, `docs/PROJECT_SUMMARY.md`,
`docs/paper3_ml_architecture.tex` içindeki **her ampirik iddiayı** satır satır
`RESULTS.md` ile karşılaştır. Bilinen sorunlu yerler (başlangıç, tükettiği değil):
- `README.md` ~69-77 ve ~120 — "Sparse writes / long-lived memory (cross-chunk
  carry)" çerçevesi ve "+0.48 nat, 13/16, p=0.012"
- `docs/PROJECT_SUMMARY.md` ~45-50 — "sparse-write **carry** task"
- `docs/paper3_ml_architecture.tex` — hiç elden geçmedi, erratum öncesi yazılmış
  olabilir; özellikle dikkat et.

Çıktı: `docs/internal_tr/IDDIA_DENETIMI_2026-08.md`, tablo:

| belge:satır | iddia | RESULTS referansı | durum | önerilen metin |

`durum` ∈ {DOĞRU, BAYAT, ABARTILI, DESTEKSİZ, ÇELİŞKİLİ}. Dayanak bulamıyorsan
DESTEKSİZ yaz; "muhtemelen §X" yazma. **Düzeltmeleri uygulama** — kullanıcı
onaylasın (açıkça "uygula" derse README ve PROJECT_SUMMARY'yi düzeltebilirsin).

**B2 — Confound yazısı taslağı.** `docs/drafts/kv_cache_confound.md`.
İddia: hibrit attention/O(1) modellerde ölçülen uzun menzilli geri getirme artık
KV-cache tarafından taşınıyor olabilir ve alanda genelde cache açıkken ölçülüyor.
Teşhis protokolü: chunk başına cache sıfırla + üst sınır kolu + checkpoint
yükleme doğrulaması. **Kendi hatamızı örnek kullan** (§22 needle → §30 cache-reset
→ erratum) — yazının en güçlü kısmı bu. Sayılar `RESULTS.md`'den, uydurma yok.
İlgili hattı (Mamba-in-Llama, LoLCATs, MOHAWK, LAWCAT) **atıf** olarak an;
"bunlar hatalı" iddiası kurma, o iddianın kanıtı bizde yok.

---

## Dosya sahipliği (çakışma önleme)

| dosya | sahip |
|---|---|
| `RESULTS.md` | **GPU oturumu** — sen DÜZENLEME, hata bulursan raporla |
| `notebooks/` | **GPU oturumu** — dokunma |
| `review_scripts/` | **sen** — GPU oturumu bu süre boyunca dokunmayacak |
| `docs/internal_tr/`, `docs/drafts/` | **sen** (yeni dosyalar) |
| `README.md`, `PROJECT_SUMMARY.md`, `.tex` | **sen** — ama onaysız düzeltme yok |

## Yapmayacakların

- **GPU/compute işi başlatma.** Kotanın tamamı §36'ya ayrılmış.
- **`git push` yapma.** Commit atabilirsin; push kullanıcıya aittir.
- **Checkpoint (`.pt`) commit etme.** `.gitignore` bunları dışarıda tutuyor ve
  ağırlık yayınlamak `LISANS_KARAR_REHBERI.md` Kapı 2'yi tetikler; ağırlık
  lisansı kararı henüz verilmedi ve yayınlanan ağırlık geri alınamaz.
- **Testi zayıflatma.** `smoke_test.py` ve `verify_claims.py` bekçidir; eşik
  gevşetme, `skip`, assert silme yasak. Test kırmızıysa kod düzeltilir.
- **Yeni ampirik iddia üretme.** Ölçülmemiş sayı yazma; tahminse "tahmin" de.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

## Değişiklikten sonra

Bellek yolunu veya matematiği değiştiren her şeyden sonra:
`python smoke_test.py` + `python review_scripts/verify_claims.py`.
İkisi yeşil değilse iş bitmemiştir.

## Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor).
Ne yapıldı + neden + hangi §.

## Belirsizlikte

Dur ve kullanıcıya sor. Özellikle: bir iddianın dayanağını bulamıyorsan, bir
değişiklik tarihsel karşılaştırılabilirliği kıracaksa, ya da lisans/atıf ile
ilgili bir şey varsa. Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
