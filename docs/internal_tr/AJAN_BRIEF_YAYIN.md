# Ajan brifingi — yayın hattı ve iddia denetimi

*Bu dosya, HFP projesinde **yayın hattında** çalışacak ikinci bir ajana verilecek
görev tanımıdır. Bilgisayar/GPU işi yapan asıl oturum paralel olarak §36'yı
koşuyor. Hazırlanma: 2026-08-01, repo HEAD `dca60f8`.*

---

## Prompt (kopyala-yapıştır)

Sen HFP projesinde çalışıyorsun: `C:\Users\yilma\Documents\HFP_Project`.
Bu, önceden eğitilmiş bir LLM'in (Qwen2.5-1.5B) attention katmanlarının bir
kısmını O(1)-bellekli recurrent bir mekanizmayla değiştiren bir araştırma
projesi. Sahibi Kayrahan Yılmaz, tek başına ve bedava GPU kotalarıyla çalışıyor.

### Önce oku (bu sırayla)

1. `AGENTS.md` — çalışma kuralları. **İki kural ödünsüz:** bilimsel dürüstlük ve
   lisans/IP koruması (AGPL-3.0).
2. `RESULTS.md` — **otorite deney kaydı.** Başındaki ERRATUM'u mutlaka oku.
   §1'den §36'ya; kayıtların ~2/3'ü negatif.
3. `README.md`, `docs/PROJECT_SUMMARY.md`, `docs/paper3_ml_architecture.tex` —
   dışarıya bakan belgeler; senin denetleyeceğin şeyler bunlar.

### Kritik bağlam — bunları bilmeden başlama

**1. Strateji belgeleri kullanıcının değil, önceki asistanların yazımı.**
`docs/internal_tr/YAYIN_PLANI.md`, `YOL_HARITASI_*.md` ve `DEVIR.md` §6
("kullanıcı bağlamı") üçüncü şahısta yazılmış asistan çıkarımlarıdır.
Kullanıcı bunu açıkça belirtti: hedefleri o belgelerde yazandan farklı.
**Bunları kullanıcı niyeti olarak alma.** İçlerindeki *olgusal* envanter
(hangi sonuç nerede) kullanılabilir; *stratejik* önermeler (öncelik sırası,
"para şöyle gelir", kanal sıralaması) kullanıcıya sorulmadan varsayılamaz.
`RESULTS.md` bu ayrımın dışındadır — o ölçüm kaydıdır ve otoritedir.

**2. §28a'nın atfı YENİ düzeltildi (§29b, 2026-08-01).** Dışarıya bakan
belgeler bu düzeltmeyi henüz içermiyor. Özet:
- §28a'nın etkisi **kendi metriğinde replike oldu**: +0.4080 nat, 14/16 seed,
  p=0.0005 (orijinal +0.484, 13/16, p=0.0124). §28a çürütülmedi.
- **Ama o metrik karışıktı.** `carry_curriculum.py`'nin B-chunk etiketleri hem
  chunk-içi çiftleri hem cross-chunk hedefini içeriyor; P=6'da denetlenen 7
  tokenın 6'sı chunk-içi.
- Ayrıştırınca: chunk-içi **+0.4298** (p=0.0008), cross-chunk **+0.2770**
  (eşleşmiş t p=0.1103, SD 0.653). Sınır ötesi bileşen **gerçek ama daha küçük
  ve belirgin şekilde daha gürültülü**.
- Sonuç: §28a'yı bir **"cross-chunk carry"** kazancı gibi sunmak sınır-ötesi
  kısmı olduğundan büyük gösteriyor.

**3. Erratum hâlâ geçerli ve merkezi.** Hibrit attention/O(1) modellerde
görünen uzun menzilli geri getirme tamamen artık KV-cache tarafından
taşınabilir. Kendi sonuçlarımızda yakalandı ve erratum'la düzeltildi (§30).

**4. Son commitler** (`git log`): §35a (geçersiz koşu kaydı), §29b (ön-kayıt +
sonuç), §36 (ön-kayıt). Bunlar `RESULTS.md`'de; dış belgelerde yok.

### Görevin

**A — İDDİA DENETİMİ (birincil, önce bunu bitir).**
`README.md`, `docs/PROJECT_SUMMARY.md` ve `docs/paper3_ml_architecture.tex`
içindeki **her ampirik iddiayı** satır satır `RESULTS.md` ile karşılaştır.
Bilinen sorunlu yerler (başlangıç noktası, tükettiği yer değil):
- `README.md` satır ~69-77 ve ~120 — "Sparse writes / long-lived memory
  (cross-chunk carry)" çerçevesi ve "+0.48 nat, 13/16, p=0.012"
- `docs/PROJECT_SUMMARY.md` satır ~45-50 — "pre-registered, paired, 16-seed win
  in a sparse-write **carry** task"

Çıktı: `docs/internal_tr/IDDIA_DENETIMI_2026-08.md` — bir tablo:

| belge:satır | iddia | RESULTS referansı | durum | önerilen metin |

`durum` ∈ {DOĞRU, BAYAT, ABARTILI, DESTEKSİZ, ÇELİŞKİLİ}. Her satırda RESULTS'ta
hangi bölümün dayanak olduğunu göster. Dayanak bulamıyorsan DESTEKSİZ yaz —
tahmin etme, "muhtemelen §X" yazma.

**Düzeltmeleri sen uygulama.** Tabloyu üret, kullanıcı onaylasın. İstisna:
kullanıcı açıkça "uygula" derse README ve PROJECT_SUMMARY'yi düzeltebilirsin.

**B — CONFOUND YAZISI (ikincil).**
Erratum bulgusu bu projenin en yüksek dolaşım potansiyelli parçası ve compute
istemiyor. Tek başına duran, kısa bir teknik yazı taslağı hazırla:
`docs/drafts/kv_cache_confound.md`.
- Ne iddia ediliyor: hibrit attention/O(1) modellerde ölçülen uzun menzilli
  geri getirme, artık KV-cache tarafından taşınıyor olabilir ve alanda genelde
  cache açıkken ölçülüyor.
- Teşhis protokolü: chunk başına cache sıfırla + tam-attention üst sınır kolu +
  checkpoint yükleme doğrulaması. Bu üçü olmadan sonuç yorumlanamaz.
- **Kendi hatamızı örnek olarak kullan** — §22'nin needle sonuçları, §30'un
  cache-reset ölçümü, erratum. Bu, iddianın en güçlü kısmı; başkasını suçlayan
  değil kendini düzelten bir yazı.
- Somut sayılar `RESULTS.md`'den alınacak, uydurulmayacak.
- Kimseyi isim vererek suçlama. İlgili hattı (Mamba-in-Llama, LoLCATs, MOHAWK,
  LAWCAT) *atıf* olarak an, "bunlar hatalı" iddiası kurma — o iddianın kanıtı
  bizde yok.

### Yapmayacakların

- **`RESULTS.md`'yi DÜZENLEME.** Paralel oturum onu düzenliyor; çakışma otorite
  kaydı bozar. Bir hata bulursan raporla, düzeltme.
- **GPU/compute işi başlatma.** Kotanın tamamı §36'ya ayrıldı.
- **`git push` yapma.** Commit atabilirsin; push kullanıcıya aittir (AGENTS.md).
- **Yeni iddia üretme.** Görevin mevcut iddiaları kayda karşı doğrulamak, yeni
  sonuç çıkarmak değil.
- Ölçülmemiş sayı yazma. Tahminse "tahmin" de.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

### Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor). Ne yapıldı +
neden + hangi §.

### Belirsizlikte

Dur ve kullanıcıya sor. Özellikle: bir iddianın dayanağını bulamıyorsan,
bir belgenin kime ait olduğundan emin değilsen, ya da lisans/atıf ile ilgili
bir şey varsa. Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
