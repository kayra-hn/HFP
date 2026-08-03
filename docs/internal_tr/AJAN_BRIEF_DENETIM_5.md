# Ajan brifingi — denetim ajanı, 5. tur

*Hazırlanma: 2026-08-02. Ajan bu repoyu proje kökü olarak açıyor; yollar repo
köküne göreli. §37 GPU oturumunda koşuyor — `notebooks/` şu an GPU oturumunun.*

---

## Prompt (kopyala-yapıştır)

Bu repoda teknik denetim ve bakım yapıyorsun. Proje kökündeki `AGENTS.md`'yi oku —
**iki kural ödünsüz:** bilimsel dürüstlük ve lisans/IP koruması.

**Bu bir araştırma reposu.** Negatif sonuçlar, erratum'lar ve dürüstlük uyarıları
kasıtlıdır; "temizlenmez".

### Bu turun gerekçesi: bir hata sınıfı üç kez tekrarladı

Bugün notebook'larda üç hata çıktı ve **üçü de aynı sınıftan** — hepsi
koşulmadan, GPU olmadan, hatta CPU bile olmadan yakalanabilirdi:

1. **§29b:** hücre bir script'in **stdout'undan regex'le** sayı kazıyordu
   (`cross-chunk dogrulama loss:`). O print satırı başka bir işte değişti →
   regex eşleşmedi → 48 kolun hepsi sessizce "IRAKSADI" olacaktı. Koşudan önce
   yakalandı, ama şansla.
2. **§36/§37:** hücre 2 `find_one`'ı tanımlıyordu; hücre 2 değiştirilince tanım
   düştü, ama hücre 3 hâlâ `find_one(...)` çağırıyordu → `NameError`.
3. **§36/§37:** `find_one('config.json')` **en yeni** `config.json`'u
   döndürüyordu ve repoda `hf_upload/hf_release/config.json` var — o klasörde
   ağırlık yok, dolayısıyla model bulunamıyordu.

Ortak kök: **notebook'ların hücreler-arası bağımlılıkları hiç kontrol edilmiyor.**
AST parse ediliyor, hüküm mantığı sentetik veriyle sınanıyor, ama "hücre N,
hücre N−1'in tanımladığı adı mı kullanıyor" sorusu hiç sorulmuyor.

---

## GÖREV 1 — Notebook statik kontrolü (BU TURUN ASIL İŞİ)

`review_scripts/check_notebooks.py` yaz. Notebook'ları **koşmadan**, salt statik
olarak denetler. Saf Python + `ast` + `json`; torch gerekmez, CPU'da saniyeler
sürer.

**K1 — Hücreler-arası ad çözümleme.** Kod hücrelerini sırayla gez. Her hücre
için, o hücrede kullanılan serbest adları (`ast.Name` yükleme bağlamında) çıkar
ve şunlarla karşılaştır: o ana kadarki hücrelerde tanımlanmış adlar + importlar +
Python builtin'leri. Tanımsız bir ad kullanılıyorsa **hata**.

Dikkat edilecekler: fonksiyon gövdesindeki yerel adlar, comprehension
değişkenleri, `global`/`nonlocal`, `for`/`with`/`except as` bağlamaları,
`import x as y`, ve yıldızlı importlar (bunları bilinmeyen sayıp uyar).
Tam doğruluk gerekmez; **yanlış negatiften çok yanlış pozitif** tercih edilir —
ama yanlış pozitif oranı raporda görünsün.

**K2 — stdout kazıma tespiti.** Bir notebook hem `subprocess.run` hem
`re.compile`/`re.search` kullanıyor ve regex `capture_output` sonucuna
uygulanıyorsa **uyar**. Gerekçe §29b'de: script'in print formatı değişince
ölçüm sessizce bozulur. Doğru desen, script'in **yazdığı dosyayı okumak**.

**K3 — Kimlik satırı eksikliği.** Her notebook'un ilk kod hücresi kendi kimliğini
basmalı: hangi § olduğu ve beklenen çıktı klasörü. Basmıyorsa **uyar**.
Gerekçe: bugün dört kez yanlış notebook koşuldu (`chain35v4` yazan §35'i §37
sanarak). **Sen notebook'ları DEĞİŞTİRME** — eksikleri raporla, satırı GPU
oturumu ekleyecek (`notebooks/` şu an onun).

**K4 — Sabit yol / bayat referans.** Notebook'larda geçen dosya yolları repoda
var mı? (`.gitignore`'daki yollar hariç — onlar kasten yok, ama bunu raporda
belirt.)

Çıktı: satır satır bulgu + özet. `--strict` bayrağıyla hata varsa çıkış kodu 1.
**CI'a EKLEME** — önce yanlış pozitif oranını görmek gerekiyor; öneriyi rapora yaz.

**Doğrulama:** kendi kontrolünü kanıtla. `notebooks/` altındaki mevcut
notebook'larda koştur ve yukarıdaki üç bilinen hatadan **hangilerini yakaladığını**
raporla. K1'in `find_one` hatasını yakalaması beklenir (o hata artık düzeltilmiş
olabilir — o zaman git geçmişinden `9de9776`..`48ea8bd` arası bir sürümde test et
ve öyle raporla). Yakalamıyorsa kontrol zayıftır ve öyle yazılır.

---

## ~~GÖREV 2 — Satır sonu normalizasyonunu TAMAMLA~~ ✅ BİTTİ, YAPMA

**Bu görev artık gerekli değil; GPU oturumu 2026-08-02'de tamamladı.**

4. turda `.gitattributes` eklenmiş ve `git add --renormalize` koşulmuştu, ama
yalnızca git'in sakladığı düzelmişti; çalışma ağacı CRLF kalmıştı. GPU oturumu
başka bir iş sırasında index'i yeniden kurunca 30 dosya "değişmiş" göründü.
Kontrol edildi: **30 dosyanın 30'unda içerik farkı sıfır**, fark yalnızca satır
sonu. `git checkout -- .` ile çalışma ağacı LF'e çekildi.

Doğrulandı: `hfp/core/*.py`, `pyproject.toml`, `README.md` → LF; `git status`
temiz; `compileall` ve notebook AST parse'ı geçiyor.

**Yapman gereken tek şey:** raporunda bunu teyit et (`file hfp/core/hfp_utils.py`
LF demeli). Yeniden normalize etmeye kalkma.

---

## GÖREV 3 — `hf_upload` üretilebilir kopyaları (KULLANICI ONAYINA BAĞLI)

4. tur `build_hf_release.py`'yi tamamladı ve üretilen paketi mevcut `hf_release/`
ile karşılaştırdı; beklenen bayatlık dışında fark çıkmadı. 12 üretilebilir `.py`
kopyasının silinmesi **önerildi ama uygulanmadı** — kullanıcı onayı bekliyor.

**Bu turda da uygulama.** Kullanıcı açıkça "sil" demediyse dokunma. Dediyse:

- Silinecek: `hf_upload/*.py` ve `hf_upload/hf_release/*.py` — **yalnız
  `build_hf_release.py`'nin `CANONICAL_FILES`'ında yer alanlar.**
- **Kalacak:** elle yazılmış 8 varlık (`GRAFT_MODEL_CARD.md`, `README.md`,
  `hf_release/README.md`, `make_hf_checkpoint.py`, `fix_hf.py`, `config.json`,
  `generation_config.json`, `YAYIN_ADIMLARI.md`). Bunlar `a5ba92a` ile git
  takibinde ve `GRAFT_MODEL_CARD.md` Kapı 2'nin lisans-uyum belgesi.
- Silmeden **önce** `python scripts/build_hf_release.py` ile paketi bir kez daha
  üret ve `dist/` altında hazır olduğunu doğrula. Üretilemiyorsa silme.

---

## Sınıf ayrımı (aynen geçerli)

- **SINIF 1 mekanik** → uygula. Objektif, doğrulanabilir, yargı gerektirmez.
- **SINIF 2 iddia/bilim** → raporla, **uygulama.**
- **SINIF 3 tasarım** → öner, **uygulama.**
- **Emin değilsen Sınıf 2 say ve uygulama.**

## Biriken sert kurallar

- **Yeşil mühür, kapsamadığı şey hakkında hiçbir şey söylemez.**
- **Takipsiz yolda yapılan değişiklik "uygulandı" sayılmaz.**
- **Bir Sınıf 1 partisi bitince kombinasyonu doğrula**, tek tek değil.
- **Fark raporlarken iki yolu ve satır sonu normalizasyonunu belirt.**
  Format: `A ↔ B: N satır (satır sonu normalize)`.
- **Bir değişmez tutmuyorsa kodu düzeltip geçme** — dur, raporla.
- **Sana verilen bir gerekçe yanlış olabilir.** 3. turda oldu (CRLF). Bir görevin
  dayandığı ölçümü kendin doğrulayabiliyorsan doğrula; tutmuyorsa uygulamadan
  önce söyle.

## Dosya sahipliği

| yol | sahip |
|---|---|
| `hfp/`, kök scriptler, `.github/`, `.gitattributes`, `.gitignore`, `pyproject.toml` | **SEN** |
| `scripts/`, `reproduce.py`, `review_scripts/` | **SEN** |
| `docs/internal_tr/DENETIM_*.md`, `TEKRARLANABILIRLIK_*.md`, `HF_UPLOAD_ANALIZI.md` | **SEN** |
| `notebooks/` | **GPU oturumu — DOKUNMA.** §37 koşuyor. Oku, raporla. |
| `RESULTS.md` | GPU oturumu — **DOKUNMA**, hata bulursan raporla |
| `docs/drafts/`, `README.md` iddia metinleri, `PROJECT_SUMMARY.md`, `*.tex` | GPU oturumu — **DOKUNMA** |
| `hf_upload/` | Görev 3'teki koşullar dışında **DOKUNMA** |

## Yapmayacakların

- `RESULTS.md`'yi ve `notebooks/`'u düzenleme.
- Negatif sonuçları, erratum'ları, dürüstlük uyarılarını "temizleme".
- Testi zayıflatma: eşik gevşetme, `skip`, assert silme **yasak**.
- GPU işi başlatma — kota §37'de. CPU serbest.
- `git push` yapma.
- Checkpoint (`.pt`) commit etme — `LISANS_KARAR_REHBERI.md` Kapı 2.
- Yeni bağımlılık ekleme — öner, lisansını belirt, uygulama.

## Bitirmeden önce

`python smoke_test.py` + `python review_scripts/verify_claims.py` +
`python review_scripts/verify_graft.py` — üçü de yeşil.

## Raporun başına

Kaç bulgu, sınıf dağılımı, **hangi alanlara hiç bakmadığın**, ve uyguladığın her
değişiklik için **onu hangi doğrulamanın kapsadığı**.

## Commit kuralı

Türkçe, **ASCII**. Ne yapıldı + neden + hangi sınıf. Küçük ve ayrı commit'ler.
