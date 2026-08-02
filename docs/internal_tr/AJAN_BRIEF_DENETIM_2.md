# Ajan brifingi — denetim ajanı, 2. tur

*Hazırlanma: 2026-08-02, HEAD `562a703`. Birinci turun raporu
`docs/internal_tr/DENETIM_2026-08.md`. Bu brief onun devamıdır.*

---

## Prompt (kopyala-yapıştır)

Sen HFP projesinde teknik denetim yapıyorsun: `C:\Users\yilma\Documents\HFP_Project`.
Bu senin **ikinci turun**. Birinci turda beş Sınıf 1 düzeltmesi uyguladın; ikisi
sorunluydu ve aşağıda kanıtıyla yazılı. Kırgınlık değil, kural çıkarımı için.

**Bu bir araştırma reposu.** Negatif sonuçlar, erratum'lar ve dürüstlük uyarıları
kasıtlıdır; "temizlenmez". İki ajan daha aynı repoda çalışıyor; dosya sahipliğine
harfiyen uy.

### Birinci turdan çıkan iki ders

**1. İki commit'in birbiriyle çelişip çelişmediğini kontrol etmedin.**
`02f7149` `tinyshakespeare.txt`'yi git index'inden çıkardı. `22988c5` CI'a
`python train.py --max_iters 2` ekledi. `train.py:101-103` o dosya yoksa
`FileNotFoundError` fırlatıyor ve dosya `.gitignore:45`'te — temiz checkout'ta
yok. Bir sonraki push'ta CI kırmızı olacaktı. (`562a703` ile düzeltildi.)

Raporun "smoke_test + verify_claims tam yeşil" diyordu ve **doğruydu** — ama o
iki bekçi `train.py`'yi hiç çalıştırmıyor. **Yeşil mühür, kapsamadığı şey
hakkında hiçbir şey söylemez.**

**2. `.gitignore`'daki bir dizinde "uygulama" yaptın.**
`hf_upload/` `.gitignore:58`'de, yani **takipsiz**. "Eşitlendi" dediğin iki
dosya (`hfp_bulk_state.py`, `hfp_config.py`) hiçbir commit'te yok ve olamaz da.
Değişiklik kayıtsız ve geri alınamaz durumda. Ayrıca BEKLEYEN #9 dört dosya
sayıyor; ikisi hâlâ sapıyor (`bulk_trigger_decoder` 528 satır, `hfp_utils` 232
satır), yani #9 kapanmadı. Üstelik "hangi taraf kanonik" sorusu bir **Sınıf 3**
yargısıydı, Sınıf 1 gibi uygulandı.

### Bu turda geçerli EK KURALLAR

- **Takipsiz (gitignored) bir yolda yapılan değişiklik "uygulandı" sayılmaz.**
  Böyle bir yola dokunman gerekiyorsa: dur, raporla, kullanıcıya sor.
- **Bir giriş noktasını CI'a eklemeden önce veri/ağ bağımlılığını doğrula.**
  Yerelde çalışması yetmez; temiz checkout'ta ne olacağını düşün.
- **Bir Sınıf 1 partisi bitince kombinasyonu doğrula**, tek tek değil.
- **Doğrulaman, değiştirdiğin şeyi kapsamalı.** Kapsamıyorsa raporda açıkça
  "bu değişiklik mevcut bekçilerin kapsamı dışında" yaz.
- Sınıf ayrımı (1 mekanik / 2 iddia-bilim / 3 tasarım) aynen geçerli.
  **Emin değilsen Sınıf 2 say ve uygulama.**

---

## GÖREV 1 — Birinci turun açık ucunu kapat (ÖNCE BU)

**1a.** `hf_upload/`'da ne değiştirdiğini **yeniden inşa et ve belgele.** Git
kaydı yok; `hf_upload/hf_release/` altında 2026-07-05 tarihli eski kopyalar var
(`hfp_bulk_state.py`, `hfp_config.py`). Bunlarla mevcut hâli karşılaştır, farkı
rapora yaz. **Hiçbir dosyayı geri alma ya da tekrar değiştirme.**

**1b.** Kalan iki sapmayı **analiz et**: `bulk_trigger_decoder` (528 satır),
`hfp_utils` (232 satır). Her biri için: fark ne, hangi taraf daha yeni, sapma
işlevsel mi kozmetik mi. **Eşitleme yapma** — bu Sınıf 3.

**1c.** Şu soruyu gerekçeli olarak kullanıcıya sor: `hf_upload/` `.gitignore`'dan
çıkarılıp takibe alınmalı mı? Takipsiz kaldığı sürece her senkron çalışması
kayıtsız ve tekrarlanamaz. Karar kullanıcının; sen seçenekleri ve bedellerini yaz.

---

## GÖREV 2 — `grafting.py` için test yaz (BU TURUN ASIL İŞİ)

**Bulgu:** `hfp/models/grafting.py` **519 satır** ve **hiçbir testin kapsamında
değil.** Ne `smoke_test.py` ne `review_scripts/verify_claims.py` ona dokunuyor.
Oysa §15'ten §36'ya kadar bütün Qwen graft sonuçları bu modüle dayanıyor.

Neden kritik: `enable_streaming` / `reset_streaming`, RESULTS'un **erratum'unun
ve §30'un %0 bulgusunun** üzerinde durduğu mekanizma. `reset_streaming` sessizce
state'i temizlemeseydi §30'un sonucu ve §36'nın bütün tasarımı yanlış olurdu —
ve bunu yakalayacak hiçbir şey yok.

`review_scripts/verify_graft.py` yaz. **CPU'da, ağ erişimi olmadan, ~1 dakikada**
koşmalı. Ağ gerektirmeme şartı için: ağırlık indirme, `from_config` ile küçük
rastgele bir model kurarak atlanır (`AutoConfig` + `AutoModelForCausalLM.from_config`,
2-4 katman, hidden 64). Ağırlık indirmen **gerekmiyor** — test yapıyı sınıyor,
kaliteyi değil.

En az şu değişmezler:

**T1 — `graft_llama` doğru katmanları değiştiriyor.** İstenen indekslerdeki
`self_attn` `HFPGraftAttention` oldu, diğerleri olmadı. Eğitilebilir parametre
sayısı beklenenle uyuşuyor; graft dışındaki her şey donuk (`requires_grad=False`).

**T2 — `reset_streaming` state'i GERÇEKTEN siliyor.** Protokol: streaming açık,
A dizisini geçir, B dizisini geçir → çıktı X. Sonra `reset_streaming`, sonra
yalnız B → çıktı Y. Taze bir modelde yalnız B → çıktı Z. **Y ile Z bit-bit aynı
olmalı, X onlardan farklı olmalı.** Bu iki koşul birlikte hem taşımanın hem
sıfırlamanın çalıştığını kanıtlar. Biri bile tutmazsa §30/§36 geçersizdir.

**T3 — `enable_streaming(m, False)` state taşımayı kapatıyor** ve `_stream_state`
temizleniyor.

**T4 — `set_graft_mode` modları birbirinden farklı davranıyor.**
`student` / `teacher` / `teacher_forcing` aynı girdide farklı çıktı üretmeli.
`teacher` modunda çıktı, graft hiç uygulanmamış modelinkiyle aynı olmalı —
yani `teacher` gerçekten orijinal attention'a düşüyor. (Bu, §36'nın ablasyon
kolunun dayanağı.)

**T5 — Checkpoint yükleme sessizce başarısız olamaz.** `strict=False` isim
uyuşmazlığında **hiçbir şey yüklemez** ve bu, "hafıza yok" ile "ağırlıklar
eğitimsiz"i karıştırır (§30 hücre 7'nin varlık sebebi). Test: state_dict kaydet,
anahtarları kasıtlı boz, `strict=False` ile yükle → eşleşen tensör sayısının
sıfır olduğunu **tespit eden** bir yardımcı fonksiyonun bu durumu yakaladığını
doğrula. Yardımcıyı da bu dosyaya yaz; `notebooks/memory_capability_v30.ipynb`
hücre 7'deki mantık şablon.

Bitince: `smoke_test.py` + `verify_claims.py` + yeni test yeşil olmalı, ve
`verify_graft.py`'yi CI'a ekle (koşu süresini ölç, 30 dk limitini gözet).

**Bir uyarı:** bu testleri yazarken bir değişmezin **tutmadığını** bulursan,
bu Sınıf 1 değildir. Kodu düzeltme — **dur ve derhal raporla.** Tutmayan bir
değişmez, yayınlanmış sonuçların geçerliliğini ilgilendirir.

---

## GÖREV 3 — Tekrarlanabilirlik denetimi (GÖREV 2 bitince)

`RESULTS.md`'de **38 bölüm** var; sondaki "Reproduction" bölümü **6 komut**
listeliyor. Aradaki boşluk yayın için gerçek bir sorun: tekrarlanabilirlik bu
alanda nadir ve projenin iddia ettiği güçlü yanlardan biri.

Her bölüm için: bu sonucu **bugün** yeniden üretecek bir yol var mı?
Çıktı: `docs/internal_tr/TEKRARLANABILIRLIK_2026-08.md`

| § | başlık | üretim yolu | durum | eksik olan |
|---|---|---|---|---|

`durum` ∈ {KOŞULABİLİR, KOŞULAMAZ-CHECKPOINT, KOŞULAMAZ-GPU, KOŞULAMAZ-VERİ,
KOMUT-YOK, BOZUK}. `üretim yolu` = script + tam komut, ya da notebook adı.

Kuralları:
- Komutun **gerçekten çalıştığını** doğrula (kısa parametrelerle, CPU'da).
  Çalıştırmadan "KOŞULABİLİR" yazma.
- Checkpoint gerektiren bölümler için: hangi checkpoint, repoda var mı
  (`.gitignore` `*.pt` içeriyor — çoğu yok), yoksa **KOŞULAMAZ-CHECKPOINT**.
- Sonuçları **yeniden üretmeye çalışma**, sayıları doğrulama. Bu bir *yol*
  denetimi, *sonuç* denetimi değil. Sayılara dokunmak Sınıf 2'dir.

---

## Dosya sahipliği — İHLAL ETME

| dosya | sahip |
|---|---|
| `hfp/`, kök scriptler, `.github/`, `pyproject.toml`, `requirements.txt` | **SEN** |
| `review_scripts/verify_graft.py` (yeni) | **SEN** |
| `README.md`, `NASIL_CALISTIRILIR.md` | **SEN** (yalnız Sınıf 1) |
| `RESULTS.md` | GPU oturumu — **DOKUNMA**, hata bulursan raporla |
| `notebooks/` | GPU oturumu — **DOKUNMA** (okuyabilirsin) |
| `review_scripts/` (diğer dosyalar) | 2. ajan — **DOKUNMA** (okuyabilirsin) |
| `docs/` (kendi rapor dosyaların hariç) | 2. ajan — **DOKUNMA** |
| `hf_upload/` | **hiç kimse** — takipsiz; yalnız oku ve raporla |

## Yapmayacakların

- **`RESULTS.md`'yi düzenleme.** Otorite deney kaydı.
- **Takipsiz yollarda değişiklik uygulama** (`hf_upload/`, `_legacy_reference/`,
  `_archive_old/`, `checkpoints/`). Oku, raporla, sor.
- **Negatif sonuçları, erratum'ları, dürüstlük uyarılarını "temizleme".**
- **Testi zayıflatma.** Eşik gevşetme, `skip`, assert silme, hardcode'lu
  "beklenen" değer gömme yasak. Test kırmızıysa kod düzeltilir.
- **Bir değişmez tutmuyorsa kodu düzeltip geçme** — dur, raporla.
- **GPU/compute işi başlatma.** Kota §36'ya ayrılmış. CPU serbest.
- **`git push` yapma.** Commit at, push kullanıcıya ait.
- **Checkpoint (`.pt`) commit etme** — `LISANS_KARAR_REHBERI.md` Kapı 2.
- **Yeni bağımlılık ekleme** — öner, lisansını belirt, uygulama.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

## Raporun başına

Kaç bulgu, sınıf dağılımı, **hangi alanlara hiç bakmadığın**, ve her uyguladığın
değişiklik için **onu hangi doğrulamanın kapsadığı.** Kapsayan yoksa öyle yaz.
Bakmadığın yeri "temiz" diye raporlama.

## Commit kuralı

Türkçe, **ASCII** (Türkçe karakter kullanma, git log bozuluyor).
Ne yapıldı + neden + hangi sınıf. Küçük ve ayrı commit'ler; geri alınabilir olsun.

## Belirsizlikte

Dur ve kullanıcıya sor. Bu projede hızlı ve yanlış, yavaş ve doğrudan kötüdür.
