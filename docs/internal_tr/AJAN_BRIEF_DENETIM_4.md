# Ajan brifingi — denetim ajanı, 4. tur

*Hazırlanma: 2026-08-02, HEAD `a5ba92a`. Ajan bu repoyu proje kökü olarak açıyor;
yollar repo köküne göreli.*

---

## Prompt (kopyala-yapıştır)

Bu repoda teknik denetim ve bakım yapıyorsun. Proje kökündeki `AGENTS.md`'yi oku —
**iki kural ödünsüz:** bilimsel dürüstlük ve lisans/IP koruması.

**Bu bir araştırma reposu.** Negatif sonuçlar, erratum'lar ve dürüstlük uyarıları
kasıtlıdır; "temizlenmez".

### 3. turun sonucu: sen haklıydın, brief yanlıştı

3. tur brief'i sana "`hf_upload` 528/232 satır sapıyor, raporunu düzelt" dedi.
**O iddia yanlıştı ve hata brief'i yazan taraftaydı.** Sayılar CRLF/LF
artefaktıydı: `hfp/core/bulk_trigger_decoder.py` ve `hfp_utils.py` CRLF,
`hf_upload/` kopyaları LF. `diff` bütün dosyayı değişmiş sayıyor — 264×2 = 528,
116×2 = 232, tam olarak dosya uzunluğunun iki katı.

Satır sonu normalize edilince (md5 ile doğrulandı) senin 2. tur matrisin
**doğruydu**. `DENETIM_2026-08.md`'ye 3. turda yazdığın "düzeltme notu" doğru bir
ifadeyi düzeltiyor; **geri alınmalı** (silinmeyecek — geri çekme notu olarak
yazılacak, bu projede erratum silinmez).

Geri çekme kaydı: commit `c0e1795`.

**Buradan çıkan kural:** bir fark rapor ederken hangi iki yol arasında olduğunu
**ve satır sonu farkının içerik farkından ayrıldığını** yaz.
Format: `A ↔ B: N satır (satır sonu normalize)`.

---

## GÖREV 1 — Satır sonlarını normalize et (kök sebep)

Bütün bu karışıklığı yaratan şey: `hfp/core/` içinde satır sonları **karışık**.
`bulk_trigger_decoder.py` ve `hfp_utils.py` CRLF, `hfp_bulk_state.py` LF.
Bir gün daha kimse hayalet diff kovalamasın.

- Kök `.gitattributes` ekle: `.py`, `.md`, `.yml`, `.json`, `.txt` için
  `text eol=lf`; ikili uzantılar (`*.pt`, `*.safetensors`, `*.png`) için `binary`.
- Depodaki mevcut dosyaları normalize et (`git add --renormalize .`).
- **Kontrol et:** normalize sonrası `smoke_test.py` + `verify_claims.py` +
  `verify_graft.py` üçü de yeşil kalmalı. Satır sonu değişimi davranışı
  etkilememeli; etkilerse **dur ve raporla.**
- Raporda: normalize öncesi/sonrası kaç dosyanın satır sonu değişti.

`hf_upload/` altındaki **takipsiz** `.py` kopyalarına dokunma.

---

## GÖREV 2 — `build_hf_release.py`'yi tamamla, sonra kopyaları sil

**Durum:** script şu an yalnızca 6 `.py` + `LICENSE` topluyor
(`CANONICAL_FILES`). Ama yayına hazır bir HF paketi bundan fazlasını istiyor ve
o dosyalar `hfp/`'den üretilemez — elle yazılmışlar:

```
GRAFT_MODEL_CARD.md                94 satır
README.md                         120
hf_release/README.md              103
make_hf_checkpoint.py              69
hf_release/config.json             40
YAYIN_ADIMLARI.md                  30
fix_hf.py                          22
hf_release/generation_config.json   8
```

Bunlar `a5ba92a` ile **git takibine alındı** (`.gitignore`'da açık negatif
kalıplarla), yani artık script'in çekebileceği takipli bir kaynakları var.

`GRAFT_MODEL_CARD.md` özellikle kritik: `docs/internal_tr/LISANS_KARAR_REHBERI.md`
**Kapı 2** onu ismen işaret ediyor (`license: other`,
`license_name: apache-2.0-base-agpl-3.0-adapter`, `base_model: Qwen/Qwen2.5-1.5B`).
Ağırlık yayınının lisans-uyum belgesi bu.

**2a.** `build_hf_release.py`'yi bu varlıkları da paketleyecek şekilde genişlet.
Kaynak: `hf_upload/` altındaki **takipli** dosyalar. Üretilen ile elle yazılan
kaynakları script içinde açıkça ayır (`CANONICAL_FILES` / `AUTHORED_ASSETS`).

**2b.** Script'i koş, ürettiği paketi mevcut `hf_upload/hf_release/` ile
**satır sonu normalize ederek** karşılaştır. Beklenen tek fark, bilinen
bayatlık: `hfp_bulk_state` 210 satır, `hfp_config` 15 satır (`hf_release` geride).
Başka fark çıkarsa **dur ve raporla** — o, henüz anlamadığımız bir şey demektir.

**2c.** 2b temiz geçerse: takipsiz **üretilebilir** `.py` kopyalarının silinmesini
öner (12 dosya; `hf_upload/*.py` ve `hf_upload/hf_release/*.py`). **Silme** —
kullanıcı onayına bağlı. Elle yazılmış 8 dosya **kalacak**.

---

## GÖREV 3 — Kırık üretim yolları (§16, §17) — DİKKATLİ

`reproduce.py` üç kırık yol buldu. Bu, "bu sonuçlar bugün yeniden üretilemiyor"
demek ve yayın için gerçek bir sorun. Ama düzeltmesi düşündüğünden hassas.

**§16 — `baseline_compare.py`:** script integer SEED beklerken CLI `exp` veriyor
(`ValueError`). Muhtemelen argüman sırası/tipi sorunu. Görece basit.

**§17 — `lifetime_retention.py`:** `torch.set_rng_state` pickle/byte tip
uyuşmazlığı. Büyük olasılıkla checkpoint'in yazıldığı torch sürümü ile bugünkü
sürüm arasındaki fark. **Bunu düzeltmek koşunun rastgeleliğini değiştirebilir** —
yani `RESULTS.md`'deki sayıları yeniden üretmeyebilirsin.

**Yordam (ikisi için de):**
1. Önce **tanıla ve yaz**: hata tam olarak ne, ne zaman girmiş (`git log -p`),
   düzeltmenin koşu davranışını değiştirip değiştirmeyeceği.
2. Düzeltme davranışı **değiştirmiyorsa** (ör. sadece argüman ayrıştırma) →
   Sınıf 1, uygula.
3. Düzeltme davranışı **değiştirebiliyorsa** (RNG akışı, seed, veri sırası) →
   **Sınıf 3, uygulama.** Öner, gerekçesini yaz, kullanıcı karar versin.
4. Hiçbir durumda "düzelttim, artık koşuyor" deyip sayıların `RESULTS.md` ile
   uyuştuğunu **iddia etme** — o ayrı bir doğrulama ve bu turun kapsamında değil.

**§4 — `interference_eval.py`:** bu kırık değil. `lg_0_final.pt` checkpoint'ini
istiyor, yani zincirin önceki halkası (§3 eğitimi) koşulmamış. Kod sorunu değil,
**bağımlılık** sorunu — Görev 4'e ait.

**Sahiplik notu:** önceki brief'ler `review_scripts/`'i ikinci bir ajana
ayırmıştı; **o ajan hiç başlatılmadı.** Rezervasyon kaldırıldı: `review_scripts/`
tamamı sende, ama **yalnız Sınıf 1** kapsamında. Bir dosyanın davranışını
değiştirmek gerekiyorsa Sınıf 3'tür, önerirsin.

---

## GÖREV 4 — `reproduce.py` bağımlılıkları bilsin

§4, §3'ün ürettiği checkpoint'e bağlı ve şu an sessizce `FAIL` veriyor. Bu
yanıltıcı: kırık bir yol ile "önkoşulu koşulmamış" bir yol farklı şeyler.

- Her bölüm için opsiyonel `depends_on` ve `requires_file` tanımla.
- Önkoşulu eksikse durum **`ATLANDI-ONKOSUL`** olsun, `FAIL` değil.
- `--with-deps` bayrağı: önkoşulları da sırayla koşsun (ağır olabilir,
  varsayılan kapalı).
- Özet tabloda `FAIL` (gerçek kırık) ile `ATLANDI-ONKOSUL` ayrı sayılsın.

---

## Sınıf ayrımı (aynen geçerli)

- **SINIF 1 mekanik** → uygula. Objektif, doğrulanabilir, yargı gerektirmez.
- **SINIF 2 iddia/bilim** → raporla, **uygulama.**
- **SINIF 3 tasarım** → öner, **uygulama.** Davranışı ya da deney semantiğini
  değiştiren her şey.
- **Emin değilsen Sınıf 2 say ve uygulama.**

## Biriken sert kurallar

- **Yeşil mühür, kapsamadığı şey hakkında hiçbir şey söylemez.** Uyguladığın her
  değişiklik için "bunu hangi doğrulama kapsıyor" sorusunu cevapla.
- **Takipsiz yolda yapılan değişiklik "uygulandı" sayılmaz.**
- **Bir Sınıf 1 partisi bitince kombinasyonu doğrula**, tek tek değil.
- **Fark raporlarken iki yolu ve satır sonu normalizasyonunu belirt.**
- **Bir değişmez tutmuyorsa kodu düzeltip geçme** — dur, raporla.
- **Sana verilen bir gerekçe yanlış olabilir.** 3. turda oldu. Bir görevin
  dayandığı ölçümü kendin doğrulayabiliyorsan doğrula; tutmuyorsa görevi
  uygulamadan önce söyle.

## Dosya sahipliği

| yol | sahip |
|---|---|
| `hfp/`, kök scriptler, `.github/`, `.gitattributes`, `.gitignore`, `pyproject.toml` | **SEN** |
| `scripts/`, `reproduce.py`, `review_scripts/verify_graft.py` | **SEN** |
| `review_scripts/baseline_compare.py`, `lifetime_retention.py`, `interference_eval.py` | **SEN** (bu tur devredildi) |
| `docs/internal_tr/DENETIM_*.md`, `TEKRARLANABILIRLIK_*.md`, `HF_UPLOAD_ANALIZI.md` | **SEN** |
| `README.md` (iddia metinleri hariç), `NASIL_CALISTIRILIR.md` | **SEN** (yalnız Sınıf 1) |
| `RESULTS.md` | GPU oturumu — **DOKUNMA**, hata bulursan raporla |
| `notebooks/` | GPU oturumu — **DOKUNMA** |
| `review_scripts/` (diğer dosyalar) | **SEN** — Sınıf 1 ile sınırlı |
| `docs/drafts/`, `docs/PROJECT_SUMMARY.md`, `docs/*.tex` | GPU oturumu — **DOKUNMA** |
| `README.md` iddia metinleri (satır ~69-77, ~120) | GPU oturumu — **DOKUNMA** |
| `hf_upload/` takipsiz `.py` kopyaları | **dokunma** — yalnız oku |

## Yapmayacakların

- `RESULTS.md`'yi düzenleme.
- Takipsiz yollarda değişiklik uygulama.
- Negatif sonuçları, erratum'ları, dürüstlük uyarılarını "temizleme".
- Testi zayıflatma: eşik gevşetme, `skip`, assert silme, hardcode'lu "beklenen"
  değer gömme **yasak**.
- GPU/compute işi başlatma. Kota §36'ya ayrılmış. CPU serbest.
- `git push` yapma.
- Checkpoint (`.pt`) commit etme — `LISANS_KARAR_REHBERI.md` Kapı 2.
- Yeni bağımlılık ekleme — öner, lisansını belirt, uygulama.
- Lisans başlıklarına, telif bildirimlerine, atıf zincirine dokunma.

## Bitirmeden önce

`python smoke_test.py` + `python review_scripts/verify_claims.py` +
`python review_scripts/verify_graft.py` — üçü de yeşil.

## Raporun başına

Kaç bulgu, sınıf dağılımı, **hangi alanlara hiç bakmadığın**, ve uyguladığın her
değişiklik için **onu hangi doğrulamanın kapsadığı**.

## Commit kuralı

Türkçe, **ASCII**. Ne yapıldı + neden + hangi sınıf. Küçük ve ayrı commit'ler.
