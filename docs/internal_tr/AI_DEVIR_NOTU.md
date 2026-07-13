# AI Devir Notu — HFP Projesi (2026-07-13)

> Bu belge, projeye devam edecek asistan (Claude/Fable) için önceki oturumun
> devir teslimidir. Okuma sırası: **bu belge → SONRAKI_ADIMLAR_PLANI.md →
> RESULTS.md → GPU_ROADMAP.md**. Kod detayı gerekirse:
> `hfp/core/hfp_bulk_state.py` (çekirdek) ve `hfp/models/grafting.py` (Faz 3).

## 1. Tek paragrafta proje

HFP: pencereli lokal attention + katman-başına O(1) özyineli bellek (M, z)
mimarisi. Ayırt edici mekanizma kübik-plato retention (`cubic_flux`);
kanıtlanmış reçete bileşenleri DPFP (kapasite) ve cubic (WikiText-2'de
`cubic+additive+dpfp` PPL 183.6, 3 seed). Şu an iki cephe var:
(A) dış baseline/karar deneyleri, (B) Qwen2.5-1.5B'ye grafting (Faz 3).

## 2. ŞU AN KOŞAN DENEYLER (sonuçları kullanıcı yapıştıracak)

1. **Kaggle/Colab — `notebooks/colab_gla_benchmark_v3.ipynb`**
   - Görev A TAMAMLANDI: GLA en iyi LR 3e-4, seed-0 val 5.4575 (PPL≈234);
     seed 1-2 koşuldu, değerleri analiz hücresi okuyacak.
     HFP-best 183.6 → K1 büyük ihtimalle GEÇECEK (resmî hüküm analiz hücresinden).
   - Görev B KOŞUYOR: `cubic+additive+dpfp` vs `cubic+delta+dpfp` vs GLA;
     seq 256'da eğit → eval@{256,1024,2048}. K2 kararı eval-2048'de >2 SE.
2. **Colab — `notebooks/colab_graft_qwen_v2.ipynb`** (grafting)
   - Smoke 6/6 geçti; Qwen graft edildi (13/28 katman, ~325K param);
     teacher yolu birebir doğrulandı. Zero-shot sanity (düzeltilmiş PPL ile)
     ve Stage 1/2 süreçte. Checkpoint'ler Colab `/content/hfp_graft_*.pt`.

## 3. Sonuçlar gelince yapılacaklar (önceden kayıtlı — değiştirme)

- Karar kapıları **SONRAKI_ADIMLAR_PLANI.md §1**'de: K1 (GLA), K2 (yazım
  kuralı reçetesi), K3 (grafting a-e). Kriterler ÖNCEDEN yazıldı;
  sonuca göre kriter yazmak yok.
- Doldurulacak placeholder'lar: `docs/paper3_ml_architecture.tex` ve
  `hf_upload/GRAFT_MODEL_CARD.md` içindeki `[SONUC:...]` alanları.
- Güncellenecek: RESULTS.md (K1/K2 tabloları + §8 reçete kilidi),
  `docs/tr/DENEY_SONUCLARI.md` (Ek 19+), GPU_ROADMAP §10 checklist.
- Grafting checkpoint analizi: `review_scripts/alpha_gate_analysis.py`
  (α dağılımı → "arşiv vs çalışma-belleği kafaları", K3-c).
  Kullanıcı .pt dosyasını proje klasörüne atarsa analizi SEN koş.
- Stage 2 sonrası benchmark: `review_scripts/graft_benchmark_eval.py`
  (MMLU/HellaSwag, −%5 kriteri).

## 4. Bu oturumda öğrenilen dersler (tekrar yaşama)

- **Metrik hijyeni:** HF modelleri `labels`'ı İÇERİDE kaydırır — dışarıdan
  kaydırılmış y verme (çift-kaydırma PPL 17922 artefaktı yaşandı). Kendi
  modellerimiz (GLAForCausalLM, HFPForCausalLM eğitim wrapperı) kaydırMAZ.
  Her yeni eval yolunda önce bilinen bir referans değeri yeniden üret.
- **Training-length cliff LM'de de geçerli** (RESULTS §11): uzun bağlamda
  EĞİTME; kısa eğit → uzun değerlendir.
- **Naif GLA baseline'ı LM ölçeğinde diverge eder:** çıkış LayerNorm + pre-LN
  + 1/√H logit ölçeği üçü birden gerekti (CHANGELOG v2.2'de dürüst not).
- **Kullanıcı akışı:** Colab/Kaggle ücretsiz katman kullanıyor. Notebook
  sürüm karmaşası çok yaşandı → notebook'ları KENDİ KENDİNE YETEN yaz
  (kritik kod gömülü), kullanıcıya "File → Upload notebook" ile doğrudan
  yükletmek en sağlamı. GPU assert'i ve NaN erken-durdurmayı koru (bir kez
  15.5 saat CPU'da boşa koştu). Kullanıcıya adım yıkma; tek hücrelik,
  kopyala-yapıştır çözümler ver; dosya klasördeyse işi kendin yap.
- **Sandbox/git tuhaflıkları:** mount senkronu gecikebilir (yazdığın dosya
  bash'te eski görünebilir — birkaç sn bekle); git index ara ara bozulur
  (`rm .git/index && git reset` onarır); `git status`'taki ~40 dosyalık "M"
  listesi CRLF gürültüsüdür, commit'e KARIŞTIRMA (bir kez karıştı, reset
  gerekti). Commit/push'un güvenilir yolu kullanıcının kendi terminali.
- **Güvenlik:** kullanıcının HF token'ı sohbete/traceback'lere sızdı;
  İPTAL EDİLMESİ hâlâ hatırlatılmalı. Token'ları asla dosyaya yazma;
  `getpass().strip()` kullan (sondaki boşluk 'Illegal header' hatası verdi).

## 5. Kullanıcı tarafında bekleyenler (nazikçe takip et)

1. `git push` — lokalde commit'ler birikmiş olabilir; ayrıca şu değişiklikler
   commit BEKLİYOR olabilir (git status'a bak): docs güncellemeleri (v2.2
   changelog, RESULTS §10/§11, plan, README grafting bölümü, DENEY_SONUCLARI
   Ek 17-18), temizlik silmeleri (9 dosya), `grafting.py` OOM düzeltmesi,
   yeni scriptler (alpha_gate_analysis, graft_benchmark_eval, graft_smoke),
   `colab_gla_benchmark_v3.ipynb`, `GRAFT_MODEL_CARD.md`, bu belge.
2. HF token rotasyonu (yukarıda).
3. Deney çıktılarının yapıştırılması (§2).

## 6. Reçete durumu (sık karışıyor)

- WikiText-2 (seq 256, 3 seed): additive > delta (183.6 vs 191.2).
- Kullanıcının önceki tercihi delta idi ("uzun bağlamda daha iyi" hipotezi);
  bu hipotez K2 deneyiyle karara bağlanacak — sonuç ne çıkarsa RESULTS §8 +
  grafting default'u ona hizalanır.
- Grafting'de bu tartışma α-gate ile modele bırakıldı (`write_rule="hybrid"`,
  kafa başına öğrenilebilir additive↔delta interpolasyonu; matematik numpy
  ile 8 konfigürasyonda sıralı referansa karşı doğrulandı).

## 7. Dosya haritası (bu oturumda eklenenler)

```
hfp/models/grafting.py                    Faz 3 çekirdeği (α-gate, GQA, distilasyon)
review_scripts/graft_smoke.py             graft regresyon testi (Colab'de 6/6)
review_scripts/alpha_gate_analysis.py     checkpoint analizi (GPU'suz)
review_scripts/graft_benchmark_eval.py    MMLU/HellaSwag kıyası (−%5 kriteri)
notebooks/colab_graft_qwen_v2.ipynb       grafting hattı (kanonik)
notebooks/colab_gla_benchmark_v3.ipynb    GLA + K2 karar deneyi (kanonik, self-contained)
hf_upload/GRAFT_MODEL_CARD.md             yayın taslağı ([SONUC] alanlı)
docs/internal_tr/SONRAKI_ADIMLAR_PLANI.md ana plan (karar kapıları burada)
docs/internal_tr/AI_DEVIR_NOTU.md         bu belge
```
