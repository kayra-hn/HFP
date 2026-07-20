# HFP — Sonraki Adımlar Planı (Deney-Sonrası Yol Haritası)

> Hazırlanma: 2026-07-12. Bağlam: GLA benchmark v2 (Kaggle) ve Qwen2.5-1.5B
> grafting (Colab) koşuları başlatılmak üzere. Bu belge, sonuçlar geldiğinde
> ne yapılacağını ÖNCEDEN sabitler (post-hoc rasyonalizasyon yok) ve sonrasının
> tam sırasını verir. GPU_ROADMAP.md'nin devamı niteliğindedir.

---

## 0a. Durum güncellemesi (2026-07-12, akşam)

- **Grafting:** `graft_smoke.py` Colab T4'te 6/6 GEÇTİ. Qwen2.5-1.5B graft edildi
  (13/28 katman, 324.805 eğitilebilir param). Teacher yolu birebir doğrulandı.
  İlk turda bulunan üç sorun çözüldü: PPL çift-kaydırma metriği, Stage-1 OOM
  (rec_block 16 + katman-anında backward), HF token boşluğu. Sıradaki: doğru
  metrikle zero-shot sanity → Stage 1.
- **GLA baseline:** naif implementasyon LM ölçeğinde diverge etti; üç aşamada
  stabilize edildi (çıkış LayerNorm → pre-LN → 1/√H logit ölçeği). Sağlamlık
  testi: loss 15.9→14.2 düzgün düşüş, patlama yok. `colab_gla_benchmark_v3.ipynb`
  kendi kendine yeten sürüm (git zincirinden bağımsız, doğrudan yükleme).
  Bu stabilizasyon zorunluluğu dürüst-not olarak CHANGELOG'a işlendi.
- **v1 Görev B bulgusu RESULTS §11'e işlendi:** training-length cliff LM'de de
  geçerli (3 seed, iki kol da ln|V| platosunda) → K2 deneyi train-short →
  infer-long olarak yeniden tasarlandı.
- **Kota notu:** ilk Kaggle oturumu yanlışlıkla CPU'da koştu (~15.5 saat, geçersiz).
  GPU assert'leri eklendi; tekrarlanamaz.

## 0. Anlık durum

**Koşuda / başlatılacak:**
| İş | Platform | Notebook | Süre tahmini |
|---|---|---|---|
| Görev A: GLA baseline (seq 256, LR taraması + 3 seed) | Kaggle (arka plan) | `kaggle_gla_lm_benchmark_v2.ipynb` | ~2-3 saat |
| Görev B: delta-vs-additive, train@256 → eval@{256,1024,2048} | Kaggle (aynı notebook) | 〃 | ~4-6 saat |
| Grafting Stage 1+2 + validasyon | Colab (T4) | `colab_graft_qwen_v3_kaggle.ipynb` | ~5-8 saat |

**Tamamlanmış ve geçerli:**
- WikiText-2 ablasyonu (3 seed): `cubic+additive+dpfp` PPL 183.6 (en iyi, en düşük varyans).
- Uzun-ufuk retention karar deneyi: cubic+dpfp 63.9% vs exp+dpfp 20.7% (>4 SE).
- Uzunluk genellemesi, girişim-sınırlılık, DPFP kapasite ekseni (çok-seed).
- `graft_smoke.py` 6/6 (Colab T4); α-gate melez matematiği sıralı referansa karşı 8 konfigürasyonda birebir doğrulandı.

**Geçersiz sayılan (v1 artefaktları — hiçbir karara girmeyecek):**
- v1 GLA sonuçları (çıkış normu eksikti → NaN).
- v1 Görev B (seq-1024'te eğitim → training-length cliff; yeni bulgu olarak §3'e not edilecek).
- Çift-kaydırma PPL değerleri (17922 vb.).

---

## 1. Karar kapıları (ÖNCEDEN yazılı kriterler)

### K1 — Aile-standardı (GLA baseline)

> **[DURUM 2026-07-13]** İlk "GEÇTİ" hükmü geri çekildi: metrik artefaktı
> (çift-kaydırma, DENEY_SONUCLARI Ek 21) HFP kollarını skip-one hedefiyle
> koşturmuştu; ayrıca LM config'i penceresiz (tam attention) çıktı.
> Düzeltilmiş tek-seed prob yönü güçlü koruyor (next-token PPL 55.4 vs
> GLA 226.7). Resmî hüküm: FIX M1 + pencereli O(1) config + 3 seed
> yeniden koşumda. Aşağıdaki kriter metni değişmedi.
> HFP-best (`cubic+additive+dpfp`, 183.6), GLA'nın en iyi LR'ındaki 3-seed
> ortalamasının −2 SE bandında veya üstündeyse → "aile-standardı kadar iyi +
> O(1) + ekstra eksenler" konumu DOĞRULANDI.

- **Geçerse:** README + RESULTS'a "GLA-sınıfı baseline'a karşı konumlandı" eklenir; dış iddia serbest.
- **Kalırsa:** fark raporlanır, dürüst deftere yazılır; iddia "GPT-2'den iyi, GLA'nın X PPL gerisinde" olarak daraltılır. GLA'nın hangi bileşeninin (veri-bağımlı kapı?) farkı açtığına dair tek ablasyon denenir (HFP'ye veri-bağımlı decay eklemek zaten `importance_gate` üzerinden kısmen var — analiz önce, kod sonra).

### K2 — Delta uzun-bağlam hipotezi
> Eval 2048'de `cubic+delta+dpfp`, `cubic+additive+dpfp`'yi 3-seed ort. >2 SE
> geçerse → hipotez DOĞRULANDI, resmi reçete delta kalır.
> Aksi halde → resmi reçete **`cubic+additive+dpfp`** olur; delta yalnız
> key-update/streaming nişinde anılır.

- Sonuç ne olursa olsun RESULTS §8 ("current recipe") ve grafting varsayılanı buna göre HİZALANIR — mevcut reçete çelişkisi kapanır.
- α-gate melezi bu karardan bağımsız yaşar: grafting'de model kafa başına kendisi seçiyor (bkz. K3-c).

### K3 — Grafting validasyonu (strateji dok. §4 kriterleri)
- **(a) Zero-shot sanity:** graft sonrası, eğitimsiz PPL < 1000. Kalırsa ağırlık transferi/ölçek hatası → `out_gain` init ve conv kimlik-init kontrol; Stage 1'e GEÇİLMEZ.
- **(b) Kısa-bağlam kaybı:** Stage 2 sonrası WikiText PPL ≤ 1.05 × orijinal Qwen. 1.05-1.20 arası "kabul edilebilir ilk geçiş" (token bütçesi artırılarak kapanabilir); >1.20 ise hibrit oranı düşürülür (13→9 katman).
- **(c) α-gate okuması (bilimsel yan ürün):** Stage 1 sonunda kafa-başına α dağılımı kaydedilir. Beklenti: çoğu kafa additive'e (α<0.3), az sayıda kafa delta'ya (α>0.7) yakınsar → "arşiv kafaları vs çalışma-belleği kafaları" bulgusu. Dağılım tek kutupluysa bu da bulgu (yazım kuralı evrensel tercih).
- **(d) Needle:** 2K'da bulmalı (sanity); 8K+ ilk geçişte bulunamazsa başarısızlık değil — token bütçesi yetersizliği olarak yorumlanır, Stage ölçeklemeye girdi olur.
- **(e) VRAM/state:** grafted state boyutu bağlam uzunluğundan bağımsız (otomatik rapor).

---

## 2. Sonuç senaryoları → eylem ağacı

```
K1 geçti + K3(a,b) geçti  → Ana hat: Grafting ölçekleme (Bölüm 4) + yayın hazırlığı (Bölüm 5)
K1 geçti + K3 kaldı       → Grafting hata ayıklama turu (hibrit oranı, LR, token bütçesi); GLA sonucuyla docs güncelle
K1 kaldı + K3 geçti       → Grafting yine de değerli (pretrained yetenek + O(1)); K1 farkının analizi paralel yürür
K1 kaldı + K3 kaldı       → Konsolidasyon: dokümantasyon dürüstlük turu + küçük-ölçek güçlü sonuçlarla OSF/paper güncelle; ölçek denemesi ertelenir
K2 hangi yöne çıkarsa     → Reçete kilitlenir, tüm dokümanlar + grafting default'u hizalanır (yarım gün iş)
```

---

## 3. Dokümantasyon borçları (deneyler biter bitmez, yarım gün)

1. **RESULTS.md yapısal düzeltme:** çift "§5" numaralandırması düzeltilir (WikiText-2 bölümü §10 olur); "current recipe" K2 kararıyla hizalanır.
2. **Yeni bulgular eklenir:**
   - GLA baseline tablosu (K1 sonucu, hangi yöne çıkarsa çıksın).
   - Delta-vs-additive uzun-eval tablosu (K2).
   - *Training-length cliff LM'de de geçerli* notu (v1 Görev B'nin dürüst raporu: seq-1024 eğitimi ln(vocab) platosunda kaldı; train-short → infer-long LM için de zorunlu).
   - Grafting ilk sonuçları (K3 a-e).
3. **DENEY_SONUCLARI.md (TR çeviri)** senkronize edilir.
4. **GPU_ROADMAP.md §10 checklist** işlenir; açık soru 2/4/5/7 durumları güncellenir.
5. **README:** "Status of results" bölümü yeni bulgularla; grafting kullanım örneği eklenir.
6. **osf_companion.tex** → v2.3 derlenir, OSF'e yüklenir.
7. **paper3_ml_architecture.tex:** GLA baseline + grafting bölümü taslağı.

---

## 4. Grafting ölçekleme hattı (K3 geçerse — ana yatırım)

Sıra kaldıraç × maliyete göre:

1. **Token bütçesi büyütme (ilk geçiş 2000+1000 adım ≈ ~10M token; hedef 100-300M):**
   L4/A100 Colab Pro ya da Kaggle P100; veri WikiText-103 → FineWeb-Edu streaming'e geçiş.
   Başarı: needle 8K→32K'ya taşınması, PPL oranının 1.05×'e inmesi.
2. **Hibrit oran taraması:** {%25, %50 (mevcut), %75} HFP katmanı. Ölçüt: PPL kaybı vs KV-cache küçülmesi eğrisi. Strateji dokümanının "en büyük risk" dediği soru bu — tek boyutlu tarama yeter.
3. **RoPE kararının ablasyonu:** bypass (mevcut) vs RoPE'li q/k'yı HFP'ye besleme. Tek koşu, needle+PPL kıyası.
4. **Needle ölçek testi:** 32K → 128K, O(1) yolun asıl şovu. KV'li katmanlar için sliding-window cache (ya da tam-HFP varyant) gerekebilir — 128K'da 13 full-attn katmanın KV'si de büyür; "KV katmanlarına window=4K" hibrit-pratik çözümdür ve dürüstçe raporlanır.
5. **Kısa-bağlam benchmark:** MMLU + HellaSwag (lm-eval-harness) — strateji dok. kriteri: orijinalin −%5 bandı.
6. **Yayın çıktısı:** `Qwen2.5-1.5B-HFP-O1` HF model kartıyla (Apache 2.0 taban + AGPL HFP modülü lisans notu; yalnız HFP parametreleri +~325K param yüklenir, taban model referansla).

## 5. Yayın / iletişim hattı (paralel, düşük efor)

- HF model yayını (üstteki 4.6) + `hf_upload/` akışına grafting checkpoint desteği.
- OSF güncelleme notu (`_arsiv/OSF_GUNCELLEME_NOTU.md (arşivlendi)` şablonuyla).
- README'ye "Grafting quickstart" — üç satırlık kullanım.
- (Opsiyonel) kısa teknik blog/thread: "O(1) bellek pretrained modele nasıl aşılanır" — dürüstlük çerçevesi korunarak.

## 6. Orta vade (FUTURE_RESEARCH_HYPOTHESES sırası — ancak 4. bölüm bitince)

Öncelik sırası (test maliyeti × bilgi değeri):
1. **α-gate / M-normu analizi** (hipotez 2.1 "önem-tabanlı unutma"): grafted modelde M'nin L2 norm haritası — neredeyse bedava (checkpoint'ten okunur), makale-değeri yüksek.
2. **Online sıkıştırma** (2.2): M özdeğer spektrumu — yine checkpoint analizi.
3. **Continual learning** (3.1): task-incremental text classification; Mamba/Llama kıyası ancak grafting stabil olunca.
4. **RL kredi ataması** (3.2): en pahalı, en spekülatif — en sona.

## 7. Riskler ve hafifletmeler

| Risk | Belirti | Hafifletme |
|---|---|---|
| Stage 1 MSE düşüyor ama Stage 2 PPL kötü | katman-MSE ile logit hedefi uyumsuz | KL ağırlığı/TEMP taraması; Stage 1'i uzat |
| α-gate hep 0'a çöker | delta yolu hiç öğrenilmiyor | α_init'i -1'e çek; delta'yı 2-3 katmanda sabitle |
| cubic z-scan uzun seq'te yavaş | Stage süreleri 2-3× | `decay_mode="exp"` graft varyantı A/B (cubic kazancı LM'de zaten additive'le geliyordu; graft'ta da test edilmeli) |
| Colab kesintileri | yarım Stage | 500-adım checkpoint zaten var; Kaggle arka plan tercih |
| Kaggle kota (30 sa/hafta) | koşular yarım | Görev A+B tek oturum (~8 sa) planla; grafting Colab'de |
| Token güvenliği | sohbet/traceback sızıntısı | Mevcut HF token İPTAL edilip yenilenecek (yapılacaklar listesinin başında) |

## 8. Önerilen takvim

| Gün | İş |
|---|---|
| 1 | Kaggle: Görev A+B arka plan koşusu. Colab: grafting Stage 1 → K3(a) kontrolü |
| 1-2 | Grafting Stage 2 + validasyon → K3(b-e) |
| 2 | K1/K2 sonuçları okunur → karar kapıları işletilir |
| 2-3 | Dokümantasyon borçları (Bölüm 3) — yarım gün |
| 3+ | Senaryo ağacına göre: ölçekleme (Bölüm 4) ya da hata ayıklama turu |

## 9. Metodoloji hatırlatmaları (değişmez)

- ≥3 seed; tek-seed "tek-seed" etiketlenir.
- Mode-başına LR taraması (özellikle cubic).
- Önceden yazılı kriter; sonuç kritere göre okunur, kriter sonuca göre yazılmaz.
- Negatif sonuç da sonuçtur → dürüst deftere (v1 Görev B cliff bulgusu buna örnek).
- Metrik doğrulaması: her yeni eval yolunda önce bilinen bir referans değer yeniden üretilir (çift-kaydırma dersinden).
