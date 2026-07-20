# HFP — GPU Deney Yol Haritası, Açık Sorular ve Potansiyeller

> Amaç: GPU kiralandığında ne koşulacağını, hangi sırayla, hangi başarı
> kriteriyle ve neyin hâlâ açık olduğunu tek yerde toplamak. Tüm scriptler repo
> kökünde hazır; komutlar Git Bash / Linux kabuğunda çalışır.
> Son güncelleme: 2026-07-12. Deney-sonrası karar kapıları ve devam planı için
> bkz. `SONRAKI_ADIMLAR_PLANI.md` (bu belgenin devamı).

---

## 0. Mevcut durum (özet)

**Kanıtlanan (çok-seed, sağlam):**
- Uzunluk genellemesi: ctx 160'ta eğit → 1280'de (8×) çalış; sabit-gap recall uzunlukla artar (train-short / infer-long).
- Bellek girişim-sınırlı, decay-sınırlı değil.
- DPFP kapasite ekseni (Schlag ve ark. 2021): girişim-yoğun uzun gap'te baseline'ın 2-6×'i, öğrenmeyi de stabilize ediyor.
- Delta yazım: anahtar-güncelleme görevinde çok-seed 2× (ort. %16→%33).
- `cubic_flux` uzun-ufuk hipotezi DOĞRULANDI (cubic+dpfp 63.9% vs exp+dpfp 20.7%, >4 SE; Ek 16) — artık parked değil.
- LM doğrulaması: WikiText-2 ablasyonu (3 seed) en iyi reçete `cubic_flux + additive + dpfp` PPL 183.6 (Ek 17); yazım kuralı K2 ile **additive'e kilitlendi** (Ek 20).
- GLA aile-baseline'ı — K1 REVİZYONDA: metrik artefaktı (Ek 21) hükmü geri çektirdi; düzeltilmiş tek-seed prob yönü koruyor (55.4 vs 226.7) ama pencereli O(1) koşusu bekleniyor.

**Açık / kanıtlanmamış:**
- Mamba kıyası yok (GLA kıyası tamam, Ek 19).
- Bazı pozitifler tek-seed (dpfp×1280 s2, streaming-mix s0).
- İki-kademeli (two-tier) bellek: prototip doğrulandı ama adil test edilmedi (regime öğrenilemiyordu).

---

## 1. Ortam kurulumu (GPU makinesinde, bir kez)

```bash
# CUDA'li torch (GPU makinesinin CUDA surumune gore; asagidaki cu121 ornek):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40" numpy
# Checkpoint klasoru (scriptler HFP_CKPT_DIR'i okur; default 'checkpoints'):
export HFP_CKPT_DIR=checkpoints
export PYTHONPATH="$PWD:$PYTHONPATH"
python smoke_test.py                      # once bu GECMELI
python review_scripts/verify_claims.py    # chunk-tutarlilik + causal + kubik ODE kesinligi
```
Tüm deney scriptleri GPU'yu otomatik kullanır (`cuda` varsa). GPU'da zaman
bütçesini büyüt: `export BUDGET=600` (tek çağrı başına saniye).

---

## 2. Faz 1 — çalışmaya hazır deneyler (tek komut)

Bunların hepsi zaten scriptli; sürücü checkpoint döngüsüyle sonuna dek koşar:

```bash
BUDGET=600 bash review_scripts/run_remaining_experiments.sh
```

Sürücünün kapsadığı hücreler, amaç ve başarı kriterleri:

### 1.1 Harici GLA baseline — `review_scripts/baseline_compare.py` (3 seed)
- **Amaç:** İncelemenin son açık şartı — HFP'yi verimli-özyineli aileye (GLA) karşı konumlamak. Aynı dense görevi, aynı kovalar.
- **Başarı ölçütü (iddia için):** HFP (exp+dpfp) ile GLA aynı parametrede; HFP'nin uzun-gap kovalarında ≥ GLA olması, "HFP en azından aile-standardı kadar iyi" der. Amaç kazanmak değil, **konumlamak**.
- **Beklenti:** GLA yakın çıkar; HFP'nin pencereli-attention'ı kısa gap'te avantaj, uzun gap'te dpfp belirleyici.

### 1.2 Saf-bellek ablasyonu — `review_scripts/pure_memory_ablation.py` (3 seed)
- **Amaç:** Ring buffer'ı kapatıp (`max_short_len=1`) "bilgi gerçekten M/z'den akıyor" iddiasını temizlemek (inceleme §6).
- **Başarı ölçütü:** gap≥16 kovalarında `pure` ≈ `full` → recall gerçekten M/z'den. `pure` çökerse iddia düzeltilmeli.

### 1.3 DPFP × uzunluk — eksik seed 0,1 (`length_gen.py`, LG_VARIANT=dpfp)
- **Amaç:** Ek 9'daki tek-seed (s2) dpfp@1280 bileşimini çok-seed'e çıkarmak.
- **Başarı ölçütü:** s0/s1 de 128-255 ve 256+ kovalarında elu'yu belirgin geçerse bulgu sağlamlaşır.

### 1.4 Streaming-mix — eksik seed 1,2 (`streaming_mix.py`, 4 kol)
- **Amaç:** Ek 12'nin tek-seed (s0) sonucunu çok-seed'e çıkarmak; delta+dpfp'nin güncelleme-ağır akıştaki yerini görmek.
- **Başarı ölçütü:** delta'nın güncelleme kovasındaki avantajı ≥2 seed'de tutarlıysa.

---

## 3. Faz 1b — cubic_flux ADİL testi (yeni; önceden-kayıtlı)

> Bu, projenin asıl özgün iddiasının karar deneyi. cubic+additive zaten kaybetti;
> test edilecek doğru hücreler cubic+dpfp ve two-tier-cubic-slow, seyrek + uzun-gap
> rejiminde, **mode-başına LR taramasıyla** (yoksa incelemedeki gibi LR artefaktı çıkar).

**Tasarım (2×2 + two-tier):**
- Eksenler: retention ∈ {exp, cubic_flux_chunked} × feature-map ∈ {elu, dpfp} × bellek ∈ {tek-kademe, two-tier(cubic-slow)}.
- Rejim: seyrek (P=8), ctx≥640, eğitim ctx 160 (öğrenilebilir zemin) → eval uzun.
- Kovalar: özellikle **128-255 ve 256+** (platonun teorik avantaj bölgesi).
- LR: her mode için {3e-4, 1e-3, 3e-3} tara; en iyi LR'yi raporla.
- Seed: ≥3.

**ÖNCEDEN yazılan başarı kriteri (post-hoc rasyonalizasyonu önler):**
> "cubic_flux+dpfp, 256+ kovasında exp+dpfp'yi 3 seed ortalamasında **>2 standart hata** geçerse, cubic'in uzun-ufuk avantajı DOĞRULANDI sayılır. Aksi halde hipotez reddedilir / parked kalır."

**Yapılacak:** `review_scripts/cubic_longhorizon.py` (henüz yok — yazılacak).
`dense_retention.py` + `length_gen.py` iskeletini kullanır; `decay_mode=cubic_flux_chunked` ve `two_tier.py` konsolidasyonunu çağırır. (İstersen bunu ben yazayım.)

**SONUÇ (Tamamlandı):** `cubic_flux_chunked` + `dpfp`, `exp` + `dpfp`'yi 256+ gap kovasında >4 SE (63.9% vs 20.7%) farkla geçerek uzun-ufuk avantajını DOĞRULADI.

---

## 4. Faz 2 — ölçek (GPU'nun asıl işi)

### 2.1 Gerçek LM eğitimi (ilk dil-modeli sonucu) [TAMAMLANDI]
- Kazanan reçeteyle (exp+additive+dpfp+standard FFN) `tinyshakespeare` veya daha büyük veri.
- **Eşit-parametreli** GLA/Mamba baseline'a karşı perplexity.
- Çıktı: ilk "HFP dilde rekabetçi mi" verisi (şu an hiç yok).
- **SONUÇ (Tamamlandı):** HFP (cubic+delta+dpfp) TinyShakespeare'de GPT-2 baseline'ını (PPL 257 vs 300) geçti. Hangi bileşenin en çok katkı sağladığını bulmak için LM ablasyon deneyi devam ediyor.

### 2.2 cubic_flux uzun-ufuk (chunked)
- `cubic_flux_chunked` (tam paralel form, doğrulanmış) + `two_tier.py`.
- Faz 1b kazanırsa, ölçekte tekrarla.

---

## 5. Faz 3 — pretrained LLM'e grafting/distilasyon (yüksek kaldıraç, ayrı yön)

> "HFP belleğini var olan bir LLM'e entegre et" fikri. Sıfırdan eğitmekten ucuz
> ve daha güçlü sonuç verebilir (dil yeteneği hazır gelir). Literatür var:
> T2R, SUPRA, Mamba-in-Llama, LoLCATs.

- Küçük pretrained model: GPT-2 small veya Llama-3.2-1B.
- `HFPBulkState` bellek modülünü, HF attention alt-modülünün yerine drop-in sar (`decay_mode="exp"`, `key_feature_map="dpfp"`).
- **Tüm** attention'ı değil, bir **alt kümeyi** değiştir; birkaç full-attention katmanı tut (hibrit) → recall korunur.
- W_q/K/V'yi pretrained'den başlat; MLP/embedding'i başta dondur.
- **Distilasyon:** hibrit ↔ orijinal attention-çıktısı eşleme + logit KL; kademeli çöz.
- Uzun-bağlam recall'da orijinalin KV-cache'ine karşı ölç.
- Not: recall için full-attention tutarsan uçtan uca O(1) değil, hibrit (recall ↔ bellek ödünleşmesi).

---

## 6. Açık sorular (open questions)

1. ~~**cubic_flux'ın uzun-ufuk avantajı gerçek mi?**~~ **(EVET)** Seyrek + uzun-gap + dpfp'de exp'i 3x recall avantajıyla geçti (Faz 1b).
2. **Two-tier (cubic-slow) öğrenilebilir rejimde uzun-gap'te iyileştiriyor mu?** Adil testi hiç yapılmadı.
3. **exp'in çok-ölçekli λ'sı cubic'in avantajını ne kadar yiyor?** exp zaten multi-timescale; cubic'e ne kalıyor?
4. **DPFP ölçekte nereye kadar?** key_dim 4× büyük modelde de girişim sınırını aşıyor mu, maliyet/fayda?
5. ~~**Delta chunkwise ölçekte:** dense LM'de additive'i geçiyor mu?~~ **(HAYIR — K2, Ek 20)** eval@2048'de delta sayısal olarak daha kötü (additive 1.8 SE önde); reçete additive'e kilitlendi. Güncelleme-ağır *gerçek akış* (kod/diyalog) nişi hâlâ açık — delta orada yaşıyor.
6. **LM-benchmark:** GPT-2'yi geçti (aynı-hedefli kıyas, Ek 15). GLA hükmü Ek 21'deki metrik artefaktıyla ASKIDA — düzeltilmiş tek-seed 55.4 vs 226.7 yönü koruyor; resmi hüküm pencereli + eş-hedefli yeniden koşumda. Kalan: Mamba.
7. **Grafting:** HFP-belleği pretrained modele distille edilebilir mi? *(BAŞLADI: `hfp/models/grafting.py` + `colab_graft_qwen_v3_kaggle.ipynb`; Qwen2.5-1.5B, 13/28 katman, kafa-başına bellek, α-gate melez yazım, teacher-forcing distilasyon. Smoke 6/6 geçti; Stage 1 koşulacak. Plan K3.)*
8. **Streaming kararlılığı pratik fark yaratıyor mu?** cubic'in self-limiting'i (max|M| çok küçük) çok uzun akışta exp'e görünür avantaj mı?

---

## 7. Potansiyeller / kalibre beklentiler (kanıt değil, hipotez)

- **Genel dense/LM rejiminde cubic > exp:** düşük (~%15-20). exp'in multi-scale λ'sı güçlü, cubic'in platosu yoğun akışta z büyüyünce kayboluyor.
- **Seyrek + uzun-gap + dpfp nişinde cubic avantajı:** orta (~%40-50). DPFP kanalları seyrek tutar → plato korunur → exp'in üstel uçurumuna karşı polinomsal kuyruk ölçülebilir fark verebilir. Ama "herkesin zayıf olduğu bölgede daha az zayıf" tipi kazanç.
- **cubic streaming kararlılığı:** yüksek. Patlamıyor (verify: max|M|=14 vs exp 254 @4000 token).
- **DPFP ölçekte de kazanır:** orta-yüksek. Mekanizma ölçekten bağımsız (rank-collapse geciktirme).
- **Grafting/distilasyon çalışır (exp+dpfp):** orta. Literatür destekliyor; cubic ile değil, exp ile denenmeli.
- **HFP'nin LM'de GLA/Mamba'yı geçmesi:** ilk 'geçti' hükmü metrik artefaktıyla askıda (Ek 21); düzeltilmiş prob güçlü yön veriyor ama tam-attention config'iyle. Resmî güncelleme pencereli koşum sonrası.

---

## 8. Metodoloji kuralları (her deneyde uyulacak)

- **Mode-başına LR taraması** (cubic LR'a çok duyarlı; sabit LR kıyası haksız — inceleme bunu gösterdi).
- **≥3 seed**; tek-seed sonuç "tek-seed" etiketlenir, iddia edilmez.
- **Önceden yazılmış başarı kriteri** (post-hoc rasyonalizasyon yok).
- **Yoğun çok-sorgulu süpervizyon** (tek-sorgulu format optimizasyon artefaktı üretiyor).
- **Bellek iddiaları için saf-bellek ablasyonu** (ring buffer kapalı).
- **Herhangi bir dış iddiadan önce harici baseline** (GLA/Mamba).
- Negatif sonuç da sonuçtur; dürüst deftere (RESULTS.md §7) yazılır.

---

## 9. Önerilen sıra (kaldıraç × ucuzluk)

1. Ortam + `smoke_test` + `verify_claims` (dakikalar).
2. `run_remaining_experiments.sh` — Faz 1 (baseline + ablasyon + eksik multi-seed'ler). Ucuz, açık halkaları kapatır.
3. Faz 1b — cubic_flux adil testi (karar deneyi). Projenin asıl sorusu.
4. Faz 2.1 — ilk gerçek LM + eşit-param GLA baseline (ilk dil sonucu).
5. Faz 3 — grafting/distilasyon (en yüksek kaldıraçlı, ayrı yön).
6. Faz 2.2 — cubic uzun-ufuk ölçek (yalnız Faz 1b kazanırsa).

---

## 10. Deneyler bitince — güncelleme checklist

- [x] `DENEY_SONUCLARI.md` — Ek 16-18 + GLA/K1 (Ek 19) + K2 reçete kilidi (Ek 20) işlendi. Kalan: grafting ekleri (K3 bekliyor).
- [x] `RESULTS.md` — §12 (GLA/K1), §13 (K2 kilidi), §8 reçete güncellendi. Kalan: grafting bölümü (K3).
- [ ] `osf_companion.tex` → yeniden derle → `osf_companion.pdf` (v2.3); OSF'e yükle.
- [x] GitHub README reçete + GLA konumlandırması hizalandı. Kalan: HF model card (K3 sonrası).
- [ ] K1 yeniden koşumu: eş-hedef (FIX M1) + pencereli O(1) config + 3 seed — Ek 21 sonrası zorunlu.
- [x] Karar: cubic_flux kazandı mı? → **DOĞRULANDI** (uzun-ufuk karar deneyi: cubic+dpfp 63.9% vs exp+dpfp 20.7%, >4 SE, her iki kolda LR taramalı; RESULTS §6 / DENEY_SONUCLARI Ek 16. WikiText-2'de de en iyi reçetenin parçası: cubic+additive+dpfp PPL 183.6, RESULTS §10).

---

## Ek: GPU temini ve krediler (doğrulanacak — güncel fiyat/politika değişir)

- **Ucuz on-demand:** Vast.ai, RunPod, Lambda — saatlik, küçük deneyler için ideal.
- **Bedava küçük ölçek:** Google Colab / Kaggle (Faz 1 hücreleri buraya sığar; ≤1M param, CPU/T4 yeter).
- **Akademik/araştırma kredileri:** Google Cloud research credits, AWS Cloud Credits for Research, TPU Research Cloud (TRC), HuggingFace community GPU grants. Başvuru için kısa bir proje özeti + repo linki genelde yeterli — HFP'nin dürüst README'si + companion tam bu iş için elverişli.
- Faz 1 + 1b için tüketici GPU'su (tek 3090/4090 sınıfı, saatlik ~birkaç $) fazlasıyla yeter; Faz 2/3 için biraz daha büyük. Güncel fiyatları koşmadan önce kontrol et.
