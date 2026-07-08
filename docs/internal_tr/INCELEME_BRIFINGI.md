# HFP — Kod İnceleme Brifingi (bağımsız değerlendirici için)

Sen kıdemli bir ML/mimari değerlendiricisin. Görevin: aşağıdaki iddiaları **koda
karşı kanıtlayarak** doğrulamak veya çürütmek. Övme; zayıf noktaları, tutarsızlıkları,
gizli hataları açıkça söyle. "Fizik ilhamı" retoriğini yok say — yalnızca kodda
gerçekten ne olduğuna ve ampirik olarak neyin kanıtlandığına bak. Her iddia için:
(a) kodda karşılığı doğru mu, (b) testi geçiyor mu, (c) iddia sonuçla örtüşüyor mu.

Repo kökü: `hfp_arch/`. Torch + transformers gerektirir. `python smoke_test.py`
her şeyden önce çalışmalı.

---

## 1. HFP nedir (kalibreli iddia — abartısız)

Sabit-bellekli ($O(1)$ çıkarım) causal dil modeli: pencereli lokal attention +
katman-başı recurrent lineer-attention belleği (`M ∈ ℝ^{key_dim×H}`, `z ∈ ℝ^{key_dim}`).
Uzun-menzil bilgi yalnızca bu bellekten akmak zorunda. Ayırt edici yanı **bellek
belleğin retention yasası** ve **kapasite ekseni** — ikisi de seçilebilir flag.

**Bu bir GPT rakibi DEĞİL.** Konum: verimli-uzun-bağlam ailesine (Mamba/GLA/RetNet)
aday; gerçekçi hedef hibritlerde bellek bileşeni olmak. Tüm iddialar ML iddiasıdır;
fizik makaleleri yalnızca ilham, kanıt değil.

## 2. Üç ayrı (dik) mekanizma — asıl inceleme hedefi

1. **Retention yasası (`decay_mode`)** — `hfp/core/hfp_bulk_state.py`
   - `"exp"` (baseline): geometrik decay `M_t=λ⊙M_{t-1}+k_tv_tᵀ`, `λ=σ(decay)`, paralel chunkwise.
   - `"cubic_flux"` (HFP): `dθ/dτ=−η·θ³` ODE'sinin kararlı ayrıklaştırması →
     **state-büyüklüğüne bağlı** decay `λ_t = 1/√(1+2η·z_t²)`. Boş kanal→plato (unutma yok),
     dolu kanal→aktif unutma. Sıralı O(L). `η=exp(log_eta)` öğrenilebilir, per-key-channel.
2. **Binding conv (`conv_kernel`)** — aynı dosya, `update()` içi.
   - Q/K yoluna depthwise causal conv (kernel 3); V orijinal x'ten (temiz). Amaç:
     v-pozisyonundaki anahtarın önceki k'yi kodlaması → associative recall mümkün olsun.
     `conv_kernel=1` = kapalı (ablasyon). Retention'dan bağımsız.
3. **Kapasite (`key_feature_map`)** — aynı dosya, `_feat()`.
   - `"elu"` (baseline): elu+1, `key_dim=H`. `"dpfp"`: Deterministic Parameter-Free
     Projection, `key_dim=2·H·nu` → efektif boyut büyür, rank-collapse gecikir.

Bunların dışında: `bulk_trigger_decoder.py` (pencereli attention + EntangledFFN),
`modeling_hfp.py` (embed×√d + PE×0.3 içerik/pozisyon dengesi), `configuration_hfp.py`
(tüm flag'ler), `run_experiment.py` (recall/lm/retention görevleri).

## 3. Bu oturumda yapılan düzeltmeler (kronoloji + gerekçe)

- **K2**: causal chunkwise lineer attention; tek-forward'da da bellek parametreleri gradyan alır.
- **K1**: MQAR label hizalaması (kq-pozisyonu logiti → vq); eski çifte-shift NaN'ı giderildi.
- **Multi-scale decay init**: λ kanal başına 0.90..0.999 (tek-ölçekli 0.9 ufku ~10 token idi).
- **K7 (kök neden)**: embed×√d + PE×0.3. Öncesinde PE normu (8.0) embedding normunu (0.23)
  ~35× boğuyordu → içerik-tabanlı recall imkânsızdı, MQAR loss `ln(val_space)`'te sabitti.
- **K8**: binding conv (yukarıda).
- **cubic_flux** ve **DPFP**: yukarıdaki iki yeni eksen (default kapalı; baseline=exp/elu).

## 4. Torch'suz doğrulanmış olanlar (numpy + `py_compile`)

Bu ortamda torch kurulamadı; matematik numpy ile bağımsız doğrulandı:
- K2 chunkwise == naive token-token recurrence: fark ~1e-15.
- cubic_flux: state sınırlı (patlama yok), full==state-taşıyan-chunked = **0.00** (chunk-tutarlı),
  retention eğrisi üstelden belirgin daha düz (plato imzası).
- binding conv: causal (son token önceki çıktıları etkilemiyor) + full==split = 0.0.
- DPFP kapasite: elu+1 N=16'da %13'e çöker, DPFP %82 tutar (rank-collapse hipotezi doğrulandı).
- key_dim≠H iken her iki recurrence şekil/aritmetik doğru (exp chunk==naive 6e-16, cubic stabil).
- 4 çekirdek dosya `py_compile` temiz.

**Reviewer: bunları torch'la tekrar doğrula** (özellikle `smoke_test.py` T2/T4, ve DPFP
açıkken şekil tutarlılığı). Numpy kanıtları `/tmp` scriptleriyle üretildi; sen torch'la yap.

## 5. Ampirik sonuçlar (kullanıcı GPU, ÖN-BULGU — tek seed)

- `smoke_test.py`: tüm testler geçti (grad-akış, MQAR loss, chunk-tutarlılık, generate).
- LM (TinyShakespeare, GPT2-BPE, seq128, 1500 adım): val ppl exp≈305, cubic_flux≈295.
- **Retention A/B (asıl bulgu)**: aynı model, tek fark `decay_mode`. `local_window=16`,
  gap 32/64 saf bellek testi. cubic_flux %90 recall (gap 32:91, 64:90); exp şansta (%1),
  loss 4.6'da sabit. → Kübik-plato, exp'in beceremediği uzun-menzil recall'ı öğreniyor.
- **Beklemede (bu brifing yazılırken koşuyor)**: exp@4000 adım (adalet kontrolü — exp *hiç*
  öğreniyor mu), cubic seed 1&2 (tekrarlanabilirlik). Bunlar gelmeden "kanıt" denmemeli.

## 6. Doğrulanacak İDDİALAR (reviewer görev listesi)

1. **O(1) çıkarım**: state şekli bağlamdan bağımsız sabit mi? (`get_initial_state`,
   `use_cache` yolu). Eğitim aktivasyonu hâlâ dizi uzunluğuyla büyür — bunu iddia etmiyoruz.
2. **cubic_flux doğruluğu**: `λ_t=1/√(1+2η z²)` ODE'yi doğru ayrıklaştırıyor mu? Chunk-tutarlı mı?
   Kendini-sınırlama gerçek mi (state patlamıyor mu)?
3. **Binding zorunluluğu**: `conv_kernel=1` ile recall çöküyor mu? (ablasyon — çökmeli).
4. **HEADLINE: cubic > exp uzun-menzil recall'da**, aynı koşulda. exp@4000'de de şansta
   kalıyorsa iddia güçlü; öğreniyorsa iddia "cubic daha sample-verimli"ye iner. Kontrol et.
5. **DPFP kapasite**: `--pairs` artarken (8→16→32) dpfp, elu'dan zarif mi düşüyor?
6. **Causal doğruluk / sızıntı yok**: recurrence ve maskeler geleceği görmüyor mu? (kritik).
7. **LM rekabetçiliği**: cubic, exp'i sabote etmiyor mu? Çok-seed'de fark gürültü mü?

Komutlar:
```bash
python smoke_test.py
python run_experiment.py --task retention --steps 4000 --context 96 --max_gap 64 --local_window 16 --decay_mode exp
python run_experiment.py --task retention --steps 1500 --context 96 --max_gap 64 --local_window 16 --decay_mode cubic_flux --seed 1
python run_experiment.py --task recall --steps 1500 --context 128 --pairs 16 --local_window 16 --decay_mode cubic_flux --key_feature_map elu
python run_experiment.py --task recall --steps 1500 --context 128 --pairs 16 --local_window 16 --decay_mode cubic_flux --key_feature_map dpfp
```

## 7. Dürüst sınırlar (reviewer bunları veri kabul etsin)

- Ölçek <1M–7M param, sentetik/TinyShakespeare. Gerçek LLM veya ölçekte Mamba/GLA ile
  karşılaştırma **yok**. Küçük-ölçek galibiyeti ölçek için zayıf kanıttır.
- Headline tek seed; kontroller henüz tamamlanmadı.
- cubic_flux sıralı → yavaş (nonlineer recurrence → paralel-scan uygulanamaz). Throughput maliyeti.
- Fizik = ilham, kanıt değil. "5D" sihri yok; değer somut mekanizmalarda.
- EntangledLinear'ın `get_orthogonality_loss()`'u kodda var ama eğitime **bağlı değil** —
  paylaşılan projeksiyonlar çökmüş/artık olabilir. (Ayrıca incele.)

## 8. Reviewer'dan istenen çıktı

Her iddia için: DOĞRULANDI / KISMEN / ÇÜRÜTÜLDÜ + kanıt (kod satırı ya da koşu çıktısı).
Bulunması istenenler: gizli causal sızıntı, şekil hataları (özellikle DPFP açıkken),
cubic ODE ayrıklaştırma hatası, exp baseline'a haksız avantaj/dezavantaj, ve
"mekanizma mı yoksa dekoratif yeniden-isimlendirme mi" ayrımı. En sonda: bu mimarinin
verimli-uzun-bağlam ailesinde savunulabilir bir katkısı var mı — evet/hayır, gerekçeli.
