# HFP — Bağımsız Kod İnceleme Raporu

Değerlendirici: bağımsız ML/mimari inceleme (2026-07-04). Yöntem: tüm çekirdek dosyalar satır satır okundu; torch 2.12.1 (CPU) kuruldu; `smoke_test.py` + 15 maddelik bağımsız doğrulama süiti + CPU'da davranışsal mini-deneyler (MQAR ablasyonu, retention A/B, iki LR'de) koşuldu. Koşu scriptleri: `review_scripts/verify_claims.py`, `review_scripts/mini_ckpt.py`.

**Özet hüküm:** Mühendislik katmanı sağlam — chunk-tutarlılık, causal doğruluk, O(1) state ve kübik ODE ayrıklaştırması hepsi torch'la birebir doğrulandı. Ancak iki *davranışsal* iddia bu incelemede tekrarlanamadı: (1) binding conv ablasyonu recall'ı çökertmiyor; (2) headline "cubic > exp" bulgusu LR'a duyarlı — eşit koşullu mini-deneyde tersine döndü. Ayrıntı aşağıda.

---

## İddia iddia hükümler

### 1. O(1) çıkarım — DOĞRULANDI

`get_initial_state` (hfp_bulk_state.py:136-143) sabit şekiller döndürür: `M(B,key_dim,H)`, `z(B,key_dim)`, ring `(B,max_short_len,H)`, conv `(B,kk-1,H)`. Torch testi: 16 vs 96 token sonrası state şekilleri birebir aynı (verify_claims E maddesi, PASS). Eğitim aktivasyonlarının L ile büyüdüğü iddiası zaten yapılmıyor — tutarlı.

Küçük not: `max_position_embeddings` aşıldığında PE sessizce son pencereye kilitleniyor (modeling_hfp.py:31-32) — çökme yok ama uzun üretimde pozisyon bilgisi tekrarlanır; bunu belgeleyin.

### 2. cubic_flux doğruluğu — DOĞRULANDI

- **Ayrıklaştırma tam:** `λ_t = 1/√(1+2ηz²)` güncellemesi, `dθ/dτ = −ηθ³` ODE'sinin *birebir* zaman-1 akışıdır (θ⁻² değişkeninde ODE lineerleşir: d(θ⁻²)/dτ = 2η). 50 iterasyon vs analitik çözüm: fark < 1e-6 (verify_claims B). Bu "yaklaşık Euler" değil, tam integratör + impuls enjeksiyonlu splitting — matematiksel olarak savunulabilir.
- **Chunk-tutarlılık (torch):** 4 kombinasyonun hepsi PASS — exp/elu, exp/dpfp, cubic/elu, cubic/dpfp; full == state-taşıyan-chunked, fark ≤ 7e-7 (verify_claims A). smoke_test T4 yalnızca exp/elu'yu kapsıyordu; bu boşluk kapatıldı.
- **Kendini-sınırlama gerçek:** 4000 token agresif girdiyle (x·3) max|M| = 14.2 (exp aynı koşulda 253.9) — patlama yok (verify_claims C).
- **Dürüstlük notu:** λ_t, M'in kendi büyüklüğüne değil z'ye (anahtar akümülatörü) bağlı; "state-büyüklüğüne bağlı decay" ifadesini "z-büyüklüğüne bağlı" olarak netleştirin. Ayrıca cubic modda `decay` parametresi, exp modda `log_eta` ölü ağırlık — zararsız ama parametre sayımı raporlanırken belirtilmeli.
- **Maliyet ölçüldü:** ctx=48'de cubic ≈ exp'in 1.7-2.6× yavaşı (Python-döngülü sıralı recurrence); L büyüdükçe oran kötüleşir. Paralel-scan uygulanamazlığı doğru teşhis.

### 3. Binding conv zorunluluğu — ÇÜRÜTÜLDÜ (test edilen ölçekte)

MQAR, exp mod, 2 katman / hidden 64 / pencere 16 / 4 çift / ctx 48 / 800 adım / lr 1e-3 (şans %3.3):

| conv_kernel | recall acc |
|---|---|
| 3 (açık) | %28.7 |
| 1 (kapalı, ablasyon) | %30.0 |

Ablasyon çökmedi; hatta farksız. Mekanik açıklama: ≥2 katmanlı modelde 1. katmanın lokal attention'ı (pencere ≥ 2) k'yi v-pozisyonuna zaten karıştırır; 2. katmanın belleği bu karışmış temsili yazar — conv'un yaptığı işi attention yapıyor. K8 gerekçesindeki "k1'i kodlamadığından v1 GETİRİLEMİYORDU" analizi tek-katmanlı bellek yolu için doğru ama çok-katmanlı mimaride *zorunluluk* değil. (MQAR'ın K7 öncesi hiç öğrenilememesinin kök nedeni büyük olasılıkla yalnızca PE boğması idi; K8 katkısı bu deneyde ayırt edilemedi.) Conv'u tutmak makul (Mamba/H3 standardı, ucuz) ama "olmazsa olmaz" iddiasını geri çekin ya da tek-katman/pencere-1 koşulunda gösterin.

### 4. HEADLINE — cubic > exp uzun-menzil recall — ÇÜRÜTÜLDÜ (mevcut kanıtla savunulamaz)

İki ayrı sorun var.

**(a) Deney tasarımı:** "gap 32/64 saf bellek testi" iddiası kısmen yanlış. 4 katman × (pencere−1)=15 → yığılmış lokal attention'ın alıcı alanı ~60 token. Gap 32'deki (k,v), sorguya attention-yoluyla ulaşabilir; yalnızca gap ≥ ~61 gerçekten saf bellektir. Sizin kurulumda yalnız gap-64 satırı bellek kanıtıdır.

**(b) Tekrarlanabilirlik:** Eşit koşullu mini-A/B (2 katman, pencere 8, alıcı alan ~14, gap 16/24 saf bellek, 800 adım, tek seed, şans %3.3):

| | gap2 | gap4 | gap8 | gap16* | gap24* |
|---|---|---|---|---|---|
| exp, lr 1e-3 | 100 | 100 | 100 | 100 | 100 |
| cubic, lr 1e-3 | 100 | 100 | 100 | 100 | 100 |
| exp, lr 3e-4 | 62 | 54 | 58 | 62 | 47 |
| cubic, lr 3e-4 | 50 | 43 | 31 | 13 | 12 |

(*saf bellek bölgesi.) lr 1e-3'te ikisi de mükemmel; repo'nun default'u olan lr 3e-4'te ise **exp cubic'i geçiyor** ve cubic'in eğrisi gap ile düşüyor — sizin bulgunuzun tam tersi. Her iki mod da lr 3e-4'te ilk ~300-500 adım loss'u ln(val_space) platosunda geziyor (sizin exp'te gördüğünüz "4.6'da sabit" imzası) — bu plato bir *optimizasyon* fenomeni ve iki modu da vurabiliyor. Sizin GPU koşunuzdaki exp başarısızlığının kapasite değil LR/adım artefaktı olma olasılığı yüksek. Beklemedeki exp@4000 kontrolü bu yüzden kritik; ona ek olarak **LR taraması (en az 3e-4/1e-3/3e-3 × iki mod) ve ≥3 seed olmadan headline iddia yayınlanamaz.** Mevcut haliyle: mekanizma gerçek, avantaj kanıtlanmamış.

### 5. DPFP kapasite — KISMEN (şekil doğruluğu kanıtlı, kapasite iddiası bağımsız test edilmedi)

`_feat` (hfp_bulk_state.py:128-134) Schlag ve ark. DPFP'sine uygun; key_dim=2H·nu ile tüm şekiller ve chunk-tutarlılık torch'la PASS (exp/dpfp, cubic/dpfp). Causal sızıntı yok. Ancak "elu N=16'da %13'e çöker, DPFP %82 tutar" numpy sonucu bu incelemede tekrar üretilmedi (GPU ölçeği gerektirir) — sizin `--pairs 8/16/32` koşularınız hâlâ gerekli. Bir dikkat: DPFP özellikleri tam 0 olabilir (elu+1'in aksine kesin pozitif değil) → payda 1e-6'ya düşebilir; retrieval LayerNorm bunu emiyor ve testlerde NaN görülmedi, ama fp16'ya geçerseniz burayı izleyin.

### 6. Causal doğruluk / sızıntı yok — DOĞRULANDI (bir ölçüm-semantiği uyarısıyla)

- Tam model, 4 mod kombinasyonu: son token değiştirildiğinde önceki logitler bit-düzeyinde aynı (fark 0.0); orta token değiştirildiğinde öncesi aynı (verify_claims D). smoke T5 + benim testlerim: cache'li üretim yolu tutarlı.
- exp chunkwise cebiri bağımsız naive recurrence ile birebir örtüşüyor (4.8e-7; verify_claims F) — cross/intra/decay üsleri doğru.
- Ring buffer attention'ı yalnızca *önceki* chunk'ların buffer'ını görüyor (`past_state[0]`, decoder:163-176) — gelecek sızıntısı yok.
- **Uyarı (bug değil, ölçüm semantiği):** chunked çıkarımda attention, pencereye EK olarak ring buffer'daki son `max_short_len` (default 32) tokeni görür. Yani "chunk sınırını bilgi yalnızca M/z ile geçer" ifadesi yanlış; ring buffer da taşıyıcı. `predict_chunked`'ın dokümanı bunu söylüyor ama `run_recall` çıktısındaki ">>> bilgiyi taşıyan şey gerçekten M/z" yorumu fazla iddialı. Saf M/z ölçümü için ablasyonda ring buffer'ı da sıfırlayın ya da `max_short_len=short_len=1` ile koşun.

### 7. LM rekabetçiliği — KISMEN (veri ön-bulgu, kod adil)

Kodda exp'i sabote eden bir şey yok: iki mod aynı parametre kümesini, aynı gate/conv/feature-map yolunu paylaşıyor; tek fark retention yasası. exp'e multi-scale init (0.90..0.999) verilmiş — adil, hatta cömert. 305 vs 295 ppl tek seed'de gürültü aralığında; çok-seed olmadan sonuç çıkarılamaz. (CPU'da bu ölçek yeniden koşulamadı.) Cubic'in 2-3× yavaşlığı LM eğitiminde gerçek maliyet.

---

## Mekanizma mı, dekoratif yeniden-isimlendirme mi?

- **Gerçek mekanizma:** cubic_flux (state-bağımlı decay — ailede gerçekten yok, doğru implement edilmiş), DPFP (literatürden, doğru), binding conv (gerçek ama standart ve burada gereksiz çıktı), pencereli attention + recurrent bellek ayrımı (K5, ölçüm için doğru tasarım).
- **Dekoratif:** "Entangled/Bulk/holographic" katmanı. `EntangledFFN` pratikte **rank-kısıtlı paylaşımlı-faktörlü FFN**: `W_A = P_A·W_bulk`, `W_B = P_B·W_bulk`, rank ≤ bulk_dim. Deney config'lerinde bulk_dim=32, FFN genişliği 512 — **rank-32 FFN darboğazı** iki modu da eşit vuruyor ama LM kalitesini sınırlıyor olabilir; standart FFN'le bir kontrol koşusu önerilir. `get_orthogonality_loss()` hiçbir yerde loss'a bağlı değil (grep ile doğrulandı: yalnız `return_aux=True` dalında, modeling onu hiç çağırmıyor) → P_A/P_B üzerinde hiçbir kısıt yok; ya bağlayın ya silin. hfp_config'teki fizik bayrakları default kapalı ve loss'a bağlı değil — ölü kod; dürüst etiketlenmiş ama repo'dan çıkarılması daha temiz.

Küçük bulgular: `_offset_from_state` yorumu 6-elemanlı tuple diyor, gerçek 7 (bayat yorum). `gate_dropout` logit'lere uygulanıyor — düşürülen kanal gate=0.5 alır (0 değil); niyet buysa belgeleyin. smoke_test cubic/dpfp yollarını hiç örtmüyordu; `review_scripts/verify_claims.py` bunu kapatıyor, CI'ya ekleyin.

---

## Nihai karar: verimli-uzun-bağlam ailesinde savunulabilir katkı var mı?

**Bugünkü kanıtla hayır; bir koşulla evet-adayı.** Kod tabanı, iddialarını test edecek altyapıya sahip ve mühendislik doğruluğu (bu incelemenin doğrulayabildiği her şey) sağlam — bu, bu tür projelerde nadir ve övgüyü hak ediyor. Ancak ayırt edici tez olan "kübik-plato retention'ı, geometrik decay'in yapamadığı uzun-menzil recall'ı sağlar" şu an tek seed'li, LR-taramasız, kısmen attention-erişilebilir gap'lerle ölçülmüş bir bulguya dayanıyor ve eşit koşullu tekrar denemesinde tersine döndü. Savunulabilir katkıya giden yol: (1) exp@4000 + iki mod × ≥3 LR × ≥3 seed taraması, (2) gap > katman×(pencere−1) garantili tasarım, (3) ring-buffer'sız saf-bellek ablasyonu, (4) en az bir GLA/Mamba-sınıfı baseline. Bunlardan sonra cubic avantajı hayatta kalırsa, "hibritlerde bellek bileşeni" konumlandırması gerçekçi ve yayınlanabilir olur.
