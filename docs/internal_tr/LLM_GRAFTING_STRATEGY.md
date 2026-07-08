# HFP-to-LLM Grafting & Distilasyon Stratejisi (Faz 3)

Bu belge, O(1) karmaşıklığına sahip **HFP (Hyper-Flux Projection)** bellek modülünün, önceden eğitilmiş (pre-trained) modern bir açık kaynaklı Büyük Dil Modeline (örn. Llama-3.2-1B, Qwen-1.5B) entegre edilme sürecini (Grafting) detaylandırır.

**Ana Hedef:** Sıfırdan dil modeli eğitmenin getirdiği devasa donanım maliyetlerinden kaçınarak, konuşmayı ve dünyayı çoktan öğrenmiş zeki bir AI'ın beynine "sonsuz hafızamızı" cerrahi bir yöntemle yerleştirmek.

---

## 1. Mimari Entegrasyon (Grafting) Yaklaşımı

Tüm Attention katmanlarını söküp atmak yerine, dilin temel sentaksını koruması için **Hibrit (Karma)** bir mimari tercih edilmelidir. (Literatürdeki Mamba-in-Llama, LoLCATs ve T2R yaklaşımları bu yöntemin başarısını kanıtlamıştır).

### Tasarım Detayları:
*   **Katman Değişimi:** Dil modelinin katmanlarının **sadece belirli bir yüzdesi** (örn: %50 veya %75) HFP modülleriyle değiştirilir. Örneğin; 1, 2, 3. katmanlar HFP olurken 4. katman geleneksel Attention olarak bırakılır.
*   **Ağırlık Transferi (Warm-start):** Çıkarılan Attention modüllerindeki $W_q, W_k, W_v$ matris ağırlıkları çöpe atılmaz. Bu ağırlıklar doğrudan HFP modülündeki izdüşüm (projection) matrislerini başlatmak (initialize) için kopyalanır.
*   **RoPE (Rotary Position Embeddings) Çatışması:** Llama ve Qwen, RoPE kullanır. HFP ise kendi konumsal yapısına sahiptir. *Kritik Karar:* HFP katmanlarına giren token'lardan RoPE etkisini çıkarmalı mıyız (inverse RoPE), yoksa RoPE'li hallerini doğrudan HFP matrislerine besleyip modelin bunu öğrenmesini mi beklemeliyiz? (Öncelikli test: RoPE'yi HFP'den önce bypass etmek).

---

## 2. Öğretme ve Eğitme Süreci (Knowledge Distillation)

Grafting (Cerrahi) tamamlandığında model "afallayacaktır". HFP'yi nasıl kullanacağını öğretmek için Öğretmen-Öğrenci (Teacher-Student) distilasyonu kullanılacaktır.

### Adımlar:
1.  **Dondurma (Freezing):** Değiştirilmeyen Attention katmanları, FFN'ler (MLP) ve Embedding (Kelime) katmanları tamamen dondurulur. Sadece yeni eklenen HFP parametrelerinin gradyanları (requires_grad=True) açık bırakılır.
2.  **Öğretmen Model:** Llama-3.2'nin orijinal hali (Teacher) belleğe yüklenir.
3.  **Kayıp Fonksiyonları (Loss):**
    *   **Gizli Durum Eşleşmesi (Hidden State L2 Loss):** HFP katmanının çıktısı, Öğretmen modelin aynı sıradaki Attention katmanının çıktısına olabildiğince benzemeye zorlanır (MSE loss).
    *   **Logit KL Divergence:** En son kelime tahmin olasılıkları (Logits), Öğretmen modelin olasılık dağılımıyla eşleştirilir.
4.  **Veri Seti:** SlimPajama veya Fine-Web gibi yüksek kaliteli metinlerin küçük bir alt kümesinde (1-5 Milyar token) kısa süreli eğitim yapılır.

---

## 3. Olası Sorunlar ve Riskler (Dikkat Edilmesi Gerekenler)

> [!WARNING]
> **O(1) vs Full Attention Trade-off**
> Eğer modelde hiç Attention bırakmazsak (Tümü HFP olursa) O(1) hafızaya ulaşırız ama dilde geri dönüşü olmayan bir bozulma (degradation) yaşanabilir. Eğer çok fazla Attention bırakırsak, KV-Cache şişmeye devam edeceği için O(1) hafıza amacımızdan saparız. **Optimum hibrit oranını bulmak projenin en büyük riskidir.**

> [!CAUTION]
> **Dimension (Boyut) Uyumsuzluğu**
> HFP'nin `bulk_dim` ve `rec_block` hesaplamaları, LLM'in `hidden_size`'ı (örn Llama-1B için 2048) ile kusursuz eşleşmek zorundadır. Özellikle Llama'daki GQA (Grouped-Query Attention) mekanizmasının, HFP'nin DPFP kapasite eksenine nasıl haritalanacağı (mapping) önceden matematiksel olarak çözülmelidir.

> [!IMPORTANT]
> **Öğrenme Oranı (Learning Rate) Felaketi**
> Orijinal model zaten eğitilmiştir. Çok yüksek bir LR ile başlarsak modelin halihazırda bildiği dil yeteneğini "Catastrophic Forgetting" (yıkıcı unutma) ile tamamen silebiliriz. Çok düşük bir LR ise HFP modüllerinin yakınsamamasına (öğrenememesine) neden olur. **HFP parametreleri için yüksek (örn 1e-3), dondurulmamış diğer katmanlar için çok düşük (örn 1e-5) olmak üzere iki farklı LR grubu (parameter groups) kullanılmalıdır.**

---

## 4. Önemli Testler (Validation Pipeline)

Entegrasyonun başarılı olduğunu şu metriklerle ölçeceğiz:

1.  **Sıfırıncı Adım Testi (Zero-Shot Sanity Check):** 
    Grafting yapılıp eğitim başlamadan önceki saniyede modelin PPL'i (Perplexity) tamamen çöp olmamalıdır. (Örn: Llama PPL 12 ise, Grafting sonrası 1000'i geçiyorsa ağırlık transferini yanlış yapmışız demektir).
2.  **Kısa Bağlam Kayıp Testi (Short-Context Degradation):**
    Model HFP'yi öğrendikten sonra, klasik benchmark'larda (MMLU, HellaSwag) orijinal modele göre %5'ten fazla performans kaybetmemelidir.
3.  **İğne Deliği (Needle-In-A-Haystack) Testi:**
    Asıl şov alanı! Orijinal modelin KV-Cache'i 8K veya 16K'da tükenip/yavaşlarken, HFP-LLM'e **128K tokenlık** bir metin verip (örn. dev bir kitap veya kod deposu), içinden tek bir gerçeği (iğneyi) çekip alamadığına bakacağız. (DPFP ve Cubic burada parlamalıdır).
4.  **Hız ve VRAM Testi:**
    Akış esnasında (Streaming inference), 10.000. token üretilirken VRAM kullanımının 1. token ile tamamen aynı (sabit) kalıp kalmadığı kanıtlanmalıdır.

---

## 5. Sıradaki Eylem Planı (Checklist)

- [ ] Llama-3.2-1B veya Qwen-2.5-1.5B (GQA mekanizmasına uygun olan) seçilecek.
- [ ] Orijinal modelin katmanlarının içine girilip $W_q, W_k, W_v$ matris boyutları analiz edilecek.
- [ ] `hfp/models/grafting.py` isimli yeni bir dosya oluşturulup Teacher-Student distilasyon döngüsü kodlanacak.
- [ ] (Ablasyon notebook'u bittiğinde) `Cubic_flux + Delta + DPFP`'nin nihai kod versiyonu, entegrasyon için kilitlenecek.
