# HFP: Gelecek Araştırma Hipotezleri (Test Edilecek Fırsatlar)

> [!CAUTION]
> **Bilimsellik ve Dürüstlük Notu:**
> Bu belgedeki "Gelecek Araştırmalar" bölümündeki maddeler HFP modelinin halihazırda kanıtlanmış bir özelliği **değildir**. Bunlar; Kübik çürüme, Delta kuralı ve DPFP mimarilerinin matematiksel yapıları incelendiğinde ortaya atılmış **teorik hipotezler** ve profesyonel yorumlardır. Bu fikirlerin hepsi, ürünleştirme veya makale iddiası yapılmadan önce sıkı (rigorous) ampirik testlerden geçmek zorundadır.

Bu belge, HFP'nin dil modellemesi dışındaki potansiyellerini test etmek için bir **Ar-Ge Test Yol Haritası** olarak oluşturulmuştur.

---

## 1. Hâlihazırda Kodlanmış / Doğrulanmış Özellikler

Aşağıdaki özellik önceden hipotez aşamasındayken, şu an projedeki mevcut betikler (scripts) ile test edilmiş ve doğrulanmıştır:

### 1.1. Edge-AI (Uç Cihaz) Pratikliği ve Sabit VRAM - DOĞRULANDI
*   **Kanıt Durumu:** `eval_memory_scaling.py` betiği ile test edilmiştir (Bkz: `NASIL_CALISTIRILIR.md` §5).
*   **Mekanizma:** $O(1)$ boyutundaki matrisin bulut tabanlı bir KV-cache'e ihtiyaç duymaması, modeli kısıtlı donanımlarda VRAM şişmesi olmadan çalıştırabilir.
*   **Sonuç:** Bağlam uzunluğu (context length) artsa dahi VRAM kullanımının sabit (O(1)) kaldığı doğrulanmıştır. (Not: Doğruluk testleriyle birlikte sunulmalıdır).

---

## 2. Test Edilmesi Gereken Teorik Güçler (Gelecek Araştırmalar)

### 2.1. İçkin Önem-Tabanlı Unutma (Importance-Based Retention) Hipotezi
*   **Hipotez:** $d\theta/dt = -\eta \theta^3$ denklemi, büyük normlu bilgilerin yavaş ($\sim 1/\sqrt{t}$), küçük normlu bilgilerin hızlı unutulmasını dikte eder. Modelin, dış bir dikkat skoru olmadan bu matematiksel özelliği kullanarak kendi kendine önemli kelimeleri (isimler, kod blokları) kalın yazdığı teorize edilmektedir.
*   **Test Edilecek Yöntem:** Model önce eğitilmeli veya Distillation/Grafting yoluyla akıllı hale getirilmelidir. Ardından bellek matrisindeki ($M$) ağırlık normları ($L2$) görselleştirilecek ve seyrek ama kritik bilgilerin norm büyüklükleri ölçülecektir. Hipotez ancak "önemli tokenlar istatistiksel olarak daha büyük norma sahiptir" kanıtlanırsa doğrulanacaktır.

### 2.2. Çevrimiçi Sıkıştırma (Robust Compression) Hipotezi
*   **Hipotez:** Delta yazım kuralı sadece hatayı yazarken, DPFP kapasite eksenini ayırır. Bu, sabit $M$ matrisinin veriyi üst üste bindirmek (interference) yerine, "Çevrimiçi Temel Bileşen Analizi (Online PCA)" gibi anlamsal kümeler halinde sıkıştırdığı anlamına gelebilir.
*   **Test Edilecek Yöntem:** Sentetik bir veri akışında (farklı konuları içeren metin), $M$ matrisinin özdeğer (eigenvalue) spektrumu incelenecek ve kümelenme kalitesi ölçülecektir.

---

## 3. Test Edilmesi Gereken Uygulama Alanları (Gelecek Araştırmalar)

### 3.1. Sürekli Öğrenme (Continual Learning) Kapasitesi
*   **Hipotez:** HFP'nin Delta kuralı (eski izleri doğrudan silmeme) ve O(1) boyutu, ardışık görev öğrenirken yaşanan "yıkıcı unutmayı" (Catastrophic Forgetting) klasik Transformer'lara göre çok daha iyi engeller.
*   **Test Edilecek Yöntem:** Split-CIFAR veya "Task-incremental text classification" benchmark testleri kurulacak. Geçmiş görevlerdeki doğruluk oranının Mamba ve Llama'dan anlamlı derecede yüksek olduğu test edilmeden bu iddiada bulunulmayacaktır.

### 3.2. Pekiştirmeli Öğrenme (RL) Kredi Ataması Hipotezi
*   **Hipotez:** Satranç gibi uzun ufuklu, seyrek ödüllü ortamlarda, HFP'nin lineer yazımı ve kübik unutmaması "Kredi Ataması" (Credit Assignment) problemini çözer.
*   **Test Edilecek Yöntem:** HFP, bir Decision Transformer içine yerleştirilecek ve Mujoco/Atari ortamlarında Mamba/LSTM ile karşılaştırılacaktır.

---

## Stratejik Sonuç

Yukarıdaki gelecek araştırma hipotezlerinin hiçbiri, kendi izole edilmiş ve seed-kararlılığına (multi-seed) sahip deney kodlarıyla doğrulanmadan projenin "ana özellikleri" arasına eklenmeyecektir. 

Mevcut önceliğimiz, kanıtlanmış tek gerçek olan **"DPFP+Cubic mekanizmalarını Llama/Qwen tarzı bir modele entegre etme (Distillation/Grafting)"** aşamasıdır. Diğer teoriler, bu ana hedef tamamlandıktan sonra test edilecektir.
