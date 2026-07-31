# Yol Haritası — 2026-07-28 (yeniden çizim)

**Kapsam: önümüzdeki ~4-6 hafta.** Daha ilerisi kasıtlı olarak yazılmadı; §34'ün
sonucu haritanın devamını belirleyecek.

---

## 0. Nerede duruyoruz (dürüst özet)

**Doğrulanmış ve kalıcı:**
- 6-katman O(1) graft, **149.910** eğitilebilir param (~%0.01), **PPL 1.11×,
  3 seed** (§22a-c)
- 128k bağlamda **~%8 VRAM, ~%21 decode hızlanması**; graft state 9.5 MB **sabit**
  (§23)
- Yoğunluk duvarı haritalandı, sebebi **compounding** (§24-§25; dört config)
- Retention yasası **rejime bağlı**: yoğun-yazımda fark yok (§15h), seyrek-yazımda
  var (§28a, n=16, p=0.012)
- **Metodolojik katkı:** attention/O(1) hibritlerinde uzun-menzil geri getirme
  tamamen artık KV-cache tarafından taşınabilir ve genelde cache açıkken ölçülüyor
  (§30 + erratum)

**Kapanan hatlar (yeniden açılmayacak):**
- Yoğunluk itme: seçim (§24a), stabilizasyon (§24c), exposure-bias (§25a) — üçü de
  negatif. ~6 katman pratik tavan.
- Hafıza organı **reçete seviyesinde**: `no_grad` (§31), state detach (§31),
  seyrek denetim (§33) — üçü de kaldırıldı, depolama oluşmadı.
- Cubic: bulgu geçerli ama mühendislik nedeniyle rafta (#21).

**Açık kalan TEK teknik soru:** O(1) durumu, *doğru mimari serbestlik verilirse*,
sınır ötesi adreslenebilir depolama yapabilir mi?

---

## 1. Adım — §34: Hafıza yoluna kendi projeksiyonları

**Hipotez (veriye dayalı, spekülasyon değil).** Graft, Qwen'in **donmuş** q/k/v
projeksiyonlarını paylaşıyor. Bir olgunun hangi anahtarla saklanacağını donmuş
`k_proj` belirliyor — softmax attention için optimize edilmiş bir temsil, sıkıştırılmış
çağrışımsal belleğe *adreslenebilir kayıt yazmak* için değil.

**Destekleyen kanıt:** küçük-ölçek model (kendi eğitilebilir projeksiyonlarıyla)
tek sınır geçişinde **%33-53** alıyor (§26b, K=0). Graft **%0** (§30/§31/§33).
Aynı mekanizma, aynı görev ailesi; fark projeksiyonların eğitilebilirliği.

**Tasarım.** Graft'lı 6 katmana **kendi k/v (ve q) projeksiyonları** ver; base'in
projeksiyonları o katmanlarda artık ödünç alınmaz (residual/adapter olarak
başlatılabilir: kopya ile başla, sonra serbest bırak).
- Eğitilebilir param: ~2.4M × 6 kat × 2-3 proj ≈ **40-50M** (modelin ~%3'ü)
- Tek GPU'da eğitilebilir; base yine donmuş, bilgi korunuyor
- Eğitim: §31+§33 düzeltmeleriyle (yazma-BPTT + maskeli recall denetimi) —
  onlar kalıcı, çünkü gerçek hataları düzeltiyorlardı

**Ön-kayıtlı kriter** (hafıza sondası, koşul C, eşleşmiş, gap 0):
- **≥%60** → teşhis doğru, hafıza mümkün. Sonraki: menzil genişletme + demo.
- **%30-60** → küçük-ölçek seviyesine ulaşıldı; yön doğru, kapasite/state
  büyütmesiyle devam.
- **<%30** → donmuş projeksiyon hipotezi de düşer. O zaman graft yolu hafıza için
  **kapanır**; kalan tek seçenek sıfırdan hafıza-öncelikli model (pahalı, ayrı karar).

**Maliyet:** ~2-3 saat GPU (eğitim ağırlaştı), tek koşu.

---

## 2. Paralel hat — Compute (bağlayıcı kısıt)

Bedava katmanlar (Colab/Kaggle/Lightning) 40-50M param eğitimini zorlar. Bu hat
**şimdi** başlamalı, çünkü başvurular haftalar sürüyor:

- HuggingFace community GPU grant
- NVIDIA Inception
- Bulut araştırma kredileri (Google/AWS research credits)

**Başvuru materyali hazır:** repo + RESULTS + erratum. Dürüst negatifler
**avantaj** — metodolojik ciddiyet gösteriyor.

---

## 3. Adım — sonuca bağlı

**§34 ≥%30 ise:** menzil genişletme (uzun BPTT pencereleri) → hafıza demosu
(`hfp_demo_memory.ipynb` hazır, sadece çalışan model bekliyor) → demo elde varken
yatırım/grant konuşması anlamlı hale gelir.

**§34 <%30 ise:** graft-hafıza hattı kapanır. Elde kalan doğrulanmış çalışma
(verimli-attention reçetesi + duvar haritası + KV-cache confound uyarısı) yazıya
dökülür; hafıza tezi ancak sıfırdan model programıyla sürer ve o ayrı bir karardır.

---

## 4. Yapmayacaklarımız (drift önleme)

- Yoğunluk itmeye dönmek (kapandı)
- Reçete yamalamaya dönmek (§31/§33 ile tükendi)
- Cubic mühendisliği (rafta; koşulları #21'de)
- Delta yazım (§32) ve büyük-state kollarını **tek başına** koşmak — ancak §34
  sonrası ve onun bulgusuna bağlı anlamlı olur
- 7B/K3 gibi büyük base'lere geçmek (§30 çözülmeden anlamsız)
- Ürün/pitch dilinde "hafıza organı" kullanmak (§30 geri çekti)

---

## 5. Karar noktaları

| ne zaman | soru | kim |
|---|---|---|
| §34 sonrası | hafıza mümkün mü? | veri |
| §34 <%30 ise | sıfırdan model programı mı, yayın mı? | Kayrahan |
| grant çıkarsa | compute nereye? | Kayrahan |

**Not:** bu belge §34'ün sonucuyla yeniden yazılacak. Daha ilerisini şimdi
planlamak veri olmadan tahmin olur.
