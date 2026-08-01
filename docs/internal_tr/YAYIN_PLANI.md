# Yayın / Değerlendirme Planı — CEPTE (2026-07-28)

**Durum: ERTELENDİ, iptal değil.** Mimari yola gidiliyor; bu plan hazır bekliyor.
Elindeki çalışma bugün itibarıyla yayınlanabilir durumda — ne zaman istersen
buradan devam edilir. Hiçbir yeni deney gerektirmiyor.

---

## Neyin sana ait olduğu (yayında iddia edilecek katkılar)

1. **Graft reçetesi.** 6 katman [3,7,11,15,19,23], iki aşamalı öğretmensiz
   distilasyon, S2 cross-chunk müfredat, `out_gain=0.1` init, DPFP, exp decay,
   (opsiyonel) eğitilebilir memory projeksiyonları. On koşuda hata ayıklanarak
   var edildi; tekrarlanabilir, belgeli, bedava T4'te koşuyor.
   İki Pareto noktası: **150k param → 1.111× PPL (3 seed)** | **19M param → 1.043×**.

2. **Yoğunluk duvarının haritası ve mekanizması.** 6→13 katman 1.6-1.8× PPL;
   dört bağımsız konfigürasyon aynı bantta (naif seçim, ilkeli seçim,
   stabilizasyon, student-forward). S1 per-katman MSE ~0.089 iyi ama uçtan-uca
   kötü → sınır **birikimli hata (compounding)**, per-katman kapasite değil.

3. **Retention yasasının rejime bağlılığı.** Yoğun-yazımda fark yok (§15h);
   seyrek-yazımda ön-kayıtlı, eşleşmiş, n=16: **+0.48 nat, 13/16, p=0.012** (§28a).
   η ayarı iki yönde de iyileştirmiyor, büyük η kararsız (§27).

4. **KV-cache confound uyarısı (en yüksek etki potansiyeli).** Hibrit
   attention/O(1) modellerde görünen uzun-menzil geri getirme tamamen artık
   KV-cache tarafından taşınabilir; alanda genelde cache açıkken ölçülüyor.
   Teşhis protokolü: chunk başına cache sıfırla + üst-sınır (tam attention) kolu.
   Kendi sonucumuzda yakalandı, erratum'la düzeltildi (§30 + RESULTS başı).

5. **Metodoloji.** Ön-kayıt, güç kontrolleri, üst-sınır kolları, checkpoint
   parmak izi, yayınlanmış negatifler (kayıtların ~2/3'ü negatif).

**Sana ait OLMAYAN (açıkça atfedilecek):** graft paradigması (Mamba-in-Llama,
LoLCATs, MOHAWK, LAWCAT), lineer attention, DPFP (Schlag), delta kuralı
(DeltaNet), gated decay (GLA).

---

## Kanallar (etki sırasına göre)

**1. Confound yazısı — tek başına, kısa.** Blog/HN/X. Başlık fikri:
*"Your hybrid model's long-range recall might just be the KV cache."*
Karşı-sezgisel, faydalı, ve tek bir grafikle anlatılır (koşul A/B/C tablosu).
En yüksek dolaşım potansiyeli olan parça. **Bunu ilk yap.**

**2. arXiv preprint.** Dürüst ampirik çalışma; yukarıdaki 1-4 katkı.
Başlık taslağı: *"Grafting O(1) memory into a pretrained LLM: a density wall,
a regime-dependent retention law, and a KV-cache confound."*
Workshop seviyesinde gerçekten yayınlanabilir. `docs/PROJECT_SUMMARY.md`
zaten özet çekirdeği.

**3. HuggingFace model + model card.** Checkpoint yayını (Kapı-2 lisans
kontrolünden sonra — `LISANS_KARAR_REHBERI.md`). Qwen atfı zorunlu.

**4. Notebook'un kendisi.** "Bedava T4'te graft reçetesi" olarak açık kaynak;
tekrarlanabilirlik alanda nadir, tek başına değerli.

**5. Portföy.** İş/kontrat/işbirliği görüşmelerinde repo + RESULTS + erratum.

---

## Yapmadan önce gereken tek şey

**1.043× sonucu tek seed.** Manşet yapılacaksa 2-3 seed'e çıkarılmalı
(§34 config'i, RUN_SEED=1,2). Her biri ~2-3 saat. Bu, yayın öncesi tek borç.

---

## Para konusunda dürüst not

Bu kanalların hiçbiri doğrudan ödemiyor. Zincir: **itibar → fırsat**
(iş, kontrat, işbirliği, grant, ileride fon). Doğrudan gelir freelance/kontrat
tarafından gelir; bu çalışma o kapıyı belirgin biçimde kolaylaştırır.
