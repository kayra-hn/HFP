# Yol Haritası — 2026-07-28 (v2: mimari yol)

**Karar:** graft hattı hafıza için kapandı (§34a). Mimariyi değiştirme yoluna
gidiliyor. Yayın planı **cepte, ertelendi** — `YAYIN_PLANI.md`.

**Kapsam: ~4-6 hafta.** Sonrası ilk sonuca göre yazılacak.

---

## 0. Kritik içgörü — nereden başlamalı

Sezgisel cevap "sıfırdan büyük model eğit" olurdu. **Yanlış olur:** pahalı, yavaş,
ve öğreteceği şey belirsiz. Doğru başlangıç noktası zaten elimizde:

**Küçük-ölçek HFP modeli (kendi eğitilebilir projeksiyonlarıyla) TEK sınır
geçişinde ortalama %33.2 alıyor (seed aralığı %8.5–69.5), iki sınırda şansa
düşüyor** (§26b, `exp` kolu).

> **Düzeltme (2026-08-01):** bu satır önce "%33-53" diyordu, sanki tek bir
> konfigürasyonun aralığıymış gibi. Değil: **%33.2 `exp`, %53.4 `cubic`** kolunun
> ortalaması. §35 `decay_mode='exp'` sabitliyor → doğru taban ~%33.
> Ayrıca §26b'nin asıl dersi bu ortalama değil, **seed varyansı**: 6 seed'de
> %8.5–69.5. Bu yüzden §35 v3'te birincil metrik sonda doğruluğu değil,
> düşük-varyanslı **cross-chunk doğrulama kaybı** oldu (bkz. RESULTS §35).

Yani mekanizma **çalışıyor ama zincirlenmiyor**. Bu, graft'ın %0'ından çok daha
bilgilendirici bir başlangıç: sorun "hiç yok" değil, "bir adımdan sonra kayboluyor".
Ve o modeli eğitmek **CPU'da bile mümkün** — GPU kotasına bağımlı değilsin.

**Merkezi soru:** state neden bir sınırı geçiyor da ikincisini geçemiyor?

---

## 1. Adım — §35: Kapasite × yazım kuralı taraması (küçük ölçek, CPU)

**Hipotez (projenin kendi bulgusuna dayanıyor).** Bellek **girişim-sınırlı**
(README/§0, ve §27a'da η taramasıyla bağımsız doğrulandı: plato uzatmak
girişimi biriktirip *kötüleştirdi*). Dolgu chunk'ları her 64 tokende bir
distraktör kv yazıyor; iki chunk sonra hedef gömülüyor.

O halde zincirlemeyi açacak iki kaldıraç:
- **Daha büyük state** (`dpfp_nu`; `bulk_dim` belleği ETKİLEMİYOR, o FFN'de) →
  girişim seyrelir. **O(1)'i bozmaz**
  — sabit büyür, bağlamla büyümez.
- **Delta yazım** (`write_rule='delta'`) → eski değeri siler, üst üste biriktirmez.
  Kodda hazır; anahtar-güncelleme görevinde 2× kazandırmıştı.

**Tasarım (v3 — otorite ön-kayıt RESULTS §35).** `carry_curriculum.py` +
`matched_probe.py`, **2 kol × 4 seed**: taban (`nu=2`/additive) vs
(`nu=4`/delta), K ∈ {0,1,2,4}.
**Birincil metrik:** **eşleşmiş cross-chunk doğrulama kaybı** (nat, K=2, seed
başına 64 farklı örnek). **İkincil/betimleyici:** K=2 ve K=0 sonda doğruluğu.

**Ön-kayıtlı kriter (tarama, n=4 eşleşmiş):**
- **SİNYAL:** eşleşmiş ortalama Δ ≤ **−0.15 nat** VE 4/4 seed aynı yönde
  → tek izinli devam: ayrıştırma + güç (nu4/additive vs nu2/delta, 8-12 seed).
- **TERS:** Δ ≥ +0.15 nat VE 0/4 → girişim hipotezi bu yönde çürüdü, hat kapanır.
- **SONUÇSUZ:** diğer her şey → "etki yok" YAZILMAZ. Karar Kayrahan'ın:
  8-12 seed'e çıkmak ya da hattı park edip okuma/adresleme teşhisine geçmek.

**v2 neden iptal edildi:** 2 seed + mutlak %30 eşiği. §26b aynı aracı 6 seed /
200 denemeyle koştu ve K=0'da %8.5–69.5 yayılım ölçüp "yetersiz güç" dedi.
Daha az seed'le daha kesin hüküm vermek, projenin kendi **güç kontrolü** kuralını
ihlal ederdi. v3 bütçe-nötr: kol sayısı seed'e takas edildi (yine 8 koşu).

**Maliyet (DÜZELTİLDİ — ilk tahmin yanlıştı):** CPU'da taban kol 1200 adımda
~95 dk sürdü; ilk sürüm ~25 saat = Kaggle'ın 12 saatine sığmıyordu. v2'de kapsam
kısıldı (CTX 128, 600 adım, BS 4, CARRY_MAX 8) ve **GPU (T4)** kullanılıyor.
v3 koşu sayısını değiştirmedi. **Süre ölçülmeden tahmin edilmeyecek** — ilk kolun
100 adımını ölç, oradan çarp.

---

## 2. Adım — sonuca bağlı

**§35 açılırsa:** menzil genişletme (K=8,16,32) → hangi noktada doyuyor?
Ardından: o konfigürasyonu graft'a geri taşımak anlamlı mı, yoksa küçük model
kendi başına mı ilerlemeli? (Karar noktası.)

**§35 açılmazsa:** sınır okuma/adresleme yolunda demektir. O zaman sıradaki
inceleme: state'e yazılan şey geri okunabiliyor mu? (Doğrudan sonda: hedef kv'yi
state'e yaz, aynı chunk içinde oku — çalışıyor mu? Sınırdan sonra oku — çalışıyor
mu?) Bu, "yazma mı okuma mı bozuk" sorusunu kesin ayırır.

---

## 3. Yapmayacaklarımız (drift önleme)

- **Sıfırdan büyük model eğitmek** — pahalı, ve §35 cevabı gelmeden yön belirsiz
- Graft'a hafıza için geri dönmek (§34a kapattı)
- Yoğunluk itme, reçete yamalama (kapandı)
- 7B/K3 gibi büyük base'ler
- Ürün/pitch dilinde "hafıza organı" kullanmak

---

## 4. Cepte bekleyenler (iptal değil, ertelendi)

- **Yayın planı** — `YAYIN_PLANI.md`. Bugün yayınlanabilir durumda, yeni deney
  gerektirmiyor. Tek borç: 1.043× sonucunun 2-3 seed'e çıkarılması.
- **Cubic paralel formu** — rafta, koşulları BEKLEYEN #21'de
- **Demo notebook'u** — `hfp_demo_memory.ipynb`, çalışan hafıza bekliyor

---

## 5. Karar noktaları

| ne zaman | soru | kim |
|---|---|---|
| §35 sonrası | zincirleme açıldı mı? | veri |
| §35 açılırsa | küçük modelde mi devam, graft'a mı taşı? | Kayrahan |
| istediğin an | yayını cepten çıkar mı? | Kayrahan |

**Not:** bu belge §35'in sonucuyla yeniden yazılacak.
