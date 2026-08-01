# Yol Haritası — 2026-07-28 (v2: mimari yol)

**Karar:** graft hattı hafıza için kapandı (§34a). Mimariyi değiştirme yoluna
gidiliyor. Yayın planı **cepte, ertelendi** — `YAYIN_PLANI.md`.

**Kapsam: ~4-6 hafta.** Sonrası ilk sonuca göre yazılacak.

---

## 0. Kritik içgörü — nereden başlamalı

Sezgisel cevap "sıfırdan büyük model eğit" olurdu. **Yanlış olur:** pahalı, yavaş,
ve öğreteceği şey belirsiz. Doğru başlangıç noktası zaten elimizde:

**Küçük-ölçek HFP modeli (kendi eğitilebilir projeksiyonlarıyla) TEK sınır
geçişinde %33-53 alıyor, iki sınırda şansa düşüyor** (§26b).

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

**Tasarım.** `carry_curriculum.py` + `matched_probe.py`, 2×2 (state küçük/büyük ×
additive/delta), 4-6 seed, K ∈ {0,1,2,4,8}. Birincil metrik: **K=2'de eşleşmiş
sonda doğruluğu** (şu an şansa düşen nokta). İkincil: K=0 (regresyon var mı?) ve
eğitim kaybı.

**Ön-kayıtlı kriter:**
- **ZİNCİRLEME AÇILDI:** bir kol K=2'de ≥%30 (şans %3.3) VE K=0'da regresyon yok
  → girişim hipotezi doğrulandı; kaldıraç belli, menzil genişletmeye geçilir.
- **KISMİ:** K=2'de %10-30 → yön doğru, daha agresif kapasite denenir.
- **ETKİ YOK:** her kol K=2'de <%10 → girişim de değil; sınır daha temel
  (okuma yolu / adresleme). O zaman mimari soru yeniden formüle edilir.

**Maliyet:** CPU'da koşar, GPU gerekmez. Kol başına ~10-20 dk.

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
