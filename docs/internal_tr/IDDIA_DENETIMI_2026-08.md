# İddia denetimi — dışarıya bakan belgeler vs `RESULTS.md`

*2026-08-02, HEAD `2ced80f` (§36 sonucu dahil). Hazırlayan: GPU oturumu.*

**Hiçbir düzeltme uygulanmadı.** Bu belge bulgu listesidir; metin değişiklikleri
onaya bağlı.

## Neden şimdi

İki yeni sonuç dış belgelere hiç yansımadı:

- **§29b (2026-08-01):** §28a kendi metriğinde replike oldu (+0.4080 nat, 14/16,
  p=0.0005) **ama o metrik karışıktı.** Ayrıştırınca chunk-içi **+0.4298**
  (p=0.0008), cross-chunk **+0.2770** (eşleşmiş t p=0.1103, anlamlı değil).
  Yani §28a'yı bir *cross-chunk carry* kazancı diye sunmak sınır-ötesi kısmı
  olduğundan büyük gösteriyor.
- **§36 (2026-08-02):** LM ölçeğinde, cache confound'u kaldırılmış protokolle,
  cubic'in state'ini sınır ötesine taşımak tahmini **0.28 nat/token
  kötüleştiriyor** (Δ_cubic −0.2817 vs Δ_exp +0.0350, fark −0.3167,
  Wilcoxon p=2.7e−09). Ön-kayıtlı hüküm **TERS**; cubic'in LM-ölçek hattı kapandı.

Bunun sonucu: dışarıya bakan belgelerdeki **"cubic seyrek/uzun-ömürlü rejimde
kazanır, o rejim için önerilir"** çerçevesi artık ölçümle çelişiyor.

---

## Bulgu tablosu

`durum` ∈ DOĞRU / BAYAT / ABARTILI / DESTEKSİZ / ÇELİŞKİLİ

### `README.md`

| satır | iddia | RESULTS dayanağı | durum | öneri |
|---|---|---|---|---|
| 69-72 | "Sparse writes / long-lived memory (**cross-chunk carry**): cubic measurably better — +0.48 nat, 13/16, p=0.012" | §28a; **§29b ayrıştırması** | **ABARTILI** | "cross-chunk carry" etiketini kaldır. Etki bu görevde gerçek ama ayrıştırınca chunk-içi +0.43 (p=0.0008), cross-chunk +0.28 (p=0.11). Yeni metin: *"sparse-write synthetic carry task: +0.41 nat, 14/16, p=0.0005 (§29b replikasyonu). Ayrıştırma: kazancın çoğu chunk-içi retention'dan; sınır-ötesi bileşen daha küçük ve t-testinde anlamlı değil."* |
| 76-78 | "Practical rule: **`cubic_flux_chunked` for sparse-write / long-lived-memory work**" | §28b (raf), §36 | **ÇELİŞKİLİ** | **En yüksek öncelik.** §28b cubic'i mühendislik nedeniyle rafa kaldırmıştı; §36 LM ölçeğinde ölçülebilir **dezavantaj** buldu. Bir okuyucuya cubic'i "uzun-ömürlü hafıza işi için" önermek artık kendi kaydımızla çelişiyor. Yeni kural: *"`exp` her yerde varsayılan. `cubic_flux` küçük-ölçek sentetik seyrek-yazımda ölçülebilir bir avantaj gösterdi; LM ölçeğinde, cache confound'u kaldırılmış protokolle test edildi ve dezavantaj çıktı (§36)."* |
| 66-68 | "Dense/saturated (LLM-graft): no measurable difference (§15h)" | §15h; **§36 protokol notu** | **BAYAT** | Doğru ama eksik. §15h **cache açıkken** ölçtü; erratum bu protokolün state'e kör olduğunu gösterdi. Not ekle: *"§15h cache açıkken ölçüldü; cache sıfırlanınca aynı ikizler farklı davranıyor (§36)."* |
| 117-119 | "**Official recipe (locked)**: `cubic_flux_chunked` decay + additive + dpfp" | §10; §28b raf; §36 | **ÇELİŞKİLİ** | "locked" ifadesi §28b'nin raf kararından ve §36'dan önce yazılmış. Ya varsayılanı `exp` yap (BEKLEYEN #3'ün öngördüğü karar), ya da "WikiText-2 ablasyonunda (§10) seçilen reçete; sevk varsayılanı `exp` (§15h, §28b, §36)" de. |
| 120 | "`cubic_flux` long-horizon win: **3x recall advantage** — confirmed at n=16, p=0.012 (§28a)" | §6 **ve** §28a | **ÇELİŞKİLİ (birleştirme)** | İki ayrı deney tek cümleye sıkıştırılmış. "3× recall" §6'nın sonucu (%63.9 vs %20.7); "n=16, p=0.012" §28a'nın ve endpoint'i **kayıp**, recall değil. Ayır. Ayrıca §36 kaydını ekle. |
| 128-132 | "İki Pareto noktası ... ~19M → **1.043× PPL**" | §34a | **BAYAT** | 1.043× **tek seed**. README'de belirtilmemiş; RESULTS belirtiyor. "(single seed)" ekle — bu, yayın öncesi tek borç olarak zaten kayıtlı. |
| 121-127 | §30 uyarısı (cache confound, %0, dört müdahale) | §30, §31, §33, §34a | **DOĞRU** | Değişiklik gerekmiyor. İyi yazılmış; §36'ya bir cümlelik atıf eklenebilir. |
| 110-116 | Length generalization, interference-limited, DPFP | §3, §4, §5 | **DOĞRU** | — |

### `docs/PROJECT_SUMMARY.md`

| satır | iddia | RESULTS dayanağı | durum | öneri |
|---|---|---|---|---|
| 45-52 | "**3. Retention law matters only in sparse-write regimes.** ... pre-registered, paired, 16-seed win in a **sparse-write carry task** (+0.48 nat, 13/16, p=0.012) ... so the effect is real, modest, and **regime-specific**" | §28a; §29b; §36 | **ABARTILI + ÇELİŞKİLİ** | İki düzeltme: (a) "carry" etiketi (§29b ayrıştırması), (b) "regime-specific" artık yetersiz — §36 LM ölçeğinde **ters yönde** anlamlı sonuç verdi. Bu, tek sayfalık dış özet; §36'nın **kendi maddesi olmalı**, dipnot değil. Öneri: 3. maddeyi "retention law: küçük ölçekte ölçülebilir, LM ölçeğinde ters" olarak yeniden yaz ve §36 sayılarını ver. |
| 67-82 | "Honest status" — graft hafıza için kapandı | §30-§34a | **DOĞRU** | — |
| 28-35 | İki Pareto noktası, VRAM/latency | §23, §34a | **DOĞRU** | 1.043×'in tek seed olduğu burada yazılı ("single seed" ibaresi yok ama "meeting the bar for the first time" var) — netleştirilebilir. |

### `docs/paper3_ml_architecture.tex` — **en ağır sorun burada**

Bu taslak **erratum öncesi** yazılmış ve §30'un bulgusundan hiç haberi yok.
Merkezi tezi cubic ve cubic LM ölçeğinde ters sonuç verdi.

| satır | iddia | durum | not |
|---|---|---|---|
| 14 | Başlık: "O(1)-Memory Language Modeling **via Cubic-Plateau Retention**" | **ÇELİŞKİLİ** | Makalenin ana tezi, §36'da ölçülebilir dezavantaj çıkan mekanizma. Yeniden konumlandırma gerekiyor (BEKLEYEN #13 zaten bunu öngörmüştü). |
| 23 | Özet: "enables **robust long-horizon retention without catastrophic interference**" | **DESTEKSİZ** | §30: cache sıfırlanınca geri getirme %0. §36: cubic'in state'i taşındığında zarar veriyor. Bu cümle mevcut kayıtla savunulamaz. |
| 76 | "later **verified to outperform** exp in sparse long-horizon retention" | **ABARTILI** | §29b atfı daralttı, §36 LM ölçeğinde tersini buldu. "verified" fazla güçlü. |
| 108 | "validates `cubic_flux + additive + dpfp` as the **target recipe for scaling**" | **ÇELİŞKİLİ** | §28b rafa kaldırdı, §36 ters çıktı. Ölçekleme hedefi olarak sunulamaz. |
| 187 | "(c) **appears beneficial** for sparse long-horizon recall" | **ÇELİŞKİLİ** | Aynı gerekçe. |
| — | **KV-cache confound / erratum hiç yok** | **DESTEKSİZ (eksik)** | Taslak §30'dan önce yazılmış. Projenin en yüksek etkili bulgusu makalede yok, ve makalenin retrieval iddiaları o bulgudan **doğrudan etkileniyor**. |

---

## Öncelik sırası

1. **`README.md` 76-78 — pratik kural.** Bir okuyucunun eyleme geçireceği tek
   cümle bu ve şu an kendi kaydımızla çelişiyor. Tek satırlık düzeltme.
2. **`README.md` 117-119 — "locked" reçete.** Aynı çelişki, farklı yer.
3. **`PROJECT_SUMMARY.md` 45-52.** Dış paydaşa giden tek sayfa; §36 kendi
   maddesini hak ediyor.
4. **`README.md` 69-72 ve 120.** Atıf düzeltmesi + birleştirmenin ayrılması.
5. **`.tex` taslağı.** En ağır ama en az acil — henüz yayınlanmadı. Yeniden
   konumlandırma ayrı bir iş (BEKLEYEN #13). **Bu hâliyle kimseye
   gösterilmemeli.**

## Bu denetimin kapsamadıkları

- `docs/tr/DENEY_SONUCLARI.md` (Türkçe ayna) — bakılmadı, muhtemelen aynı
  sorunları taşıyor.
- `hf_upload/GRAFT_MODEL_CARD.md` ve `hf_upload/README.md` — bakılmadı;
  ağırlık yayınından önce denetlenmeli (Kapı 2).
- `docs/osf_companion.pdf` — ikili, denetlenmedi.
- `RESULTS.md`'nin kendi iç tutarlılığı — bu denetim dış belgeleri RESULTS'a
  karşı ölçtü, RESULTS'u kendi içinde denetlemedi.
