# GÖREV: HFP Projesinin Yeni Halini Değerlendir (Kod + Makaleler)

Sen kıdemli bir değerlendirici ajanısın (ML mühendisliği + teorik fizik). Görevin,
"Hyper-Flux Projection (HFP)" projesinin GÜNCEL halini —hem kodu hem iki makaleyi—
eleştirel, dürüst ve bütüncül biçimde analiz etmek. Övme; zayıf noktaları ve tutarsızlıkları
açıkça söyle. İddiaları kod ve makaleyle KANITLAYARAK değerlendir.

## Okuman gereken dosyalar

Makaleler (fizik çekirdeği — ANA FİKİR bunlarda):
- C:\Users\yilma\Downloads\HFP_Paper_I_Revised.tex
  (5D Einstein-dilaton; no-go teoremi; ufuksuz brane-capped ECO; bilgi = Kasner modüllerinde)
- C:\Users\yilma\Downloads\HFP_Paper_II_Revised.tex
  (kuantum mekaniğinin geometrik yorumu; çökme = modül seçimi; Born ağırlığı Haar ölçüsünden)

Kod (temiz sürüm — burada çalış):
- C:\Users\yilma\Documents\HFP_Project\  (paket doğrudan `hfp/` altında; komutlar repo kökünden)
  - hfp/core/hfp_bulk_state.py      (lineer-attention O(1) matris belleği: M, z + decay + ring buffer)
  - hfp/core/bulk_trigger_decoder.py(decoder katmanı, EntangledFFN/EntangledLinear, TunnelingDropout)
  - hfp/core/physics_optimizers.py  (opsiyonel termodinamik optimizer — lineer relaksasyon)
  - hfp/core/hfp_config.py, hfp/core/hfp_utils.py
  - hfp/models/modeling_hfp.py, hfp/models/configuration_hfp.py (HuggingFace entegrasyonu)
  - train.py, run_experiment.py (MQAR + LM deneyleri), eval_*.py
  - DEGISIKLIKLER.md (yapılan düzeltmelerin kaydı), NASIL_CALISTIRILIR.md

Eski/orijinal referans kod (karşılaştırma için, DEĞİŞTİRME): C:\Users\yilma\Documents\HFP_Project\_legacy_reference\

## Bağlam (kısa)

- ANA FİKİR: 5 boyuttan 4 boyuta PROJEKSİYON/düşüş. Kodda karşılığı iddia edilen: her katmanın
  kendi "Bulk" hafızası + EntangledLinear (tek bir Bulk ağırlığından iki projeksiyon P_A, P_B).
- İLKE: Kod ve mimari SAĞLIKLI olmalı; fizik "İLHAM" olarak kalır (ölü kod ya da çürütülmüş teori
  olarak değil). Fizik bir izomorfizm/simülasyon DEĞİL, ilham kaynağı olarak sunulmalı.
- Yakın zamanda yapılan düzeltmeler (doğrula): max_position_embeddings hatası giderildi;
  chunked prefill'de global pozisyon offseti eklendi; ölü parametreler (medium/long_freq) temizlendi;
  dekoratif fizik aux-kayıpları default kapatıldı ve dürüstçe "opsiyonel/deneysel" işaretlendi;
  gate-entropy gradyanlı+opsiyonel bağlandı; optimizer kübik→lineer relaksasyona güncellendi
  (default artık standart AdamW+cosine); weight tying eklendi.
- Bilinen dürüst sonuç durumu: (a) O(1) VRAM benchmark'ı GERÇEK ve güçlü;
  (b) termodinamik optimizer normal LR'de sade AdamW'den farksız (atıl);
  (c) passkey grafiği eğitimsiz model → %0 (yetenek kanıtı değil). run_experiment.py MQAR ile
  gerçek bir bellek-tutma testi öneriyor.

## Senden istenen analiz (her başlık için KANIT + öneri)

1. KOD SAĞLIĞI: Mimari tutarlı mı? Gizli bug, kopuk gradyan, ölü kod, tutarsız iki-config,
   ring buffer'da yazılmamış sıfır-slot dikkati (D2), mixed-precision dtype riski (D3) var mı?
   HuggingFace entegrasyonu (generate/cache/past_key_values) doğru çalışır mı?

2. KOD ↔ MAKALE TUTARLILIĞI: Makaledeki 5D→4D projeksiyon, modül-seçimi, bilgi-in-moduli
   fikirleri kodda GERÇEKTEN karşılık buluyor mu, yoksa yalnızca isim benzerliği mi?
   EntangledLinear "projeksiyon" iddiasını hak ediyor mu? Fizik dürüstçe "ilham" seviyesinde mi
   sunulmuş, yoksa hâlâ fazla iddia (izomorfizm/simülasyon) kalıntısı var mı?

3. MAKALELERİN BİLİMSEL SAĞLAMLIĞI: Paper I'de brane'in ad-hoc'luğu, WEC ihlali, Liouville≠AdS5,
   dar KK penceresi ve özgünlük; Paper II'de Born kuralı (kanıtlanan vs varsayılan), no-signaling
   (equivariance) açık problemi. Argümanlar sağlam mı, hangi boşluklar kapatılmalı?

4. SONUÇLARIN DÜRÜSTLÜĞÜ: Mevcut grafiklerin (O(1) VRAM, optimizer stability, passkey) hangileri
   gerçek kanıt, hangileri fazla iddialı? MQAR deney tasarımı (run_experiment.py) bellek-tutmayı
   doğru ölçüyor mu? Eksik/yanıltıcı ölçüm var mı?

5. YOL HARİTASI: En yüksek kaldıraçlı 3-5 somut adım. Kod tarafı ve makale tarafı ayrı ayrı.
   "Yeni ihtimal açabilecek" fizik-ilhamlı fikirleri (ör. Casimir'den brane, thick-brane,
   lineer-dilaton holografisi) kodla veya makaleyle bağlayacak öneriler dahil.

## Çıktı formatı
- Kısa bir "Genel Değerlendirme" (5-8 cümle).
- Sonra her başlık (1-5) için: BULGULAR (kanıtla) → RİSK/ÖNEM → SOMUT ÖNERİ.
- Sonda: "En kritik 3 madde" ve "En güçlü 3 yön".
- Abartıdan kaçın; her iddiayı dosya/satır referansıyla destekle. Emin olmadığın yeri "doğrulanmalı"
  diye işaretle.
