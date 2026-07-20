# Geliştirme Planı — jcode ile temiz & profesyonel ilerleme

Bu belge, HFP'yi jcode ile geliştirirken kullanacağın hazır promptları ve sırayı
içerir. Amaç: kodu iyileştirmek, düzenli dosyalamak ve belgeleri profesyonel
göstermek — **ama her adım bilimsel dürüstlük kurallarına (bkz. `AGENTS.md`) bağlı
kalarak.** Her promptu jcode oturumuna olduğu gibi yapıştırabilirsin.

> ⚠️ "Profesyonel görünüm" = açıklık, tutarlılık, tekrarlanabilirlik ve düzen
> demektir. **Sonuçları olduğundan iyi göstermek değildir.** Bir sayı, grafik ya
> da iddia asla cilalanıp abartılmaz; sadece daha net ve düzgün sunulur.

Önerilen sıra: **1) Kod düzeni → 2) Belgeler → 3) Model iyileştirme.** Önce zemini
temizle, sonra anlatımı düzelt, en son davranışı değiştiren riskli işi yap.

---

## Adım 1 — Kod dosyalama & organizasyon (davranışı DEĞİŞTİRMEDEN)

Bu adım yalnızca düzen/okunabilirlik; hiçbir sayısal sonuç değişmemeli.

```
Görev: Repoyu davranışı DEĞİSTİRMEDEN daha temiz ve profesyonel hale getir.
Kapsam:
- docs/internal_tr/REPO_STRUCTURE.md'yi oku; gerçek yapıyla tutarlı mı, güncelle.
- Kök dizindeki gevşek scriptleri (eval_*.py, run_experiment.py, smoke_test.py,
  train.py) mantıklı bir düzende değerlendir; TAŞIMADAN önce her taşımanın import
  yollarını ve NASIL_CALISTIRILIR.md'deki komutları bozup bozmadığını kontrol et.
  Riskliyse taşıma, sadece öner.
- Ölü kod / kullanılmayan importları temizle; _legacy_reference/ dokunma.
- Tutarlı biçimlendirme uygula (varsa proje stiline uy; yeni araç ekleme).
- Değişken/fonksiyon isimlerinde açık iyileştirmeler yap ama genel API'yi koru.
Kısıtlar:
- Sayısal davranış birebir aynı kalmalı. Her değişiklikten sonra
  python smoke_test.py VE python review_scripts/verify_claims.py YEŞİL olmalı.
- hf_upload/ ve hfp/ arasındaki kopya dosyaları (modeling_hfp.py,
  configuration_hfp.py) senkron tut.
Çıktı: Ne değiştirdiğini kısa bir özetle listele; taşınan/riskli şeyleri ayrı belirt.
Testler geçmeden "tamam" deme.
```

---

## Adım 2 — Belge & anlatım iyileştirme (profesyonel, dürüst)

README/RESULTS/docs'u daha net ve profesyonel yap; iddiaları abartma.

```
Görev: Belgeleri profesyonel, tutarlı ve dürüst hale getir. İddiaları
GÜÇLENDIRME; yalnızca netlik, tutarlılık ve sunum kalitesini artır.
Kapsam:
- README.md, RESULTS.md, CHANGELOG.md, NASIL_CALISTIRILIR.md ve docs/tr +
  docs/internal_tr'yi gözden geçir.
- Tutarlılık: terminoloji, notasyon (M, z, λ, η), komut örnekleri ve sayılar
  belgeler arası çelişmesin. README'deki "Honesty note" çizgisini koru.
- Her rapor edilen sayının kaynağı (hangi komut/ayar ürettiği) belli olsun;
  belirsiz/eski sayı varsa "doğrulanmadı" diye işaretle, UYDURMA.
- Dil: Türkçe belgeler ile İngilizce README arasında anlam tutarlılığı sağla
  (birebir çeviri şart değil, ama çelişki olmasın).
- Yapı: başlık hiyerarşisi, çalışan içindekiler/bağlantılar, kod bloklarının
  doğruluğu, yazım hataları.
Kısıtlar:
- Hiçbir metrik/grafik değiştirilmez veya güzelleştirilmez. Sadece daha iyi
  ANLATILIR. Kanıtlanmamış hiçbir şeye "kanıtlandı" denmez.
- Belgede geçen her komutun gerçekten çalıştığını (kopyala-çalıştır) doğrula.
Çıktı: Değişiklik listesi + hâlâ doğrulanması gereken (şüpheli/eski) sayıların ayrı listesi.
```

---

## Adım 3 — Model iyileştirme (GERÇEK ve ÖLÇÜLMÜŞ)

Burada davranış değişir; en yüksek dikkat bu adımda.

```
Görev: HFP modelinde somut bir iyileştirme yap ve etkisini DÜRÜSTÇE ölç.
Önce plan:
- docs/internal_tr/FUTURE_RESEARCH_HYPOTHESES.md ve SONRAKI_ADIMLAR_PLANI.md'yi
  oku; oradan tek, net ve ölçülebilir bir iyileştirme hipotezi seç (ör. belirli
  bir görevde recall doğruluğu ↑ veya LM ppl ↓). Bana seçtiğin hipotezi ve nasıl
  ölçeceğini SUN, onaydan sonra kodla.
Uygulama & ölçüm:
- Değişiklikten ÖNCE baseline'ı koştur ve sayıları kaydet (aynı seed, aynı ayar).
- Değişikliği yap; SONRA aynı komutla tekrar koştur. Yalnızca değiştirdiğin
  şey farklı olsun (kontrollü karşılaştırma); başka hiçbir hiperparametreyi
  gizlice değiştirme.
- İyileşme gerçekse rapor et; yoksa ya da kötüleştiyse bunu AÇIKÇA söyle —
  olumsuz sonuç da geçerli sonuçtur, gizlenmez.
Kısıtlar (AGENTS.md):
- Sonuç uydurma, seed-hacking/cherry-pick, test gevşetme, eval sızıntısı YOK.
- recall testinde --local_window kısıtı korunur (bilgi yalnızca bellekten aksın).
- Her değişiklikten sonra python smoke_test.py + review_scripts/verify_claims.py YEŞİL.
Çıktı: baseline sayı → yeni sayı, üreten TAM komutlar (seed dahil), ve dürüst yorum.
Testler yeşil ve iyileşme ölçülmeden "tamam" deme.
```

---

## Her adım sonrası kontrol listesi

- [ ] `python smoke_test.py` → tüm testler PASS
- [ ] `python review_scripts/verify_claims.py` → geçti
- [ ] Değişiklikler kısa ve dürüstçe özetlendi (ne, neden, hangi sayıyla)
- [ ] Belgelerdeki komutlar gerçekten çalışıyor
- [ ] Şüpheli/doğrulanmamış hiçbir sayı "kesin" gibi sunulmadı
- [ ] **Lisans korundu:** AGPL-3.0 başlıkları/telif yerinde; yeni dosyalar AGPL;
      AGPL ile bağdaşmayan/dış kaynaklı kod atıfsız eklenmedi (bkz. AGENTS.md → Lisans)

> Not: Ortam şu an Python 3.12.7; AGENTS.md hedefi 3.10. Testler geçiyor ama
> CI-eşdeğer/yayın işi için 3.10'a geçmeyi düşün.
