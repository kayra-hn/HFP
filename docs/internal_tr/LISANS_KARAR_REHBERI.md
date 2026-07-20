# LİSANS KARAR REHBERİ — kritik anlar ve hazır adımlar

> ⚠️ **Bu belge projenin en önemli hatırlatıcısıdır.** Üç kritik "karar kapısı"
> vardır; her biri geldiğinde bu belge açılır ve checklist uygulanır.
> Temel ilke: **Telif sahibi kendi lisansıyla bağlı değildir** — AGPL başkalarını
> kısıtlar, seni değil. Bütün strateji, telifin %100 sende kalmasına dayanır.
> (Not: Bu rehber hukuki danışmanlık değildir; kapı 3'te avukat şart.)

---

## KAPI 1 — İlk dış katkı (PR) geldiğinde

**Risk:** CLA'sız kabul edilen TEK bir PR bile çift-lisans (ticari) yolunu
sonsuza dek kilitler — o katkı yalnız AGPL-only kalır ve ticari lisansta
kullanılamaz.

Checklist:
- [ ] PR sahibi `CLA.md`'yi kabul etti mi? Kanıt: PR açıklamasında açık onay
      cümlesi + commit'lerde `Signed-off-by: Ad Soyad <eposta>`.
- [ ] Onay yoksa: PR'a nazik standart yanıt — "Merge edebilmemiz için CLA.md'yi
      kabul etmen (PR'a yazman) ve commit'leri signed-off yapman gerekiyor."
- [ ] Onaysız HİÇBİR katkı merge edilmez — küçük typo düzeltmesi bile
      (istisna yok; typo'yu kendin ayrıca düzelt, PR'ı kapat).
- [ ] (Ölçek büyürse) CLA-assistant benzeri bir GitHub botu kur.

Mevcut durum: `CLA.md` v1.0 relisans hakkını açıkça veriyor (§2) — değiştirme;
değiştireceksen önce avukata danış.

---

## KAPI 2 — HF'ye model ağırlığı yüklerken

**Risk:** Ağırlık lisansı kod lisansından AYRI bir karardır; bir kez yayınlanan
ağırlığın lisansı fiilen geri alınamaz. Ayrıca taban modelin (Qwen) atıf
yükümlülüğü atlanırsa Apache-2.0 ihlali olur.

Checklist:
- [ ] Ağırlık lisansına bilinçli karar ver (seçenekler ve bedelleri):
      - **AGPL-3.0 (mevcut tercih):** maksimum koruma, copyleft sinyali;
        sanayi kullanımını caydırır. "Fikir çalınmasın" önceliğine en uygun.
      - **Apache-2.0/MIT:** maksimum yayılım/atıf; koruma minimum.
      - **OpenRAIL / NC varyantı:** kullanım-kısıtlı orta yol; OSI-uyumlu değil.
      Karar defteri: seçimi ve gerekçeyi bu dosyanın altına tarihle yaz.
- [ ] Model kartı frontmatter'ı doğru mu?
      Grafted model için mevcut şablon DOĞRU: `license: other`,
      `license_name: apache-2.0-base-agpl-3.0-adapter`,
      `base_model: Qwen/Qwen2.5-1.5B` (bkz. `hf_upload/GRAFT_MODEL_CARD.md`).
- [ ] Qwen atfı kartta duruyor mu? ("Base model … Apache 2.0 © Alibaba Cloud"
      satırı + Qwen lisans dosyasının paketle taşınması.)
- [ ] `hf_release/` paketinde `LICENSE` (AGPL tam metin) var mı? (var — koru)
- [ ] Eğitim verisi bildirimi: WikiText-103 (CC BY-SA kaynaklı) model kartının
      "training data" bölümünde anılmalı.
- [ ] Taban model DEĞİŞİRSE (ör. Llama): yeni tabanın lisansı BAŞTAN okunur —
      Llama-türü lisanslar ek kısıt getirir; "Apache gibi" varsayma.

---

## KAPI 3 — API/ticarileşme ciddileştiğinde (veya büyük yayından ÖNCE)

**Risk 1 — Patent/yayın zamanlaması:** Yayınlanan her şey prior art olur:
kimse patentleyemez (savunma ✓) ama SEN de patentleyemezsin (TR/EPO'da
yayın-sonrası hoşgörü yok). Cubic-plateau mekanizmasında patent düşünülüyorsa
karar, makale/OSF büyük yayınından ÖNCE verilmelidir.

**Risk 2 — Çift-lisans yapısı:** Kendi API'n için AGPL engel değil (§13
başkalarını bağlar, seni değil). Ama müşteriye ticari lisans satışı
başlayacaksa yapı resmileşmeli.

Checklist:
- [ ] Fikri mülkiyet avukatıyla görüş (çift-lisans sözleşme şablonu + CLA
      geçerlilik teyidi + marka).
- [ ] Patent kararını kapat: ya "açık yayın = savunma stratejisi" diye yaz,
      ya başvuruyu yayından önce yap. Kararsız kalınmaz; karar tarihiyle
      bu dosyaya işlenir.
- [ ] Telif bütünlüğü denetimi: `git log`'da CLA'sız dış katkı var mı tara.
- [ ] "HFP" adı için marka tescili değerlendir (telif ≠ marka).
- [ ] AGPL kalır — ticari müşteri ayrı sözleşmeyle lisanslanır; repo lisansı
      değiştirilmez (topluluk güveni + koruma).

---

## Sürekli kurallar (her zaman geçerli — AGENTS.md ile uyumlu)

- LICENSE/telif başlıkları asla kaldırılmaz/zayıflatılmaz.
- Dış kod kaynak+lisans notu olmadan repoya girmez; AGPL-uyumsuz kod girmez.
- Yeni dosyalar AGPL-3.0'dır; `hf_upload/hf_release/LICENSE` paketle taşınır.
- Bu rehberdeki her karar, tarih + gerekçeyle bu dosyanın sonuna eklenir.

## Karar defteri

- 2026-07-18: Kod lisansı AGPL-3.0-only olarak teyit edildi; kök `LICENSE`
  eklendi; `pyproject.toml` lisans metadatası AGPL-3.0-only. Grafted model
  kartı şablonu: apache-2.0-base + agpl-3.0-adapter. (Ağırlık lisansı nihai
  kararı Kapı 2'de verilecek.)
