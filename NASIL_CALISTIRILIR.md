# HFP — Deney Çalıştırma Kılavuzu

Aşağıdakiler senin makinende çalışacak tam komutlar ve beklenen çıktılardır.
Tüm komutlar **repo kökünden** (`HFP_Project/`) çalıştırılır.

## 0. Kurulum + İLK İŞ: smoke test
```bash
cd HFP_Project
pip install torch transformers numpy matplotlib
python smoke_test.py        # CPU, ~1 dk
```
Smoke test şunları mühürler: bellek parametrelerinin gradyan aldığı (T2 — eski
sürümde almıyordu), MQAR loss'unun sonlu olduğu (T3 — eski sürümde NaN'dı),
recurrence matematiğinin chunk-bağımsız olduğu (T4). **T2/T4 geçmeden hiçbir
benchmark sonucuna güvenme.**

## 1. ASIL YETENEK TESTİ — Associative Recall (MQAR)
O(1) belleğin bilgiyi uzun bağlam boyunca gerçekten tutup tutmadığını ölçer.

Önemli tasarım notu: model `--local_window 32` ile kurulur — sorgu, pencere
dışındaki k-v çiftlerini attention'la GÖREMEZ; bilgi yalnızca recurrent
bellekten (M/z) akabilir. Bu olmadan test attention'ı ölçer, belleği değil.

CPU (hızlı, küçük):
```bash
python run_experiment.py --task recall --steps 1500 --context 128 --pairs 8
```
GPU (uzun bağlam stresi):
```bash
python run_experiment.py --task recall --steps 3000 --context 512 --pairs 16
```

Sonuç bloğu üç sayı verir:
```
[RECALL SONUC]  sans seviyesi = 1.00%
  full    (tek forward, pencere=32) : XX.X%
  chunked (state tasinir, chunk=64) : XX.X%
  reset   (state SIFIRLANIR)        : XX.X%
```
YORUM (dürüst okuma):
- **chunked ≫ şans VE reset ≈ şans** → bilgiyi taşıyan şey gerçekten recurrent
  bellek. Bu, yayınlanabilir asıl sonuçtur.
- **reset de yüksekse** → test belleği ölçmüyor (bilgi başka yoldan sızıyor);
  rapor edilecek dürüst bulgu, pencere/chunk ayarları gözden geçirilmeli.
- **chunked ≈ şans** → bellek bilgiyi tutamıyor; eğitim/kapasite sorunu —
  bu da rapor edilecek dürüst bulgudur, gizlenmez.
- `--context`'i büyütüp doğruluk korunuyorsa (VRAM sabitken), "O(1) bellek +
  uzun-menzil hatırlama" birleşik iddiası kanıtlanmış olur.

## 2. GERÇEK PERPLEXITY — TinyShakespeare LM
```bash
python run_experiment.py --task lm --steps 3000 --seq 256
```
Beklenen: `val loss` düşer, `ppl` yazdırılır. Küçük 4-katman modelde birkaç bin
adımda ppl ~60-150 tipiktir. Bu sayıyı README/makalede gerçek perplexity olarak
kullan. İsteğe bağlı A/B: `--lm_window --local_window 32` ile pencereli mod —
bellek LM kalitesine gerçekten katkı veriyor mu, dürüst ölçüm.

## 3. Passkey hakkında dürüst not
`eval_passkey.py` mevcut ama: rastgele 5-haneli sayıyı yalnızca dil-modeli
eğitiminden sonra hatırlamak, modele ayrıca öğretilmeyen bir görevdir → büyük
olasılıkla ~%0 verir (eski grafik bu yüzden düz sıfırdı; ayrıca eski mimaride
bellek yolu eğitilmiyordu — bkz. DEGISIKLIKLER Track E/K2). Bellek yeteneği
için §1'deki MQAR doğru testtir.

## 4. Standart LM eğitimi (checkpoint üretir)
```bash
python train.py --model hfp --optimizer adamw --max_iters 3000     # sağlıklı default
python train.py --model hfp --optimizer thermodynamic --max_iters 3000  # opsiyonel A/B
```

## 5. VRAM O(1) kanıtı
```bash
python eval_memory_scaling.py
```
Sabit-VRAM eğrisini üretir. Dürüst sunum: bu eğri tek başına "bellek çalışıyor"
demek değildir (hiçbir şey hatırlamayan sabit state de düz çizgi verir);
§1'deki chunked-recall sonucuyla BİRLİKTE sun.

## 6. Google ADK Entegrasyonu (Deneysel)
Google Agent Development Kit (ADK) entegrasyonu, HFP'yi otonom ajanların
bir "aracı (tool)" veya bellek arka ucu olarak kullanması için geliştirilmektedir.
Önemli Not: HFP **AGPL-3.0** lisanslıdır; dış servislerden bağlanırken bu
lisans sınırlarına uyulmalıdır (açık kaynak SDK'lar uyumludur).

**Kurulum:**
```bash
python -m pip install google-adk google-genai
```
**Kullanım Arayüzü:**
`hfp/agent_integration.py` içerisinde `HFPAgentWrapper` sınıfı yer almaktadır.
Bu sınıf cihaz içi O(1) sabit bellek özelliği kullanılarak chunked çalışma
destekler. Agent'a tool olarak bağlamak için:
```python
from hfp.agent_integration import HFPAgentWrapper
# config=None verilirse varsayılan model mimarisi yüklenir
agent_hfp = HFPAgentWrapper(device="cpu")
# agent_hfp.generate_response(...) ile aracı tetikleyebilirsiniz.
```
