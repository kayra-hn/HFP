# Hugging Face Yayın Adımları (Checklist)

Eğitim bittiğinde (12-13 saat sonra) sırasıyla uygulayacağınız adımlar:

## 1. Sonuçları README'ye Ekleyin
Colab ablasyon testi bittiğinde elde ettiğiniz nihai Perplexity/Loss metriklerini, bu klasördeki `README.md` dosyasında bulunan **[TODO: INSERT COLAB LM ABLATION RESULTS HERE]** başlıklı alana yapıştırın. (Bu dosyayı düzenleyip kaydedin).

## 2. Ağırlıkları (Weights) Kopyalayın
Eğitilen `model.safetensors` (veya `.pt`) dosyalarınızı root dizindeki eğitim çıktılarından (veya Colab'den indirdiğiniz klasörden) hazır edin. (Gerekirse klasörü `hf_upload` içine manuel olarak taşıyabilirsiniz).

## 3. Checkpoint Klasörünü Oluşturun
Root repoda (`HFP_Project/hf_upload`) komut satırını açıp şu komutu çalıştırın:
```bash
python make_hf_checkpoint.py --out hf_release
```
Bu komut; `make_hf_checkpoint.py` betiğini çalıştırarak, tüm `.py` dosyalarını, güncellenmiş `README.md`'yi ve `config.json` dosyasını `hf_release/` klasörünün içinde toplayacaktır.

## 4. Eğitilmiş Ağırlıkları Aktarın
Az önce Colab'den indirdiğiniz/eğittiğiniz model ağırlık dosyasını (`model.safetensors` vb.) `hf_release/` klasörünün içine sürükleyip bırakın (eski ağırlık varsa üzerine yazın).

## 5. Hugging Face'e Yükleyin
`hf_release/` klasörü artık mükemmel bir Hugging Face reposudur. İster arayüzden sürükleyip bırakarak (Hugging Face web sitesi üzerinden), isterseniz `huggingface-cli` ile yükleyebilirsiniz:

```bash
huggingface-cli upload kayrahan35/HFP-O1-Memory-Model ./hf_release .
```

*Not: Eğer repo adınız (`kayrahan35/...`) farklıysa yukarıdaki komutta düzeltmeyi unutmayın.*

Elinize sağlık! Bütün işlemler bu kadar. 🎉
