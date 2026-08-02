# TRC (TPU Research Cloud) başvuru taslağı

*2026-08-02. Başvuru formu: https://sites.research.google/trc — kayıt olup ilgi
bildiriliyor, davetler sırayla gönderiliyor.*

## Bilmen gerekenler (kaynaklardan, 2026-08-02 itibarıyla)

- Kabul edilenlere **ücretsiz Cloud TPU kotası** veriliyor, geçici süreyle,
  kendi GCP projelerine tanımlanıyor.
- TensorFlow, PyTorch, JAX destekleniyor.
- **Beklenti:** desteklenen araştırmanın dünyayla paylaşılması — hakemli yayın,
  açık kaynak kod, blog yazısı ya da başka bir yolla.
- **Dikkat: TPU ücretsiz ama diğer GCP hizmetleri değil.** Küçük VM'ler ve veriyi
  tutmak için Cloud Storage bucket'ları kendi cebinden. Bu, sıfır maliyet
  olmadığı anlamına geliyor; küçük ama sıfır değil.
- Davetler **rolling** gönderiliyor, yani sabit bir başvuru dönemi yok.

**Doğrulamadığım şeyler:** güncel kota büyüklüğü, süre uzunluğu, ve
başvurudan yanıta kadar geçen tipik süre. Başvurmadan önce
sites.research.google/trc üzerindeki FAQ'yu kendin oku — bu belge ikincil
kaynaklardan derlendi.

## Neden şimdi mantıklı

Yol haritasında "grant'lar ancak §34 tutup ölçek büyürse anlamlı, TRC en
gerçekçi" yazıyordu ve o gerekçe **graft'ı ölçeklemek** senaryosuna aitti.
§34a graft-hafıza yolunu kapattı, §36 cubic'in LM-ölçek savunmasını kapattı, ve
yön hafıza-öncelikli bir mimariyi sıfırdan eğitmeye döndü. O iş için TPU
erişimi, T4 kotasıyla haftalar süren şeyi günlere indirir.

Ayrıca TRC'nin beklentisi (açık kaynak + yayın) bu projenin zaten yaptığı şey —
AGPL-3.0 repo, tam deney kaydı, yayınlanmış negatifler.

## Başvuruda anlatılacak hikâye (taslak, İngilizce)

> **Research summary**
>
> I am an independent researcher studying whether a constant-size (O(1))
> recurrent state can serve as a genuine long-term memory in a causal language
> model, as opposed to an efficiency mechanism.
>
> Over the past month I built and published a complete experimental record
> (AGPL-3.0, ~36 pre-registered sections, roughly two thirds negative results)
> covering: a working O(1) graft into a frozen Qwen2.5-1.5B that holds
> perplexity at 1.111× baseline with 149,910 trainable parameters across 3
> seeds; a mapped density wall with a compounding-error diagnosis; and a
> methodological finding I believe is the most useful part of the work — that
> apparent long-range retrieval in attention/O(1) hybrids can be carried
> entirely by the KV cache of the un-grafted layers. Resetting that cache per
> chunk, retrieval from the recurrent state alone is 0% at every distance
> tested, including across a single chunk boundary. I recorded this as an
> erratum against my own earlier claims.
>
> Four pre-registered interventions failed to produce cross-boundary storage,
> which localises the limitation to the architecture rather than the training
> recipe. The next step is therefore a memory-first architecture trained from
> scratch with retrieval as a primary objective, rather than grafted onto a
> frozen model.
>
> **What I would use TPU access for**
>
> Training small memory-first language models (targeting roughly 100M
> parameters) from scratch, with an evaluation protocol that isolates the
> recurrent state from the attention cache from the first day. All work to date
> has run on free-tier T4 GPUs (~30 hours per week), which caps a single
> training run at multiple checkpointed sessions and makes multi-seed
> comparisons impractical. TPU access would let me run the seed counts my own
> power analyses say are required.
>
> **Sharing**
>
> The repository is already public under AGPL-3.0 with the full experimental
> record, including negative results and errata. Any TRC-supported work would be
> published the same way, and I intend to write up the KV-cache confound finding
> separately as it is relevant to anyone evaluating hybrid architectures.

## Yazarken dikkat

- **Sayılara sadık kal.** 1.111× (3 seed) kullan, 1.043×'i kullanma — o tek seed
  ve replike edilmedi. Karıştırmak DD'de en kötü izlenim.
- **Cubic'i moat gibi sunma.** §36 LM ölçeğinde dezavantaj ölçtü ve repo public;
  kim bakarsa görür.
- **"Hafıza çalışıyor" deme.** Çalışmıyor, ve zaten anlatının gücü onu dürüstçe
  ölçmüş olmandan geliyor.
- Türkçe karakter kullanma diye bir kısıt yok burada — form İngilizce.

## Sonrası

Davet gelirse: TPU'da HFP'nin recurrent yolunu koşturmak ayrı bir mühendislik
işi (JAX ya da PyTorch/XLA). Bunu davet gelmeden yapma — belki gelmez, belki
kota beklediğinden farklı olur.
