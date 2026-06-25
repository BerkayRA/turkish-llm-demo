# Türkçe Tokenizer Görselleştirici

Yönetici sunumuna uygun, Gradio tabanlı bir demo. Aynı Türkçe metni **üç
tokenizer** ile böler ve bizim SentencePiece tokenizer'ımızın daha az / daha
temiz token ürettiğini (fertility avantajı) görsel olarak gösterir.

Karşılaştırılan tokenizer'lar:

| Tokenizer | Kaynak | Not |
|-----------|--------|-----|
| **BİZİM** | SentencePiece unigram, vocab 32000 (`sp_unigram_32000.model`) | `sentencepiece` paketi ile yüklenir |
| **GPT-4o** | `tiktoken` · `o200k_base` | |
| **Llama-3** | `transformers.AutoTokenizer` (`meta-llama/Meta-Llama-3-8B`) | Çevrimdışı yoksa `tiktoken` `cl100k_base` (GPT-3.5/4 vekili) olarak düşer ve açıkça etiketlenir |

**Fertility** = token sayısı / kelime sayısı (boşlukla ayrılmış). Düşük olması
daha iyidir. Metrik fikri `turkish-corpus/src/turkish_corpus/fertility.py`
dosyasından alınmıştır.

## Özellikler

- 4 seçilebilir Türkçe örnek paragraf (haber, resmî/hukukî, sohbet, eklemeli-ağır).
- Her tokenizer için: renkli token chip'leri (token sınırları net görünsün diye
  ardışık renkler), token sayısı ve token/kelime fertility.
- Üç fertility'yi karşılaştıran bar grafik; en verimli olan vurgulanır ve
  "Bizim tokenizer: %X daha az token" özeti gösterilir.
- Tamamen Türkçe arayüz, sunum kalitesinde stil.

## Çalıştırma

```bash
cd /Users/berkayra/dev/turkish-llm-demo/tokenizer_viz
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Tarayıcıda: <http://localhost:7860>

> Not: ML wheel'leri için Python 3.10–3.12 önerilir (3.14'te bazı paketlerin
> wheel'i henüz yok).

## Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `OUR_SP_MODEL` | `/Users/berkayra/Downloads/tokenizer/turkish-llm/models_fresh/sp_unigram_32000.model` | Bizim SentencePiece modelinin yolu |
| `LLAMA3_MODEL_ID` | `meta-llama/Meta-Llama-3-8B` | Llama-3 model kimliği (çevrimdışı denenir) |

**Dağıtım kutusunda** model yolu farklıdır:

```bash
export OUR_SP_MODEL=/opt/corpus/out/tokenizer/sp_unigram_32000.model
python app.py
```

## Çevrimdışı / Eksik Bağımlılık Notları

- **Llama-3**: `transformers.AutoTokenizer` yalnızca `local_files_only=True` ile
  denenir. Model yerel önbellekte yoksa veya HF kimlik doğrulaması gerekiyorsa
  uygulama çökmez; `tiktoken` `cl100k_base` vekiline düşer ve "GPT-3.5/4 vekili"
  olarak etiketler.
- **Eksik model / paket**: Herhangi bir tokenizer yüklenemezse o kart bir uyarı
  gösterir; diğer tokenizer'lar normal çalışmaya devam eder.
