# Egemen Türkçe Yapay Zeka — Kontrol Paneli

Yönetici sunumu için tek sayfalık, yukarıdan aşağı kaydırmalı kontrol paneli.
Tamamı statik (HTML + CSS + vanilla JS + Chart.js CDN). Hiçbir derleme adımı
gerektirmez; bir dosya sunucusuyla yayınlanır.

## Dosya yapısı

```
dashboard/
├── index.html        # 6 bölümlük tek sayfa (Türkçe)
├── styles.css        # tasarım sistemi (dark luxury / editorial) — düzenlemeyin
├── config.js         # tüm IP / port / yol değerleri burada — DAĞITIMDA BUNU DÜZENLEYİN
├── dashboard.js      # tüm davranış (reveal, sayaçlar, JSONL çekme, grafik, yoklama)
├── README.md
└── sample_data/
    ├── train_log.jsonl   # örnek kayıp eğrisi fixtürü
    └── samples.jsonl     # örnek "modelin öğrenişi" fixtürü
```

## Yerelde çalıştırma

`config.js` varsayılan olarak yanındaki `sample_data/` fixtürlerini okur, böylece
gerçek eğitim başlamadan önce de panel tam görünür.

Bu klasörün içinden:

```bash
cd dashboard
python -m http.server 8080
```

Ardından tarayıcıda: <http://localhost:8080>

> `file://` ile **doğrudan açmayın** — `fetch()` (JSONL) ve bazı tarayıcı
> davranışları `file://` altında çalışmaz. Mutlaka bir HTTP sunucusu kullanın.

## Kutuda (on-prem) dağıtım

Tüm dış adresler `config.js` içinde toplanmıştır. Dağıtırken **yalnızca bu
dosyayı** düzenleyin (`styles.css`, `index.html`, `dashboard.js` dokunmadan kalır).

| Anahtar | Açıklama | Örnek |
|---|---|---|
| `BOX_IP` | Tüm servislerin çalıştığı yerinde sunucu IP'si | `10.0.0.5` |
| `TOKENIZER_VIZ_URL` | Gradio tokenizer görselleştirici (Bölüm 02 butonu) | `http://10.0.0.5:7860` |
| `CHAT_URL` | Open WebUI sohbet ürünü (Bölüm 04) | `http://10.0.0.5:11436` |
| `CHAT_EMBED` | Sohbeti iframe içinde göstermeyi dene (engellenirse butona düşer) | `true` |
| `TRAINING_DATA_BASE_URL` | Eğitim JSONL dosyalarının taban URL'i | `./sample_data` ya da `http://10.0.0.5:8088` |
| `TRAIN_LOG_FILE` | Kayıp kaydı dosya adı | `train_log.jsonl` |
| `SAMPLES_FILE` | Örnek çıktı dosya adı | `samples.jsonl` |
| `POLL_INTERVAL_SECONDS` | Canlı verinin kaç saniyede bir yeniden çekileceği (`0` = kapalı) | `15` |

Panelin kendisini de kutuda yayınlamak için bu klasörü sunun:

```bash
cd dashboard
python -m http.server 8080
# yöneticiler: http://10.0.0.5:8080
```

## Canlı eğitim verisine bağlanma

Bölüm 03 (Eğitim) iki canlı kaynağı çeker ve `POLL_INTERVAL_SECONDS` aralığıyla
yeniden çeker:

- kayıp eğrisi → `${TRAINING_DATA_BASE_URL}/${TRAIN_LOG_FILE}`
- örnek akışı → `${TRAINING_DATA_BASE_URL}/${SAMPLES_FILE}`

Gerçek eğitim çalışmasını izlemek için eğitim işinin yazdığı `out_dir`'i ayrı bir
basit sunucuyla yayınlayın ve `TRAINING_DATA_BASE_URL`'i ona yöneltin:

```bash
# eğitim kutusunda, out_dir içinden (train_log.jsonl + samples.jsonl burada üretilir)
cd out/U32_seed0
python -m http.server 8088
```

Sonra `config.js`:

```js
TRAINING_DATA_BASE_URL: "http://10.0.0.5:8088",
```

Eğitim sürdükçe panel, kayıp eğrisini ve örnek çıktıları otomatik tazeler.
Veri henüz yoksa ya da sunucuya ulaşılamıyorsa panel zarifçe **"veri
bekleniyor"** durumunu gösterir (canlı nokta gri/duraksamış olur).

> **CORS notu:** `TRAINING_DATA_BASE_URL` panelden farklı bir kökendeyse (farklı
> port da farklı kökendir), `out_dir` sunucusunun CORS başlığı göndermesi
> gerekir. Hızlı çözüm: paneli ve `out_dir`'i aynı kökenden sunun ya da CORS'a
> izin veren bir statik sunucu kullanın (ör. `npx http-server --cors`).

## Bağımlılıklar

- **Chart.js 4.4.3** — jsDelivr CDN üzerinden (`index.html` içinde sabitlenmiş).
  İnternetsiz kapalı ağda dağıtım için `chart.umd.min.js` dosyasını yerele
  indirip `index.html`'deki `<script src>`'i yerel yola çevirin.
- Fontlar (Fraunces + Inter + JetBrains Mono) Google Fonts'tan çekilir; aynı
  şekilde gerekirse yerele alınabilir.
