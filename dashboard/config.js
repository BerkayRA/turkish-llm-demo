/*
 * Egemen Türkçe Yapay Zeka — Kontrol Paneli yapılandırması
 * --------------------------------------------------------
 * Tüm dış IP / port / yol değerleri burada toplanmıştır.
 * Eğitim kutusunda dağıtırken yalnızca bu dosyayı düzenleyin.
 *
 * BOX_IP  : Demo bileşenlerinin (tokenizer, sohbet, eğitim JSON) yayınlandığı sunucu.
 * Varsayılan: localhost (yerinde dağıtımda kendi sunucu IP'nizle değiştirin, ör. 10.0.0.5)
 */
window.DASHBOARD_CONFIG = {
  // Tüm servislerin çalıştığı sunucunun IP'si (yerinde / on-prem).
  BOX_IP: "localhost",

  // --- Tokenizer görselleştirici (Gradio) — Bölüm 3 ---
  // Gradio uygulaması bu portta yayınlanır (app.py: server_port=7860).
  TOKENIZER_VIZ_URL: "http://localhost:7860",

  // --- Sohbet ürünü (Open WebUI) — Bölüm 5 ---
  // Open WebUI bu portta yayınlanır.
  CHAT_URL: "http://localhost:11436",
  // Ürün bölümünde Open WebUI'yi iframe içinde göstermeyi dene.
  // Bazı sunucular X-Frame-Options ile iframe'i engeller; engellenirse
  // panel otomatik olarak "Sohbeti Başlat" butonuna düşer.
  CHAT_EMBED: true,

  // --- Eğitim verisi (canlı kayıt) — Bölüm 4 ---
  // Eğitim işinin yazdığı JSONL dosyalarının taban URL'i.
  // Eğitim kutusu out_dir'i şu şekilde sunabilir:
  //   cd out/U32_seed0 && python -m http.server 8088
  // ardından TRAINING_DATA_BASE_URL = "http://localhost:8088"
  //
  // Varsayılan olarak panelle birlikte gelen örnek fixtür'leri okur,
  // böylece gerçek eğitim başlamadan önce de panel tam görünür.
  TRAINING_DATA_BASE_URL: "./sample_data",
  TRAIN_LOG_FILE: "train_log.jsonl",
  SAMPLES_FILE: "samples.jsonl",

  // Canlı kayıtları kaç saniyede bir yeniden çekelim (0 = kapalı).
  POLL_INTERVAL_SECONDS: 15,
};
