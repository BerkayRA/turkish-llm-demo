/*
 * Egemen Türkçe Yapay Zeka — Kontrol Paneli davranışı
 * ---------------------------------------------------
 * Tüm yapılandırma window.DASHBOARD_CONFIG (config.js) üzerinden okunur.
 * Sorumluluklar:
 *   - kaydırınca-belir (IntersectionObserver) + animasyonlu sayaçlar
 *   - bar dolgu animasyonları (dedup + tokenizer)
 *   - JSONL çekme/ayrıştırma → kayıp grafiği + örnek akışı
 *   - POLL_INTERVAL_SECONDS ile yeniden çekme (0 = kapalı)
 *   - tokenizer / sohbet URL'lerini config'ten ayarlama
 * Yalnızca compositor-dostu özellikler (opacity / transform) animasyonlanır.
 */
(function () {
  "use strict";

  // --- güvenli yapılandırma okuma + makul varsayılanlar --------------------
  var CFG = (window && window.DASHBOARD_CONFIG) || {};
  var TRAINING_BASE = CFG.TRAINING_DATA_BASE_URL || "./sample_data";
  var TRAIN_LOG_FILE = CFG.TRAIN_LOG_FILE || "train_log.jsonl";
  var SAMPLES_FILE = CFG.SAMPLES_FILE || "samples.jsonl";
  var POLL_SECONDS = typeof CFG.POLL_INTERVAL_SECONDS === "number" ? CFG.POLL_INTERVAL_SECONDS : 15;
  var TOKENIZER_VIZ_URL = CFG.TOKENIZER_VIZ_URL || "#";
  var CHAT_URL = CFG.CHAT_URL || "#";
  var CHAT_EMBED = CFG.CHAT_EMBED === true;

  var PREFERS_REDUCED_MOTION =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function joinUrl(base, file) {
    if (!base) return file;
    return base.replace(/\/+$/, "") + "/" + String(file).replace(/^\/+/, "");
  }

  function $(sel) {
    return document.querySelector(sel);
  }

  // =======================================================================
  // 1) Nav + rozet şeridi: kaydırmada görsel durum
  // =======================================================================
  function initScrollChrome() {
    var nav = $("#nav");
    var strip = $("#badgeStrip");
    function onScroll() {
      var y = window.scrollY || window.pageYOffset || 0;
      if (nav) nav.classList.toggle("scrolled", y > 12);
      // hero geçilince yüzen rozet şeridini sönükleştir (dikkat dağıtmasın)
      if (strip) strip.style.opacity = y > window.innerHeight * 0.7 ? "0" : "1";
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // =======================================================================
  // 2) Kaydırınca-belir + animasyonlu sayaçlar + bar dolguları
  // =======================================================================
  function formatInt(n) {
    // Türkçe binlik ayırıcı (nokta) — 53.691.924
    try {
      return Math.round(n).toLocaleString("tr-TR");
    } catch (e) {
      return String(Math.round(n));
    }
  }

  function animateCounter(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    if (PREFERS_REDUCED_MOTION) {
      el.textContent = formatInt(target);
      return;
    }
    var duration = 1500;
    var start = null;
    function ease(t) {
      // easeOutExpo — eşleştir styles.css --ease ruhuyla
      return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }
    function frame(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / duration, 1);
      el.textContent = formatInt(target * ease(p));
      if (p < 1) requestAnimationFrame(frame);
      else el.textContent = formatInt(target);
    }
    requestAnimationFrame(frame);
  }

  function fillDedup(el) {
    var pct = parseFloat(el.getAttribute("data-fill"));
    if (isNaN(pct)) return;
    // width animasyonu yalnızca bu küçük göstergeler için kullanılır (styles.css ön tanımı)
    el.style.width = pct + "%";
  }

  function fillFert(el) {
    var val = parseFloat(el.getAttribute("data-fert"));
    var max = parseFloat(el.getAttribute("data-max")) || 6;
    if (isNaN(val)) return;
    el.style.width = Math.min((val / max) * 100, 100) + "%";
  }

  function activateSection(node) {
    // sayaçlar
    var counters = node.querySelectorAll("[data-count]");
    for (var i = 0; i < counters.length; i++) animateCounter(counters[i]);
    // dedup bar
    var dedups = node.querySelectorAll(".dedup-fill[data-fill]");
    for (var j = 0; j < dedups.length; j++) fillDedup(dedups[j]);
    // tokenizer bar
    var ferts = node.querySelectorAll(".fert-bar[data-fert]");
    for (var k = 0; k < ferts.length; k++) fillFert(ferts[k]);
  }

  function initReveal() {
    var revealEls = document.querySelectorAll(".reveal");

    if (!("IntersectionObserver" in window) || PREFERS_REDUCED_MOTION) {
      // fallback: hepsini görünür yap + animasyonları anında bitir
      for (var i = 0; i < revealEls.length; i++) revealEls[i].classList.add("in");
      activateSection(document);
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("in");
          activateSection(entry.target);
          io.unobserve(entry.target);
        });
      },
      { threshold: 0.18, rootMargin: "0px 0px -8% 0px" }
    );

    for (var n = 0; n < revealEls.length; n++) io.observe(revealEls[n]);
  }

  // =======================================================================
  // 3) Yapılandırmadan URL'leri / sohbeti ayarla
  // =======================================================================
  function initLinks() {
    var tokBtn = $("#tokVizBtn");
    if (tokBtn) tokBtn.setAttribute("href", TOKENIZER_VIZ_URL);

    var chatBtn = $("#chatBtn");
    if (chatBtn) chatBtn.setAttribute("href", CHAT_URL);

    var urlLabel = $("#chatUrlLabel");
    if (urlLabel) urlLabel.textContent = CHAT_URL.replace(/^https?:\/\//, "") + " · yerinde";

    var frame = $("#chatFrame");
    var fallback = $("#chatFallback");

    if (CHAT_EMBED && frame) {
      // iframe'i göstermeyi dene; X-Frame-Options engellerse fallback'e düş
      var settled = false;
      function showFallback() {
        if (settled) return;
        settled = true;
        frame.hidden = true;
        if (fallback) fallback.style.display = "";
      }
      function showFrame() {
        if (settled) return;
        settled = true;
        if (fallback) fallback.style.display = "none";
        frame.hidden = false;
      }

      frame.addEventListener("load", showFrame);
      frame.addEventListener("error", showFallback);
      // yüklenmezse (engelleme çoğu zaman sessizdir) fallback'e düş
      window.setTimeout(function () {
        // load tetiklenmediyse hâlâ gömülü değildir
        if (!settled) showFallback();
      }, 4000);

      frame.setAttribute("src", CHAT_URL);
    } else {
      // gömme kapalı: yalnızca buton göster
      if (frame) frame.hidden = true;
      if (fallback) fallback.style.display = "";
    }
  }

  // =======================================================================
  // 4) JSONL çekme + ayrıştırma (satır-satır JSON.parse)
  // =======================================================================
  function parseJsonl(text) {
    var out = [];
    var lines = text.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) continue;
      try {
        out.push(JSON.parse(line));
      } catch (e) {
        // bozuk / yarım yazılmış satırı atla (canlı yazma sırasında olabilir)
      }
    }
    return out;
  }

  function fetchJsonl(url) {
    return fetch(url, { cache: "no-store" }).then(function (res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.text();
    }).then(parseJsonl);
  }

  // =======================================================================
  // 5) Kayıp grafiği (Chart.js)
  // =======================================================================
  var lossChart = null;

  function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fallback;
  }

  function buildLossChart() {
    var canvas = $("#lossChart");
    if (!canvas || typeof window.Chart === "undefined") return;
    var ctx = canvas.getContext("2d");

    var crimson = cssVar("--crimson-bright", "oklch(64% 0.23 24)");
    var gold = cssVar("--gold", "oklch(80% 0.12 85)");
    var paperFaint = "rgba(220, 218, 210, 0.5)";
    var grid = "rgba(150, 150, 170, 0.10)";

    lossChart = new window.Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Eğitim kaybı",
            data: [],
            borderColor: crimson,
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            parsing: { xAxisKey: "step", yAxisKey: "loss" },
          },
          {
            label: "Doğrulama kaybı",
            data: [],
            borderColor: gold,
            backgroundColor: "transparent",
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 3,
            pointBackgroundColor: gold,
            tension: 0.3,
            spanGaps: true,
            parsing: { xAxisKey: "step", yAxisKey: "val_loss" },
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: PREFERS_REDUCED_MOTION ? false : { duration: 600 },
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "Adım", color: paperFaint, font: { size: 11 } },
            ticks: {
              color: paperFaint,
              font: { size: 10 },
              callback: function (v) {
                return v >= 1000 ? v / 1000 + "k" : v;
              },
            },
            grid: { color: grid },
          },
          y: {
            title: { display: true, text: "Kayıp", color: paperFaint, font: { size: 11 } },
            ticks: { color: paperFaint, font: { size: 10 } },
            grid: { color: grid },
          },
        },
        plugins: {
          legend: {
            labels: { color: paperFaint, usePointStyle: true, boxWidth: 8, font: { size: 11 } },
          },
          tooltip: {
            backgroundColor: "rgba(20, 22, 32, 0.95)",
            titleColor: "#eee",
            bodyColor: "#ddd",
            borderColor: "rgba(150,150,170,0.2)",
            borderWidth: 1,
            callbacks: {
              title: function (items) {
                return "Adım " + (items[0] ? items[0].parsed.x : "");
              },
            },
          },
        },
      },
    });
  }

  function updateLossChart(rows) {
    if (!lossChart || !rows || !rows.length) return;

    // Eğitim checkpoint'ten yeniden başlatıldığında adım sayacı geriye sarar; log
    // birden fazla koşunun satırlarını içerebilir. (a) adıma göre tekilleştir — dosyada
    // sonra gelen satır (en güncel koşu) kazanır; (b) canlı imleci aşan eski koşu
    // kalıntılarını ("hayalet" noktalar) gizle. Sonuç: tek-yönlü, monoton grafik.
    var liveStep = rows[rows.length - 1].step;
    var byStep = {};
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (typeof r.step !== "number") continue;
      if (typeof liveStep === "number" && r.step > liveStep) continue;
      byStep[r.step] = r;
    }
    var ordered = Object.keys(byStep)
      .map(Number)
      .sort(function (a, b) {
        return a - b;
      })
      .map(function (s) {
        return byStep[s];
      });

    var trainPoints = [];
    var valPoints = [];
    for (var k = 0; k < ordered.length; k++) {
      var p = ordered[k];
      if (typeof p.loss === "number") trainPoints.push({ step: p.step, loss: p.loss });
      if (typeof p.val_loss === "number") valPoints.push({ step: p.step, val_loss: p.val_loss });
    }

    lossChart.data.datasets[0].data = trainPoints;
    lossChart.data.datasets[1].data = valPoints;
    lossChart.update();

    // meta panelini son satırdan güncelle
    var last = rows[rows.length - 1];
    var lastVal = null;
    for (var v = rows.length - 1; v >= 0; v--) {
      if (typeof rows[v].val_loss === "number") {
        lastVal = rows[v].val_loss;
        break;
      }
    }
    setText("#metaStep", last && typeof last.step === "number" ? formatInt(last.step) : "—");
    setText("#metaLoss", last && typeof last.loss === "number" ? last.loss.toFixed(3) : "—");
    setText("#metaVal", lastVal !== null ? lastVal.toFixed(3) : "—");
    setText(
      "#metaTokens",
      last && typeof last.tokens_seen === "number" ? formatTokens(last.tokens_seen) : "—"
    );
  }

  function formatTokens(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + " Mr";
    if (n >= 1e6) return (n / 1e6).toFixed(0) + " M";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + " B";
    return String(n);
  }

  function setText(sel, txt) {
    var el = $(sel);
    if (el) el.textContent = txt;
  }

  // =======================================================================
  // 6) Örnek akışı (modelin öğrenişi)
  // =======================================================================
  function renderSamples(rows) {
    var stream = $("#learnStream");
    if (!stream || !rows || !rows.length) return;

    var html = "";
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (typeof r.sample !== "string") continue;
      var isLatest = i === rows.length - 1;
      html +=
        '<div class="learn-item' + (isLatest ? " latest" : "") + '">' +
        '<span class="step">Adım ' + formatInt(r.step || 0) + "</span>" +
        '<p class="txt">' + escapeHtml(r.sample) + "</p>" +
        "</div>";
    }
    if (html) stream.innerHTML = html;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // =======================================================================
  // 7) Canlı/duraksamış durum göstergeleri + yoklama döngüsü
  // =======================================================================
  function setStatus(dotSel, labelSel, state, text) {
    var dot = $(dotSel);
    var label = $(labelSel);
    if (dot) dot.classList.toggle("stale", state !== "live");
    if (label) label.textContent = text;
  }

  function pollOnce() {
    var lossUrl = joinUrl(TRAINING_BASE, TRAIN_LOG_FILE);
    var samplesUrl = joinUrl(TRAINING_BASE, SAMPLES_FILE);

    fetchJsonl(lossUrl)
      .then(function (rows) {
        if (rows.length) {
          updateLossChart(rows);
          setStatus("#lossDot", "#lossStatus", "live", "canlı · " + rows.length + " kayıt");
        } else {
          setStatus("#lossDot", "#lossStatus", "stale", "veri bekleniyor");
        }
      })
      .catch(function () {
        setStatus("#lossDot", "#lossStatus", "stale", "veri bekleniyor");
      });

    fetchJsonl(samplesUrl)
      .then(function (rows) {
        if (rows.length) {
          var empty = $("#learnEmpty");
          if (empty && empty.parentNode) empty.parentNode.removeChild(empty);
          renderSamples(rows);
          setStatus("#sampleDot", "#sampleStatus", "live", "canlı");
        } else {
          setStatus("#sampleDot", "#sampleStatus", "stale", "veri bekleniyor");
        }
      })
      .catch(function () {
        setStatus("#sampleDot", "#sampleStatus", "stale", "veri bekleniyor");
      });
  }

  function initTrainingFeed() {
    buildLossChart();
    pollOnce();
    if (POLL_SECONDS && POLL_SECONDS > 0) {
      window.setInterval(pollOnce, POLL_SECONDS * 1000);
    }
  }

  // =======================================================================
  // başlat
  // =======================================================================
  function init() {
    initScrollChrome();
    initReveal();
    initLinks();
    initTrainingFeed();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
