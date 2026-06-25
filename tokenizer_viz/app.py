"""Türkçe Tokenizer Görselleştirici — yönetici sunumu için.

Üç tokenizer'ı (BİZİM SentencePiece unigram-32k, GPT-4o o200k_base, Llama-3 /
çevrimdışı yoksa tiktoken cl100k_base vekili) karşılaştırır ve bizim
tokenizer'ımızın daha az, daha temiz token ürettiğini ("fertility" avantajı)
gösterir. Fertility = token / kelime; metrik fikri turkish-corpus/.../fertility.py.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass

import gradio as gr

# --------------------------------------------------------------------------- #
# Yapılandırma
# --------------------------------------------------------------------------- #

# Yerel kopya varsayılan; dağıtım kutusunda: /opt/corpus/out/tokenizer/sp_unigram_32000.model
DEFAULT_SP_MODEL = (
    "/Users/berkayra/Downloads/tokenizer/turkish-llm/"
    "models_fresh/sp_unigram_32000.model"
)
OUR_SP_MODEL = os.environ.get("OUR_SP_MODEL", DEFAULT_SP_MODEL)

LLAMA3_MODEL_ID = os.environ.get("LLAMA3_MODEL_ID", "meta-llama/Meta-Llama-3-8B")

# Token chip'leri için alternatif arka plan renkleri (kenarlar net görünsün).
CHIP_COLORS = ("#dbeafe", "#bfdbfe")  # açık mavi tonları
OURS_CHIP_COLORS = ("#dcfce7", "#bbf7d0")  # bizimkine yeşil vurgu

# Örnek Türkçe paragraflar (haber, resmi/hukuki, sohbet, eklemeli-ağır).
EXAMPLES: dict[str, str] = {
    "Haber": (
        "Cumhurbaşkanı bugün düzenlenen basın toplantısında ekonomik reform "
        "paketinin ayrıntılarını açıkladı. Yetkililer, enflasyonla mücadelede "
        "yeni adımların atılacağını ve istihdamın artırılacağını belirtti."
    ),
    "Resmî / Hukukî": (
        "İşbu sözleşmenin ihlali hâlinde, taraflardan biri diğerine yazılı "
        "bildirimde bulunmak suretiyle sözleşmeyi feshedebilir. Anlaşmazlıkların "
        "çözümünde Ankara mahkemeleri ve icra daireleri yetkili kılınmıştır."
    ),
    "Sohbet": (
        "Yarın akşam bizimkilerle buluşup bir şeyler yiyeceğiz, sen de gelsene! "
        "Geçen seferki yerden çok memnun kalmıştık, yemekleri gerçekten harikaydı. "
        "Saat sekiz gibi orada oluruz herhâlde."
    ),
    "Eklemeli-ağır": (
        "Evlerinizden çıkarılamayanlardan, çalıştırılamayanlardan ve "
        "Avrupalılaştıramadıklarımızdan bahsediyoruz. Gözlüklerimi "
        "bulamadığımdan dolayı toplantıya katılamayabileceğimi düşünüyorum."
    ),
}
DEFAULT_EXAMPLE_TEXT = EXAMPLES["Eklemeli-ağır"]


# --------------------------------------------------------------------------- #
# Tokenizer sarmalayıcıları
# --------------------------------------------------------------------------- #


@dataclass
class TokResult:
    """Bir tokenizer'ın tek metin için sonucu."""

    name: str
    available: bool
    tokens: list[str]
    note: str = ""

    @property
    def count(self) -> int:
        return len(self.tokens)


def _word_count(text: str) -> int:
    return len(text.split())


def fertility(token_count: int, word_count: int) -> float:
    """Fertility = token / kelime. Kelime yoksa 0.0 (ZeroDivision'dan kaçın)."""
    return token_count / word_count if word_count else 0.0


def _clean_piece(piece: str) -> str:
    """SentencePiece/BPE boşluk işaretlerini görünür boşluğa çevir."""
    # SentencePiece: U+2581 (▁); GPT/BPE byte-level: U+0120 (Ġ)
    return piece.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")


class OurTokenizer:
    """SentencePiece unigram — bizim tokenizer'ımız."""

    def __init__(self, model_path: str) -> None:
        self.name = "BİZİM (SentencePiece 32k)"
        self.available = False
        self.note = ""
        self._sp = None
        try:
            import sentencepiece as spm

            if not os.path.exists(model_path):
                self.note = f"Model bulunamadı: {model_path}"
                return
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(model_path)
            self.available = True
        except ImportError:
            self.note = "sentencepiece paketi yüklü değil."
        except Exception as exc:  # pragma: no cover - savunmacı
            self.note = f"Yüklenemedi: {exc}"

    def tokenize(self, text: str) -> TokResult:
        if not self.available or self._sp is None:
            return TokResult(self.name, False, [], self.note)
        pieces = self._sp.EncodeAsPieces(text)
        return TokResult(self.name, True, [_clean_piece(p) for p in pieces])


class TiktokenTokenizer:
    """tiktoken tabanlı tokenizer (GPT-4o veya cl100k vekili)."""

    def __init__(self, name: str, encoding: str) -> None:
        self.name = name
        self.encoding_name = encoding
        self.available = False
        self.note = ""
        self._enc = None
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding(encoding)
            self.available = True
        except ImportError:
            self.note = "tiktoken paketi yüklü değil."
        except Exception as exc:  # pragma: no cover
            self.note = f"Kodlama yüklenemedi ({encoding}): {exc}"

    def tokenize(self, text: str) -> TokResult:
        if not self.available or self._enc is None:
            return TokResult(self.name, False, [], self.note)
        ids = self._enc.encode(text)
        toks = [_clean_piece(self._enc.decode([i])) for i in ids]
        return TokResult(self.name, True, toks, self.note)


class Llama3Tokenizer:
    """Llama-3 tokenizer; çevrimdışı yoksa tiktoken cl100k_base vekiline düşer."""

    def __init__(self, model_id: str) -> None:
        self.name = "Llama-3"
        self.available = False
        self.note = ""
        self._hf = None
        self._fallback: TiktokenTokenizer | None = None

        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from transformers import AutoTokenizer

            self._hf = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            self.available = True
            return
        except Exception:
            # Çevrimdışı yok / yetki gerekiyor → vekile düş.
            pass

        self._fallback = TiktokenTokenizer("Llama-3", "cl100k_base")
        if self._fallback.available:
            self.name = "Llama-3 → GPT-3.5/4 vekili (cl100k_base)"
            self.available = True
            self.note = (
                "Llama-3 çevrimdışı bulunamadı; benzer ölçekli "
                "tiktoken cl100k_base vekil olarak gösteriliyor."
            )
        else:
            self.note = "Llama-3 ve tiktoken vekili kullanılamıyor."

    def tokenize(self, text: str) -> TokResult:
        if self._hf is not None:
            toks = self._hf.tokenize(text)
            return TokResult(self.name, True, [_clean_piece(t) for t in toks])
        if self._fallback is not None and self._fallback.available:
            res = self._fallback.tokenize(text)
            return TokResult(self.name, res.available, res.tokens, self.note)
        return TokResult(self.name, False, [], self.note)


# Tokenizer'ları bir kez yükle (ağır yüklemeleri tekrarlama).
OUR_TOK = OurTokenizer(OUR_SP_MODEL)
GPT4O_TOK = TiktokenTokenizer("GPT-4o (o200k_base)", "o200k_base")
LLAMA_TOK = Llama3Tokenizer(LLAMA3_MODEL_ID)
ALL_TOKENIZERS = (OUR_TOK, GPT4O_TOK, LLAMA_TOK)


# --------------------------------------------------------------------------- #
# HTML render
# --------------------------------------------------------------------------- #


def render_chips(res: TokResult, ours: bool) -> str:
    """Token'ları renkli chip'ler hâlinde HTML olarak göster."""
    if not res.available:
        return (
            f'<div class="tok-error">⚠️ {html.escape(res.name)} '
            f"kullanılamıyor: {html.escape(res.note)}</div>"
        )
    colors = OURS_CHIP_COLORS if ours else CHIP_COLORS
    chips = []
    for i, tok in enumerate(res.tokens):
        bg = colors[i % 2]
        shown = html.escape(tok).replace(" ", "&middot;").replace("\n", "↵")
        if not shown:
            shown = "∅"
        chips.append(
            f'<span class="chip" style="background:{bg}">{shown}</span>'
        )
    return f'<div class="chip-row">{"".join(chips)}</div>'


def _card(res: TokResult, word_count: int, ours: bool, best: bool) -> str:
    fert = fertility(res.count, word_count)
    cls = "card ours" if ours else "card"
    if best:
        cls += " best"
    badge = '<span class="badge">EN VERİMLİ</span>' if best else ""
    count_txt = res.count if res.available else "—"
    fert_txt = f"{fert:.2f}" if res.available else "—"
    return f"""
    <div class="{cls}">
      <div class="card-head">
        <span class="card-title">{html.escape(res.name)}</span>{badge}
      </div>
      <div class="metrics">
        <div class="metric"><span class="num">{count_txt}</span>
          <span class="lbl">token</span></div>
        <div class="metric"><span class="num">{fert_txt}</span>
          <span class="lbl">token / kelime</span></div>
      </div>
      {render_chips(res, ours)}
    </div>
    """


def _summary_bars(results: list[TokResult], word_count: int) -> str:
    """Üç fertility'yi karşılaştıran küçük bar grafik + özet cümle."""
    rows = []
    avail = [(r, fertility(r.count, word_count)) for r in results if r.available]
    if not avail:
        return '<div class="tok-error">Karşılaştırılacak tokenizer yok.</div>'

    max_fert = max(f for _, f in avail) or 1.0
    our = next((r for r in results if r is OUR_TOK and r.available), None)
    best = min(avail, key=lambda rf: rf[1])[0]

    for res, fert in [(r, fertility(r.count, word_count)) for r in results]:
        if not res.available:
            continue
        pct = (fert / max_fert) * 100
        ours = res is OUR_TOK
        bar_cls = "bar ours" if ours else ("bar best" if res is best else "bar")
        rows.append(
            f"""
        <div class="bar-row">
          <div class="bar-name">{html.escape(res.name)}</div>
          <div class="bar-track">
            <div class="{bar_cls}" style="width:{pct:.1f}%">
              <span class="bar-val">{fert:.2f}</span>
            </div>
          </div>
        </div>"""
        )

    headline = ""
    if our is not None:
        our_f = fertility(our.count, word_count)
        others = [fertility(r.count, word_count) for r in results
                  if r.available and r is not OUR_TOK]
        if others:
            worst = max(others)
            if worst > 0 and our_f < worst:
                saving = (1 - our_f / worst) * 100
                headline = (
                    f'<div class="headline">Bizim tokenizer: '
                    f"<strong>%{saving:.0f} daha az token</strong> "
                    f"(en verimsiz alternatife kıyasla)</div>"
                )
            elif our is best:
                headline = (
                    '<div class="headline">Bizim tokenizer en düşük '
                    "fertility'ye sahip — en verimli.</div>"
                )

    return f"""
    <div class="summary">
      <div class="summary-title">Fertility Karşılaştırması (düşük = daha iyi)</div>
      {headline}
      <div class="bars">{"".join(rows)}</div>
    </div>
    """


def compare(text: str) -> tuple[str, str]:
    """Üç tokenizer'ı çalıştır; (kartlar_html, özet_html) döndür."""
    text = (text or "").strip()
    word_count = _word_count(text)
    if not text:
        empty = '<div class="tok-error">Lütfen Türkçe bir metin girin.</div>'
        return empty, ""

    results = [tok.tokenize(text) for tok in ALL_TOKENIZERS]
    avail = [(r, fertility(r.count, word_count)) for r in results if r.available]
    best = min(avail, key=lambda rf: rf[1])[0] if avail else None

    # ours bayrağı: sadece BİZİM tokenizer; best = en düşük fertility.
    cards = "".join(
        _card(r, word_count, ours=(r is OUR_TOK), best=(r is best))
        for r in results
    )
    summary = _summary_bars(results, word_count)
    info = f'<div class="wordinfo">Kelime sayısı: <strong>{word_count}</strong></div>'
    return info + cards, summary


def load_example(name: str) -> str:
    return EXAMPLES.get(name, DEFAULT_EXAMPLE_TEXT)


# --------------------------------------------------------------------------- #
# Stil + arayüz
# --------------------------------------------------------------------------- #

CSS = """
:root { --accent:#16a34a; --ink:#0f172a; }
.gradio-container { max-width: 1100px !important; }
#title { text-align:center; }
#title h1 { font-size:2rem; margin-bottom:.2rem; color:var(--ink); }
#title p { color:#475569; margin-top:0; }
.wordinfo { margin:.4rem 0 1rem; color:#475569; font-size:.95rem; }
.card { border:1px solid #e2e8f0; border-radius:14px; padding:16px 18px; margin-bottom:14px; background:#fff; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.card.ours { border-color:#86efac; background:#f7fef9; }
.card.best { box-shadow:0 4px 18px rgba(22,163,74,.18); }
.card-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.card-title { font-weight:700; font-size:1.05rem; color:var(--ink); }
.badge { background:var(--accent); color:#fff; font-size:.7rem; font-weight:700; padding:3px 9px; border-radius:999px; letter-spacing:.04em; }
.metrics { display:flex; gap:28px; margin-bottom:12px; }
.metric { display:flex; flex-direction:column; }
.metric .num { font-size:1.7rem; font-weight:800; color:var(--ink); line-height:1; }
.metric .lbl { font-size:.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em; margin-top:3px; }
.chip-row { display:flex; flex-wrap:wrap; gap:3px; line-height:1.9; }
.chip { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.86rem; padding:1px 5px; border-radius:5px; color:#1e293b; white-space:pre-wrap; }
.tok-error { color:#b45309; background:#fffbeb; border:1px solid #fde68a; padding:10px 14px; border-radius:10px; }
.summary { border:1px solid #e2e8f0; border-radius:14px; padding:18px 20px; background:#fff; }
.summary-title { font-weight:700; color:var(--ink); margin-bottom:6px; }
.headline { background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; padding:10px 14px; border-radius:10px; margin:8px 0 14px; font-size:1.02rem; }
.bar-row { display:flex; align-items:center; gap:12px; margin:8px 0; }
.bar-name { width:230px; font-size:.9rem; color:#334155; text-align:right; }
.bar-track { flex:1; background:#f1f5f9; border-radius:8px; overflow:hidden; }
.bar { height:30px; background:#94a3b8; display:flex; align-items:center; justify-content:flex-end; transition:width .4s ease; }
.bar.best, .bar.ours { background:var(--accent); }
.bar-val { color:#fff; font-weight:700; font-size:.85rem; padding-right:10px; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="Türkçe Tokenizer Görselleştirici",
                   theme=gr.themes.Soft()) as demo:
        gr.HTML(
            '<div id="title"><h1>Türkçe Tokenizer Görselleştirici</h1>'
            "<p>Tokenizer'ımızın Türkçe metinde sağladığı verimlilik avantajı "
            "(daha az, daha temiz token)</p></div>"
        )
        with gr.Row():
            with gr.Column(scale=3):
                txt = gr.Textbox(
                    label="Türkçe Metin",
                    value=DEFAULT_EXAMPLE_TEXT,
                    lines=5,
                    placeholder="Türkçe bir metin yazın veya örnek seçin...",
                )
            with gr.Column(scale=1):
                ex = gr.Radio(
                    choices=list(EXAMPLES.keys()),
                    value="Eklemeli-ağır",
                    label="Örnek Paragraflar",
                )
                btn = gr.Button("Karşılaştır", variant="primary")

        summary_out = gr.HTML()
        cards_out = gr.HTML()

        ex.change(load_example, inputs=ex, outputs=txt)
        btn.click(compare, inputs=txt, outputs=[cards_out, summary_out])
        txt.submit(compare, inputs=txt, outputs=[cards_out, summary_out])
        demo.load(compare, inputs=txt, outputs=[cards_out, summary_out])
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
