#!/usr/bin/env python3
"""Format a Turkish instruction dataset into chat messages for QLoRA SFT.

This script converts a Turkish instruction/response dataset into the
"messages" chat format expected by trl's SFTTrainer, rendering each
example with the base model's ``tokenizer.apply_chat_template``.

Default dataset
---------------
``--dataset`` defaults to ``merve/turkish_instructions`` (a Turkish
Alpaca-style instruction set on the Hugging Face Hub). It exposes
``talimat`` (instruction), ``giris`` (optional input) and ``cikti``
(response) columns. The loader also understands common Alpaca/OASST-tr
column naming so you can point it at other Turkish SFT sets.

Licensing note
--------------
Always verify the license of whichever HF dataset you pass via
``--dataset`` before any commercial use. Many Turkish Alpaca-style sets
are derived from outputs of proprietary models (e.g. GPT-3.5/4) and are
therefore *research-only* under the source model's terms, even when the
HF card lists a permissive license such as Apache-2.0/CC-BY. For a
fully unencumbered demo, use the built-in ``--offline`` fallback below,
which is original hand-written content released here under CC0.

Run examples
------------
Online (downloads dataset + tokenizer)::

    python prepare_data.py --base_model Qwen/Qwen2.5-3B-Instruct \
        --dataset merve/turkish_instructions --out data/train.jsonl

Fully offline (no network, uses built-in 20-pair fallback)::

    python prepare_data.py --offline --base_model Qwen/Qwen2.5-3B-Instruct \
        --out data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

# Turkish on-prem assistant system prompt. Kept identical across the
# pipeline (finetune + Modelfile) so train/serve behaviour matches.
SYSTEM_PROMPT: str = (
    "Sen Türkçe konuşan, yardımsever ve güvenilir bir kurum içi yapay zeka "
    "asistanısın. Soruları açık, doğru ve nazik bir dille yanıtla. "
    "Emin olmadığın konularda tahmin yürütmek yerine bilmediğini belirt."
)

# Candidate column names across common Turkish SFT dataset schemas.
INSTRUCTION_KEYS: tuple[str, ...] = ("talimat", "instruction", "soru", "prompt", "input_text")
INPUT_KEYS: tuple[str, ...] = ("giris", "input", "context", "baglam")
RESPONSE_KEYS: tuple[str, ...] = ("cikti", "output", "response", "cevap", "yanit", "completion")

# Built-in hand-written Turkish instruction/response pairs (CC0).
# Lets the whole pipeline run with zero network access.
OFFLINE_PAIRS: tuple[tuple[str, str], ...] = (
    ("Merhaba, bugün bana nasıl yardımcı olabilirsin?",
     "Merhaba! Sorularını yanıtlayabilir, metin yazmana yardım edebilir, "
     "bir konuyu özetleyebilir veya kod yazmana destek olabilirim. Nereden başlayalım?"),
    ("Türkiye'nin başkenti neresidir?",
     "Türkiye'nin başkenti Ankara'dır. İstanbul ise en kalabalık şehridir."),
    ("Fotosentezi basitçe açıkla.",
     "Fotosentez, bitkilerin güneş ışığını kullanarak su ve karbondioksiti "
     "şekere (besine) ve oksijene dönüştürdüğü süreçtir. Bu sayede hem kendi "
     "besinlerini üretirler hem de havaya oksijen verirler."),
    ("Bana kısa bir motivasyon cümlesi söyle.",
     "Bugün attığın küçük bir adım, yarın ulaşacağın büyük hedefin temelidir. "
     "Devam et!"),
    ("Python'da bir listeyi nasıl tersine çeviririm?",
     "Bir Python listesini tersine çevirmenin pratik yolu dilimlemedir: "
     "`liste[::-1]` yeni bir ters liste döndürür. Listeyi yerinde değiştirmek "
     "istersen `liste.reverse()` kullanabilirsin."),
    ("5 ile 12'nin toplamı kaçtır?",
     "5 ile 12'nin toplamı 17'dir."),
    ("Sağlıklı beslenme için üç öneri ver.",
     "1) Günde bol su iç ve şekerli içecekleri azalt. "
     "2) Öğünlerine sebze ve tam tahıl ekle. "
     "3) İşlenmiş gıdalar yerine evde hazırlanmış taze yemekleri tercih et."),
    ("İstanbul'da yağmurlu bir günde ne yapabilirim?",
     "Yağmurlu bir İstanbul gününde müzeleri gezebilir (örneğin İstanbul "
     "Arkeoloji Müzeleri), kapalı çarşıda dolaşabilir, sıcak bir kafede kitap "
     "okuyabilir ya da bir hamamda dinlenebilirsin."),
    ("E-posta yazarken nazik bir kapanış cümlesi öner.",
     "\"Yardımcı olabileceğim başka bir konu olursa memnuniyetle "
     "destek olurum. İyi çalışmalar dilerim.\" şeklinde nazik bir kapanış "
     "kullanabilirsin."),
    ("Yapay zeka nedir, kısaca anlat.",
     "Yapay zeka, bilgisayar sistemlerinin öğrenme, akıl yürütme ve problem "
     "çözme gibi normalde insan zekası gerektiren görevleri yerine "
     "getirebilmesini sağlayan teknolojidir."),
    ("Bir toplantı için gündem maddeleri nasıl hazırlanır?",
     "Önce toplantının amacını netleştir. Ardından konuları öncelik sırasına "
     "diz, her maddeye yaklaşık bir süre ayır, sorumlu kişileri belirle ve en "
     "sona kararlar ile sonraki adımlar için bir bölüm ekle."),
    ("Suyun kaynama noktası kaç derecedir?",
     "Deniz seviyesinde ve normal atmosfer basıncında su 100 santigrat "
     "derecede kaynar. Yükseklere çıkıldıkça bu sıcaklık düşer."),
    ("Bana kısa bir teşekkür mesajı yaz.",
     "Desteğin ve yardımın için içtenlikle teşekkür ederim. Katkın benim için "
     "gerçekten çok değerliydi."),
    ("Verimli çalışmak için bir teknik öner.",
     "Pomodoro tekniğini deneyebilirsin: 25 dakika kesintisiz çalış, ardından "
     "5 dakika ara ver. Dört turun sonunda 15-20 dakikalık daha uzun bir mola "
     "yap. Bu yöntem odaklanmayı kolaylaştırır."),
    ("İklim değişikliğinin bir nedenini söyle.",
     "Başlıca nedenlerden biri, fosil yakıtların (kömür, petrol, doğal gaz) "
     "yakılmasıyla atmosfere salınan sera gazlarının, özellikle "
     "karbondioksitin artmasıdır."),
    ("Kısa bir özgeçmiş özeti nasıl yazılır?",
     "Özetini iki üç cümlede tut: kim olduğunu, kaç yıllık deneyimin "
     "olduğunu, hangi alanda uzmanlaştığını ve aradığın rolü belirt. "
     "Somut bir başarına da kısaca değinmen etkili olur."),
    ("Bilgisayarım yavaş çalışıyor, ne yapabilirim?",
     "Gereksiz programları kapat, başlangıçta otomatik açılan uygulamaları "
     "azalt, disk alanını boşalt ve güncellemeleri yükle. Sorun sürerse "
     "RAM yükseltmeyi veya disk yerine SSD kullanmayı düşünebilirsin."),
    ("Bana iki dilli (Türkçe-İngilizce) selamlama örneği ver.",
     "Türkçe: \"Merhaba, hoş geldiniz!\" — İngilizce: \"Hello, welcome!\""),
    ("Bir hikayeye başlamak için ilk cümle yaz.",
     "Sabahın ilk ışıkları perdeden süzülürken, Elif uzun zamandır beklediği "
     "o mektubun nihayet kapısının önünde olduğunu fark etmemişti bile."),
    ("Düzenli uyku neden önemlidir?",
     "Düzenli uyku; hafızanın pekişmesine, bağışıklık sisteminin "
     "güçlenmesine, ruh halinin dengelenmesine ve gün boyu odaklanmaya yardımcı "
     "olur. Yetişkinler için genellikle 7-9 saat önerilir."),
)


def _first_present(row: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    """Return the first non-empty string value for any of ``keys``."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def row_to_messages(row: dict[str, Any]) -> Optional[list[dict[str, str]]]:
    """Convert a raw dataset row into a chat ``messages`` list.

    Returns ``None`` if the row lacks a usable instruction/response pair.
    """
    instruction = _first_present(row, INSTRUCTION_KEYS)
    response = _first_present(row, RESPONSE_KEYS)
    if not instruction or not response:
        return None

    extra_input = _first_present(row, INPUT_KEYS)
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]


def _offline_rows() -> list[dict[str, str]]:
    """Build dataset rows from the built-in hand-written Turkish pairs."""
    return [{"talimat": q, "cikti": a} for q, a in OFFLINE_PAIRS]


def load_rows(dataset_id: str, split: str, offline: bool) -> list[dict[str, Any]]:
    """Load raw rows either from the HF Hub or the offline fallback."""
    if offline:
        return _offline_rows()

    # Imported lazily so --offline works without datasets installed.
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset(dataset_id, split=split)
    return [dict(row) for row in dataset]


def render_example(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> str:
    """Render messages to a single training string via the chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Produce the list of serialisable training records."""
    rows = load_rows(args.dataset, args.split, args.offline)

    tokenizer = None
    if not args.no_template:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    records: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        messages = row_to_messages(row)
        if messages is None:
            skipped += 1
            continue
        record: dict[str, Any] = {"messages": messages}
        if tokenizer is not None:
            record["text"] = render_example(messages, tokenizer)
        records.append(record)

    print(f"Loaded {len(rows)} rows, kept {len(records)}, skipped {skipped}.")
    return records


def write_jsonl(records: list[dict[str, Any]], out_path: Path) -> None:
    """Write records to a UTF-8 JSONL file (Turkish chars preserved)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} examples to {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF id of the base chat model (for its chat template).",
    )
    parser.add_argument(
        "--dataset",
        default="merve/turkish_instructions",
        help="HF dataset id of a Turkish SFT set.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument(
        "--out",
        default="data/train.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the built-in 20-pair Turkish fallback (no network).",
    )
    parser.add_argument(
        "--no_template",
        action="store_true",
        help="Skip rendering 'text'; only emit raw 'messages'.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    records = build_dataset(args)
    if not records:
        raise SystemExit("No usable examples produced; check dataset columns.")
    write_jsonl(records, Path(args.out))


if __name__ == "__main__":
    main()
