"""Load a checkpoint + SentencePiece model and stream a generation to stdout.

Usage:
  python generate.py --ckpt out/ckpt_latest.pt --sp_model sp_unigram_32000.model \
      --prompt "Türkiye'nin başkenti" --max_new_tokens 120 --temperature 0.8 --top_p 0.95
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from model import ModelConfig, TurkishLM, _top_p_filter

BOS_ID, EOS_ID = 1, 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Turkish text from a checkpoint.")
    p.add_argument("--ckpt", required=True, help="path to a training checkpoint .pt")
    p.add_argument("--sp_model", required=True, help="SentencePiece .model file")
    p.add_argument("--prompt", default="", help="text prompt")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> TurkishLM:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = TurkishLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded step={ckpt.get('step')} "
          f"params={model.num_params():,} block_size={cfg.block_size}",
          file=sys.stderr)
    return model


@torch.no_grad()
def stream_generate(model: TurkishLM, sp, idx: torch.Tensor, *,
                    max_new_tokens: int, temperature: float, top_p: float) -> None:
    """Sample one token at a time, decoding incrementally so output streams."""
    block_size = model.cfg.block_size
    generated: list[int] = []
    prev_text = ""
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if 0.0 < top_p < 1.0:
                logits = _top_p_filter(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        tok = int(next_id)
        if tok == EOS_ID:
            break
        generated.append(tok)
        idx = torch.cat((idx, next_id), dim=1)

        # Decode the full continuation and print only the newly revealed suffix.
        # (SentencePiece detokenization is non-local, so re-decode each step.)
        text = sp.decode(generated)
        sys.stdout.write(text[len(prev_text):])
        sys.stdout.flush()
        prev_text = text
    sys.stdout.write("\n")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)

    model = load_model(args.ckpt, device)

    ids = [BOS_ID] + (sp.encode(args.prompt, out_type=int) if args.prompt else [])
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    sys.stdout.write(args.prompt)
    sys.stdout.flush()
    stream_generate(model, sp, idx,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p)


if __name__ == "__main__":
    main()
