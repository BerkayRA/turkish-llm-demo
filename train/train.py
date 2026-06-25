"""nanoGPT-style training loop for the Turkish LM (single-GPU, 8GB friendly).

Targets an NVIDIA Quadro RTX 4000 (8GB, Turing sm_75). Fits via a small
per-step micro-batch + gradient accumulation, bf16/fp16 AMP, and optional
gradient checkpointing.

Hyperparameters follow docs/EXPERIMENT_AB.md §11: AdamW beta=(0.9,0.95),
wd 0.1 (norms/embeddings excluded), peak LR 6e-4 -> 6e-5 cosine, ~1% warmup,
grad clip 1.0, dropout 0.0, GPT-2 init.

Logs one JSON object per line to <out_dir>/train_log.jsonl and a generated
Turkish sample per eval to <out_dir>/samples.jsonl.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time

import numpy as np
import torch

from data import BatchLoader, EOS_ID
from model import TurkishLM, configs

# Checkpoint milestones requested by the spec, plus every --ckpt_every steps.
MILESTONE_STEPS = (0, 100, 1000, 10000)
DEFAULT_PROMPT = "Türkiye'nin başkenti"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Turkish LM (single GPU).")
    p.add_argument("--data_dir", required=True, help="dir with train/ and val/ .bin shards")
    p.add_argument("--out_dir", required=True, help="output dir for checkpoints + logs")
    p.add_argument("--model", default="124m", choices=list(configs.keys()))
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4, help="micro-batch per step (fit GPU)")
    p.add_argument("--grad_accum", type=int, default=32, help="gradient accumulation steps")
    p.add_argument("--max_steps", type=int, default=600000)
    p.add_argument("--lr", type=float, default=6e-4, help="peak learning rate")
    p.add_argument("--min_lr", type=float, default=6e-5, help="final (cosine floor) LR")
    p.add_argument("--warmup", type=int, default=2000, help="linear warmup steps")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="enable gradient checkpointing (saves VRAM, ~20%% slower)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--sp_model", default=None, help="SentencePiece .model for samples")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="fixed sample prompt")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--smoke", action="store_true", help="tiny dry run (few steps)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine decay to min_lr
# --------------------------------------------------------------------------- #
def lr_at(step: int, *, peak: float, floor: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return floor + coeff * (peak - floor)


# --------------------------------------------------------------------------- #
# Optimizer with decay/no-decay parameter groups
# --------------------------------------------------------------------------- #
def build_optimizer(model, lr, weight_decay, betas, fused_ok):
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))  # dedup tied params (lm_head.weight is tok_emb.weight)
        # Exclude norms (1D), biases, and embeddings from weight decay.
        if p.ndim < 2 or name.endswith("tok_emb.weight"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    extra = {"fused": True} if fused_ok else {}
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, **extra)


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_val_loss(model, loader, eval_iters, autocast_ctx) -> float | None:
    if not loader.has_split("val"):
        return None
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = loader.get_batch("val")
        with autocast_ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return float(losses.mean())


def generate_sample(model, sp, prompt, device, block_size) -> str:
    """Generate a short Turkish continuation from the fixed prompt."""
    if sp is None:
        return "(no sp_model provided; sample skipped)"
    ids = [1] + sp.encode(prompt, out_type=int)  # bos=1
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=60, temperature=0.8, top_p=0.95, eos_id=EOS_ID)
    return sp.decode(out[0].tolist())


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, optimizer, scaler, step, args, tokens_seen):
    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),  # preserve fp16 loss scale across resume
        "step": step,
        "tokens_seen": tokens_seen,
        "config": raw.cfg.__dict__,
        "args": vars(args),
        "torch_version": torch.__version__,
    }, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.smoke:  # shrink everything for a dry run
        args.max_steps = min(args.max_steps, 5)
        args.eval_every = 2
        args.eval_iters = 2
        args.ckpt_every = 5
        args.grad_accum = 2

    # --- precision: prefer bf16 (Turing has no bf16 tensor cores but autocast
    #     still works in emulation; fp16 + GradScaler is the faster fallback) ---
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_scaler = amp_dtype == torch.float16
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=amp_dtype)
        if device_type == "cuda" else contextlib.nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # --- model ---
    cfg = configs[args.model]
    cfg.block_size = args.block_size
    model = TurkishLM(cfg).to(device)
    if args.grad_checkpoint:
        _enable_grad_checkpointing(model)
    print(f"model={args.model} total_params={model.num_params():,} "
          f"non_embed={model.num_params(non_embedding=True):,} amp={amp_dtype}")

    fused_ok = device_type == "cuda"
    optimizer = build_optimizer(model, args.lr, args.weight_decay,
                                (args.beta1, args.beta2), fused_ok)

    # --- resume ---
    start_step = 0
    tokens_seen = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])  # restore fp16 loss scale
        start_step = ckpt["step"]
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"resumed from {args.resume} at step {start_step}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    loader = BatchLoader(args.data_dir, args.block_size, args.batch_size,
                         device=device, seed=args.seed + start_step)

    sp = _load_sp(args.sp_model)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    samples_path = os.path.join(args.out_dir, "samples.jsonl")
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size

    model.train()
    t0 = time.time()
    written_milestones = set()

    for step in range(start_step, args.max_steps):
        lr = lr_at(step, peak=args.lr, floor=args.min_lr,
                   warmup=args.warmup, total=args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # --- gradient accumulation ---
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(args.grad_accum):
            x, y = loader.get_batch("train")
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step

        # --- periodic console + jsonl logging ---
        if step % args.log_every == 0:
            dt = time.time() - t0
            _append_jsonl(log_path, {
                "step": step, "loss": round(loss_accum, 5), "val_loss": None,
                "lr": lr, "tokens_seen": tokens_seen, "wall_time": round(dt, 2),
            })
            print(f"step {step:>6} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"tok {tokens_seen:,} | {dt:.0f}s")

        # --- eval + sample ---
        is_last = step == args.max_steps - 1
        if step > 0 and (step % args.eval_every == 0 or is_last):
            val_loss = estimate_val_loss(model, loader, args.eval_iters, autocast_ctx)
            dt = time.time() - t0
            _append_jsonl(log_path, {
                "step": step, "loss": round(loss_accum, 5),
                "val_loss": round(val_loss, 5) if val_loss is not None else None,
                "lr": lr, "tokens_seen": tokens_seen, "wall_time": round(dt, 2),
            })
            sample = generate_sample(getattr(model, "_orig_mod", model), sp,
                                     args.prompt, device, args.block_size)
            _append_jsonl(samples_path, {"step": step, "sample": sample})
            print(f"  eval step {step}: val_loss="
                  f"{val_loss if val_loss is None else round(val_loss, 4)} | sample: {sample[:80]!r}")

        # --- checkpoints: milestones + every ckpt_every + last ---
        hit_milestone = step in MILESTONE_STEPS and step not in written_milestones
        if hit_milestone or (step > 0 and step % args.ckpt_every == 0) or is_last:
            written_milestones.add(step)
            ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, args, tokens_seen)
            # Also maintain a rolling "latest" for easy resume.
            save_checkpoint(os.path.join(args.out_dir, "ckpt_latest.pt"),
                            model, optimizer, scaler, step, args, tokens_seen)
            print(f"  saved {ckpt_path}")

    print(f"done. {args.max_steps} steps, {tokens_seen:,} tokens, "
          f"{time.time() - t0:.0f}s wall.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_sp(path):
    if not path:
        return None
    try:
        import sentencepiece as spm
    except ImportError:
        print("warning: sentencepiece not installed; samples disabled.")
        return None
    if not os.path.exists(path):
        print(f"warning: sp_model {path!r} not found; samples disabled.")
        return None
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def _enable_grad_checkpointing(model) -> None:
    """Wrap each transformer block's forward in checkpointing to cut activation VRAM."""
    from torch.utils.checkpoint import checkpoint

    for block in model.blocks:
        orig = block.forward

        def make(fn):
            def wrapped(x, cos, sin):
                return checkpoint(fn, x, cos, sin, use_reentrant=False)
            return wrapped

        block.forward = make(orig)


if __name__ == "__main__":
    main()
