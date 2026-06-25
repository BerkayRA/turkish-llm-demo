#!/usr/bin/env python3
"""QLoRA SFT of a small base model into a Turkish chat assistant.

Targets a single NVIDIA Quadro RTX 4000 (8 GB VRAM, Turing sm_75).
To fit 8 GB we use:
  * 4-bit NF4 quantization with double-quant (bitsandbytes)
  * fp16 compute (Turing has no bf16)
  * gradient checkpointing
  * small per-device batch + gradient accumulation
  * paged_adamw_8bit optimizer
  * max_seq_len ~1024

LoRA: r=16, alpha=32, dropout=0.05 on attention + MLP projections.

Example
-------
    python finetune_qlora.py \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --dataset data/train.jsonl \
        --out_dir out/turkce-asistan-lora \
        --epochs 3 --max_seq_len 1024 \
        --batch_size 1 --grad_accum 16 --lr 2e-4

``--dataset`` accepts either a local JSONL produced by prepare_data.py
(with a "messages" or "text" field) or an HF dataset id.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# System prompt mirrors prepare_data.py so training and serving align.
SYSTEM_PROMPT: str = (
    "Sen Türkçe konuşan, yardımsever ve güvenilir bir kurum içi yapay zeka "
    "asistanısın. Soruları açık, doğru ve nazik bir dille yanıtla. "
    "Emin olmadığın konularda tahmin yürütmek yerine bilmediğini belirt."
)

# LoRA target modules covering attention + MLP projections.
# Names match Qwen2/Llama/Gemma-style architectures.
LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF id of the base chat model.",
    )
    parser.add_argument(
        "--dataset",
        default="data/train.jsonl",
        help="Local JSONL path or HF dataset id.",
    )
    parser.add_argument(
        "--out_dir",
        default="out/turkce-asistan-lora",
        help="Directory to save the trained LoRA adapter.",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Max sequence length (keep ~1024 to fit 8 GB).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device train batch size (keep small for 8 GB).",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=16,
        help="Gradient accumulation steps (effective batch = batch*accum).",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank (r)."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    return parser.parse_args()


def load_train_dataset(dataset_arg: str, tokenizer: Any) -> Any:
    """Load and normalise the training dataset to a 'text' column.

    Accepts a local JSONL (messages/text) or an HF dataset id. Any rows
    with only 'messages' are rendered via the chat template.
    """
    from datasets import load_dataset  # type: ignore

    path = Path(dataset_arg)
    if path.exists():
        dataset = load_dataset("json", data_files=str(path), split="train")
    else:
        dataset = load_dataset(dataset_arg, split="train")

    columns = set(dataset.column_names)

    if "text" in columns:
        return dataset

    if "messages" in columns:
        def _render(example: dict[str, Any]) -> dict[str, str]:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        return dataset.map(_render, remove_columns=list(columns))

    raise ValueError(
        "Dataset must contain a 'text' or 'messages' column; "
        f"found columns: {sorted(columns)}"
    )


def build_quant_config() -> Any:
    """4-bit NF4 double-quant config with fp16 compute (Turing-safe)."""
    import torch  # type: ignore
    from transformers import BitsAndBytesConfig  # type: ignore

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def build_model_and_tokenizer(base_model: str) -> tuple[Any, Any]:
    """Load the 4-bit base model + tokenizer, prepared for k-bit training."""
    from peft import prepare_model_for_kbit_training  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_quant_config(),
        device_map={"": 0},
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    return model, tokenizer


def build_lora_config(args: argparse.Namespace) -> Any:
    """Build the LoRA configuration."""
    from peft import LoraConfig  # type: ignore

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def build_sft_config(args: argparse.Namespace) -> Any:
    """Build the trl SFTConfig tuned for 8 GB VRAM."""
    from trl import SFTConfig  # type: ignore

    return SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        fp16=True,  # Turing has no bf16
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        packing=False,
    )


def main() -> None:
    """Run QLoRA SFT and save the adapter."""
    args = parse_args()
    print(f"Base model: {args.base_model}")
    print(f"System prompt: {SYSTEM_PROMPT[:60]}...")

    model, tokenizer = build_model_and_tokenizer(args.base_model)
    train_dataset = load_train_dataset(args.dataset, tokenizer)
    print(f"Training examples: {len(train_dataset)}")

    from trl import SFTTrainer  # type: ignore

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(args),
        train_dataset=train_dataset,
        peft_config=build_lora_config(args),
        processing_class=tokenizer,
    )

    trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter + tokenizer to {out_dir}")


if __name__ == "__main__":
    main()
