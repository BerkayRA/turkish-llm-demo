#!/usr/bin/env python3
"""Merge a QLoRA adapter into the base model and export for Ollama.

Pipeline:
  1. Load the base model in fp16, attach the LoRA adapter, ``merge_and_unload``.
  2. Save the merged fp16 model + tokenizer to ``--merged_dir``.
  3. Print the exact llama.cpp commands to convert the merged HF model to
     GGUF and quantize to Q4_K_M.
  4. Write an Ollama ``Modelfile`` (FROM the gguf, Turkish SYSTEM prompt,
     PARAMETER stop / temperature) and print the ``ollama create`` command.

Note: merging happens in fp16 on CPU/GPU RAM (not 4-bit) so the result is
a clean, deployable model. This step needs ~6-7 GB of RAM for a 3B model
but does NOT require the 8 GB GPU; it can run on CPU.

Example
-------
    python merge_and_export.py \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --adapter_dir out/turkce-asistan-lora \
        --merged_dir out/turkce-asistan-merged \
        --model_name turkce-asistan
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Mirrors the training system prompt so the served model behaves the same.
SYSTEM_PROMPT: str = (
    "Sen Türkçe konuşan, yardımsever ve güvenilir bir kurum içi yapay zeka "
    "asistanısın. Soruları açık, doğru ve nazik bir dille yanıtla. "
    "Emin olmadığın konularda tahmin yürütmek yerine bilmediğini belirt."
)

# Default stop tokens. Qwen2.5 uses ChatML (<|im_end|>, <|im_start|>);
# adjust for Llama (<|eot_id|>) or Gemma (<end_of_turn>) base models.
DEFAULT_STOP_TOKENS: tuple[str, ...] = ("<|im_end|>", "<|im_start|>")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF id of the base chat model used for fine-tuning.",
    )
    parser.add_argument(
        "--adapter_dir",
        default="out/turkce-asistan-lora",
        help="Directory of the trained LoRA adapter.",
    )
    parser.add_argument(
        "--merged_dir",
        default="out/turkce-asistan-merged",
        help="Output directory for the merged fp16 model.",
    )
    parser.add_argument(
        "--model_name",
        default="turkce-asistan",
        help="Name for the resulting Ollama model.",
    )
    parser.add_argument(
        "--llama_cpp_dir",
        default="../llama.cpp",
        help="Path to a llama.cpp checkout (for GGUF conversion commands).",
    )
    parser.add_argument(
        "--quant",
        default="Q4_K_M",
        help="GGUF quantization type for llama-quantize.",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merging (only regenerate Modelfile + instructions).",
    )
    return parser.parse_args()


def merge_adapter(base_model: str, adapter_dir: str, merged_dir: str) -> None:
    """Merge the LoRA adapter into the base model and save fp16 weights."""
    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    print(f"Loading base model {base_model} in fp16...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # merge on CPU to avoid GPU OOM
    )

    print(f"Attaching adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging adapter weights into base...")
    model = model.merge_and_unload()

    out = Path(merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(str(out))
    print(f"Merged fp16 model saved to {out}")


def gguf_instructions(
    merged_dir: str, model_name: str, llama_cpp_dir: str, quant: str
) -> tuple[str, str]:
    """Return (f16_gguf_path, quant_gguf_path) and print conversion steps."""
    f16_gguf = f"{merged_dir}/{model_name}-f16.gguf"
    quant_gguf = f"{model_name}-{quant}.gguf"

    print("\n" + "=" * 70)
    print("STEP: Convert the merged HF model to GGUF via llama.cpp")
    print("=" * 70)
    print(
        "# 1. Clone + build llama.cpp once (if you have not already):\n"
        f"#    git clone https://github.com/ggml-org/llama.cpp {llama_cpp_dir}\n"
        f"#    cmake -S {llama_cpp_dir} -B {llama_cpp_dir}/build && "
        f"cmake --build {llama_cpp_dir}/build -j\n"
        "#    pip install -r " + llama_cpp_dir + "/requirements.txt\n"
    )
    print(
        "# 2. Convert merged HF -> GGUF (fp16):\n"
        f"python {llama_cpp_dir}/convert_hf_to_gguf.py {merged_dir} \\\n"
        f"    --outfile {f16_gguf} --outtype f16\n"
    )
    print(
        "# 3. Quantize to " + quant + " (smaller, fits CPU/low-VRAM serving):\n"
        f"{llama_cpp_dir}/build/bin/llama-quantize {f16_gguf} {quant_gguf} {quant}\n"
    )
    return f16_gguf, quant_gguf


def write_modelfile(
    quant_gguf: str, model_name: str, modelfile_path: Path
) -> None:
    """Write an Ollama Modelfile referencing the quantized GGUF."""
    stop_lines = "\n".join(
        f'PARAMETER stop "{token}"' for token in DEFAULT_STOP_TOKENS
    )
    content = f"""# Ollama Modelfile for {model_name}
# Turkish on-prem chat assistant (QLoRA fine-tuned, {quant_gguf.split('-')[-1]}).
FROM ./{quant_gguf}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
{stop_lines}
"""
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"\nWrote Ollama Modelfile to {modelfile_path}")


def main() -> None:
    """Run merge + export, emit GGUF + Ollama instructions."""
    args = parse_args()

    if not args.skip_merge:
        merge_adapter(args.base_model, args.adapter_dir, args.merged_dir)
    else:
        print("Skipping merge step (--skip_merge).")

    _, quant_gguf = gguf_instructions(
        args.merged_dir, args.model_name, args.llama_cpp_dir, args.quant
    )

    modelfile_path = Path("Modelfile")
    write_modelfile(quant_gguf, args.model_name, modelfile_path)

    print("\n" + "=" * 70)
    print("STEP: Import into Ollama")
    print("=" * 70)
    print(
        "# Place the quantized gguf next to the Modelfile, then run:\n"
        f"ollama create {args.model_name} -f {modelfile_path}\n"
        f"ollama run {args.model_name}\n"
    )


if __name__ == "__main__":
    main()
