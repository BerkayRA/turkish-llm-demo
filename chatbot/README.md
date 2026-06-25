# Türkçe Chat Asistanı — QLoRA → Ollama

Fine-tune a small open base model into a **Turkish chat assistant** with
QLoRA, merge the adapter, convert to GGUF, and serve via **Ollama** (with
Open WebUI). Built for a single **NVIDIA Quadro RTX 4000 (8 GB VRAM,
Turing sm_75, CUDA 13)**.

## Base model

Default: **`Qwen/Qwen2.5-3B-Instruct`** — strong Turkish, ChatML template,
fits 8 GB under 4-bit QLoRA.

Alternatives (same pipeline, only `--base_model` changes):

| HF id | Notes | Chat template | Stop tokens |
|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` (default) | Best Turkish at 3B | ChatML | `<|im_end|>` |
| `meta-llama/Llama-3.2-3B-Instruct` | Gated; needs HF access | Llama-3 | `<|eot_id|>` |
| `google/gemma-2-2b-it` | Smallest; tightest 8 GB headroom | Gemma | `<end_of_turn>` |

> If you switch base models, update `DEFAULT_STOP_TOKENS` in
> `merge_and_export.py` to match the table above.

## 8 GB-fit notes (hard constraint)

The training config is tuned to fit 8 GB:

- **4-bit NF4 + double quant** (bitsandbytes) — ~3B weights in <3 GB
- **fp16 compute** — Turing (sm_75) has **no bf16**
- **gradient checkpointing** (`use_reentrant=False`)
- **`per_device_train_batch_size=1` + `grad_accum=16`** (effective batch 16)
- **`max_seq_len=1024`** — raising this is the fastest way to OOM
- **`paged_adamw_8bit`** optimizer (paged states spill to host on spikes)
- `model.config.use_cache=False` (required with checkpointing)

If you still OOM: lower `--max_seq_len` to 768/512, keep `--batch_size 1`,
raise `--grad_accum`, and close other GPU processes.

## Install

```bash
# 1. CUDA 12.x torch FIRST (cu12x runtime works under CUDA 13 drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2. the rest
pip install -r requirements.txt
```

## End-to-end run order

```bash
# 1) Prepare data — Turkish SFT set rendered via the chat template.
#    Online (real HF dataset):
python prepare_data.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --dataset merve/turkish_instructions \
    --out data/train.jsonl
#    OR fully offline (built-in 20 hand-written Turkish pairs, CC0):
python prepare_data.py --offline \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --out data/train.jsonl

# 2) QLoRA fine-tune (saves adapter to out/turkce-asistan-lora)
python finetune_qlora.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/train.jsonl \
    --out_dir out/turkce-asistan-lora \
    --epochs 3 --max_seq_len 1024 \
    --batch_size 1 --grad_accum 16 --lr 2e-4

# 3) Merge adapter -> fp16, emit GGUF + Ollama instructions + Modelfile
python merge_and_export.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter_dir out/turkce-asistan-lora \
    --merged_dir out/turkce-asistan-merged \
    --model_name turkce-asistan
```

`merge_and_export.py` prints the exact llama.cpp conversion + quantization
commands and writes a ready-to-use `Modelfile`.

## GGUF conversion (printed by step 3)

```bash
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build && cmake --build ../llama.cpp/build -j
pip install -r ../llama.cpp/requirements.txt

python ../llama.cpp/convert_hf_to_gguf.py out/turkce-asistan-merged \
    --outfile out/turkce-asistan-merged/turkce-asistan-f16.gguf --outtype f16

../llama.cpp/build/bin/llama-quantize \
    out/turkce-asistan-merged/turkce-asistan-f16.gguf \
    turkce-asistan-Q4_K_M.gguf Q4_K_M
```

## Ollama import

Place the quantized `.gguf` next to the generated `Modelfile`, then:

```bash
ollama create turkce-asistan -f Modelfile
ollama run turkce-asistan
```

The generated `Modelfile` sets the Turkish `SYSTEM` prompt and
`PARAMETER stop` / `temperature` / `top_p` / `num_ctx`.

## Serving on the box (Ollama 11435/11436 + Open WebUI)

This host runs Ollama on **ports 11435 and 11436** with Open WebUI in
front. Import the model against the right port and select it in the UI:

```bash
# Import into the Ollama instance on 11435
OLLAMA_HOST=127.0.0.1:11435 ollama create turkce-asistan -f Modelfile
OLLAMA_HOST=127.0.0.1:11435 ollama list   # confirm it registered

# (Repeat with 11436 if both instances must serve the model.)
```

In **Open WebUI**: Settings → Connections, point the Ollama base URL at
`http://127.0.0.1:11435` (or 11436), then pick `turkce-asistan` from the
model dropdown and start chatting in Turkish.

To serve the GGUF directly with low VRAM, Ollama will offload as many
layers to the 8 GB GPU as fit and run the rest on CPU automatically.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Turkish SFT set → chat messages (+offline fallback) |
| `finetune_qlora.py` | QLoRA SFT (transformers + peft + trl) |
| `merge_and_export.py` | merge → GGUF instructions → Ollama Modelfile |
| `requirements.txt` | pinned deps (cu12x torch note) |

## Dataset license note

`merve/turkish_instructions` is a Turkish Alpaca-style instruction set.
Many such sets are **distilled from proprietary models** (GPT-3.5/4) and
are therefore effectively **research-only** under the source model's
terms, regardless of the permissive license on the HF card. **Verify the
license before any commercial deployment.** For an unencumbered demo, use
`--offline`: the built-in 20-pair fallback is original, hand-written
Turkish content released here under **CC0**.
