# Turkish LM — from-scratch training stack

A small, nanoGPT-style training stack for a Llama-style decoder (RMSNorm,
SwiGLU, RoPE, tied embeddings) following `docs/EXPERIMENT_AB.md` Config A
(~124M params). Built to train on a **single 8 GB GPU** (NVIDIA Quadro RTX 4000,
Turing sm_75, CUDA 13, torch cu12x).

## Files

| file | purpose |
|---|---|
| `model.py` | Llama-style decoder + `ModelConfig`, `configs` presets (`124m`, `350m`), `generate()` |
| `data.py` | memmap uint16 `.bin` shard dataset + `BatchLoader.get_batch(split)` |
| `train.py` | training loop: AdamW, cosine LR + warmup, AMP, grad accum, checkpoints, JSONL logging |
| `generate.py` | load checkpoint + SentencePiece model, streaming generation to stdout |
| `requirements.txt` | torch (cu12x), numpy, sentencepiece |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# On the GPU box, install the CUDA build of torch explicitly:
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data layout

The loader memmaps a directory of uint16 `.bin` shards (nanoGPT format), each
produced by `np.array(token_ids, dtype=np.uint16).tofile(path)` — concatenated
document token streams with `EOS=2` between documents:

```
<data_dir>/
  train/  shard_0000.bin  shard_0001.bin  ...
  val/    val_0000.bin    ...
```

Tokenizer ids (SentencePiece `sp_unigram_32000.model`, vocab 32000):
`unk=0  bos=1  eos=2  pad=3`.

## Train

8 GB-fit baseline (micro-batch 4 × grad-accum 32 = 128 seqs/step × 1024 ctx ≈
131k tokens/step):

```bash
python train.py \
  --data_dir /scratch/$USER/turkish-llm/cache/U32 \
  --out_dir  out/U32_seed0 \
  --model 124m --block_size 1024 \
  --batch_size 4 --grad_accum 32 \
  --max_steps 600000 \
  --lr 6e-4 --min_lr 6e-5 --warmup 2000 \
  --sp_model models/sp_unigram_32000.model \
  --prompt "Türkiye'nin başkenti"
```

Resume:

```bash
python train.py --data_dir ... --out_dir out/U32_seed0 --resume out/U32_seed0/ckpt_latest.pt ...
```

Tiny dry run (verifies paths + a few steps before queuing a long job):

```bash
python train.py --data_dir <dir> --out_dir /tmp/smoke --model 124m \
  --block_size 256 --batch_size 2 --grad_accum 2 --smoke
```

### Outputs

- `out_dir/ckpt_step{N}.pt` — at steps 100, 1000, 10000, every `--ckpt_every`, and last.
- `out_dir/ckpt_latest.pt` — rolling latest, for `--resume`.
- `out_dir/train_log.jsonl` — one JSON/line: `{step, loss, val_loss, lr, tokens_seen, wall_time}`.
- `out_dir/samples.jsonl` — one JSON/line per eval: `{step, sample}` (feeds a "watch it learn" dashboard).

## Generate

```bash
python generate.py \
  --ckpt out/U32_seed0/ckpt_latest.pt \
  --sp_model models/sp_unigram_32000.model \
  --prompt "Türkiye'nin başkenti" \
  --max_new_tokens 120 --temperature 0.8 --top_p 0.95
```

## Fitting 8 GB VRAM (Quadro RTX 4000)

The 124M model in fp16/bf16 weights + AdamW fp32 state (~8 bytes/param of
optimizer state) is the dominant fixed cost; activations scale with
`batch_size × block_size`. Levers, in order of preference:

1. **Small micro-batch + gradient accumulation.** Keep `--batch_size` small
   (start at 4, drop to 2 or 1 if OOM) and raise `--grad_accum` to keep the
   global tokens/step constant (~0.5M tokens/step is the spec target; the
   default 4×32 is a conservative ~131k that still trains well).
2. **AMP.** bf16 autocast is used automatically if supported; otherwise fp16 +
   `GradScaler`. Turing has no bf16 tensor cores, so fp16 is typically faster
   there — the code picks bf16 only when `torch.cuda.is_bf16_supported()`.
3. **Gradient checkpointing.** Add `--grad_checkpoint` to trade ~20% compute for
   a large activation-memory reduction; this is what lets `block_size 1024` fit
   alongside a non-trivial micro-batch on 8 GB.
4. **Reduce `--block_size`** (e.g. 512) as a last resort if still OOM.

Suggested first config on 8 GB: `--batch_size 2 --grad_accum 64 --grad_checkpoint`
at `--block_size 1024`. If it fits comfortably, raise `batch_size` to 4 and halve
`grad_accum`.

`fused=True` AdamW and TF32 matmul are enabled on CUDA for throughput.

## Notes

- `350m` preset (24L/1024d/16h, ctx 2048) is provided for an optional scale-up
  (spec Config B); it will **not** fit 8 GB without aggressive sharding — train
  it multi-GPU (see `docs/GPU_OPS.md`).
- The model uses `F.scaled_dot_product_attention` (flash / mem-efficient) when
  available, with an explicit masked-softmax fallback.
