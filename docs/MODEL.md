# The Model — *Egemen Türkçe Yapay Zeka*

*A plain-but-honest account of what this model is, why it exists, and how it was built. Written for engineers with some ML familiarity, and for colleagues and investors who want the real picture without the hand-waving.*

---

## TL;DR

We built a **Turkish language model from the data up** — our own tokenizer, our own cleaned corpus, our own training code — and we run the whole thing **on-premise on a single 8 GB GPU**. Nothing leaves the building.

- **What it is:** a ~**110-million-parameter** GPT-style transformer, trained from random initialization (not a fine-tune of someone else's model) on **~39 billion tokens** of Turkish text we collected and cleaned ourselves.
- **Why it matters:** Turkish is **agglutinative** — words carry meaning through stacked suffixes — and generic English-tuned tokenizers shred it into inefficient fragments. Our purpose-built tokenizer represents the same Turkish text in **~40% fewer tokens**, which means cheaper, faster, more coherent modelling.
- **The sovereignty angle:** data collection, tokenization, training, and serving all happen on hardware we control. This is the difference between "send your citizens'/company's Turkish data to a foreign API" and "**the data never leaves the building**" — directly relevant to **KVKK** (Turkey's GDPR) and any regulated environment.
- **Status (2026-06-30):** training is **live and healthy** — ~36,300 steps, ~4.75 B tokens seen, training loss down from **10.5 → ~3.3**, and the model now produces fluent, grammatical Turkish. It is a **research-stage proof**, not a finished product.

This repository pairs that from-scratch model (the *proof*) with a polished interactive **Turkish chatbot** (the *product*) and a dashboard that tells the whole story.

---

## 1. What problem this solves

Most "Turkish support" in today's large models is incidental — a few percent of an English-first training run. Two concrete consequences:

1. **Tokenizer inefficiency.** A tokenizer trained mostly on English splits Turkish suffixes letter-by-letter. Every extra token is extra cost, extra latency, and a longer path for the model to learn meaning. (See §3 for our measured numbers.)
2. **Data sovereignty.** Sending Turkish corporate, legal, or citizen data to a third-party cloud API is, for many Turkish institutions, a compliance non-starter under **KVKK**. There is real demand for a capable Turkish model that runs **entirely on local infrastructure**.

This project is a concrete answer to both: a Turkish-first tokenizer and corpus, a model trained on them, and an end-to-end pipeline that runs disconnected from the internet.

## 2. What the model is (architecture)

A standard, modern **decoder-only transformer** in the Llama family — deliberately conventional, because the novelty here is the *data and tokenizer*, not the architecture.

| Property | Value |
|---|---|
| Parameters | **109,529,856** (~110 M) |
| Layers | 12 |
| Hidden size | 768 |
| Attention | multi-head, with `scaled_dot_product_attention` (flash / mem-efficient) |
| Normalization | **RMSNorm** (pre-norm) |
| Feed-forward | **SwiGLU** |
| Positional encoding | **RoPE** (rotary) |
| Embeddings | **tied** input/output |
| Context length | 1024 tokens |
| Vocabulary | 32,000 (SentencePiece, see §3) |

This is roughly "GPT-2 small, modernized." It is intentionally small enough to **train and serve on one 8 GB GPU**, which is the whole point: proving the *pipeline* works end-to-end on accessible hardware. The architecture scales cleanly — a ~350 M configuration exists in the code for when more compute is available.

## 3. The tokenizer — where Turkish actually wins

The tokenizer is a **SentencePiece Unigram** model with a **32,000-token** vocabulary, character coverage 0.9995, and **byte-fallback** (so it can never fail on unseen characters). It was trained on our own Turkish corpus, so its subword units align with Turkish morphology instead of English.

The headline metric is **fertility** — tokens per word. Lower is better: fewer, larger, more meaningful pieces.

| Tokenizer | Tokens/word on Turkish | Relative cost |
|---|---|---|
| **Ours** (`sp_unigram_32000`) | **~2.4** | baseline |
| GPT-4o (`o200k_base`) | ~4.0 | ~1.7× more tokens |
| Llama-3 | ~5.6 | ~2.3× more tokens |

In practical terms, the same Turkish paragraph costs us **~40% fewer tokens** than GPT-4o and **less than half** what a Llama-3 tokenizer would. That compounds across every training step and every inference call. The repository's **tokenizer visualizer** lets you paste Turkish text and *see* this difference, token by token.

## 4. The corpus — how the data was built

A model is only as good as its data. We assembled the corpus ourselves rather than scraping indiscriminately:

- **Source:** [HPLT v2](https://hplt-project.org/) cleaned Turkish — a **CC0 (public-domain)** web corpus. License-clean from the start.
- **Scale:** ~172 GB of raw Parquet ingested.
- **Deduplication:** near-duplicate removal via **MinHash/LSH**, which removed **~30%** of documents. Duplicate text causes memorization and wastes compute.
- **Privacy:** **KVKK-oriented PII scrubbing** (emails, phone numbers, national ID numbers) as part of cleaning.
- **Result:** **53,691,924** unique Turkish documents → tokenized to **~38.9 B training + ~0.2 B validation ≈ 39 B tokens**, stored as compact `uint16` binary shards for fast memory-mapped training.

For reference, that token budget is well beyond "Chinchilla-optimal" for a 110 M model, so data quantity is not the limiting factor — we can train long for quality.

## 5. How it was trained

Everything runs on a **single NVIDIA Quadro RTX 4000 (8 GB, Turing sm_75)**. Fitting a real training run into 8 GB drove most of the engineering decisions.

| Setting | Value | Why |
|---|---|---|
| Precision | **fp16** mixed-precision (AMP) | Turing has fp16 *tensor cores* but **no bf16 tensor cores** — see the note below |
| Effective batch | micro-batch 8 × grad-accum 16 × 1024 ctx ≈ **131 K tokens/step** | keeps a large effective batch while fitting memory |
| Memory technique | **gradient checkpointing** | trades ~20–30% compute to fit activations in 8 GB |
| Optimizer | **AdamW** (β = 0.9/0.95, weight decay 0.1, grad-clip 1.0) | standard, stable |
| LR schedule | cosine, **peak 6e-4 → floor 6e-5**, 2000-step warmup | standard nanoGPT-style |
| Checkpointing | every **250 steps** (+ milestones) | durable: a crash loses minutes, not hours |
| Data loading | memory-mapped `uint16` shards | streams 39 B tokens without loading them into RAM |

**The fp16 story (a real result, not a footnote).** The run originally used bf16. On a Turing GPU, bf16 has no hardware tensor-core support and is effectively *emulated*, while fp16 uses the real tensor cores. Switching the run to fp16 made it **~4× faster — from ~44 s/step to ~10 s/step** — with no loss of stability. That single change is the difference between this model reaching a useful checkpoint in days versus weeks.

The fp32 cross-entropy **logits tensor** (`batch × seq × vocab × 4 bytes ≈ 1 GB`) is the real memory ceiling on 8 GB — not the weights — which is why micro-batches stay small and gradient checkpointing is mandatory.

## 6. Where it is now (results)

A live snapshot as of **2026-06-30**:

- **~36,300 optimization steps**, **~4.75 B tokens** seen, ~4 days of continuous training.
- **Training loss: 10.52 → ~3.3.** (Loss is noisy step-to-step because each step sees a fresh slice of text; the *trend* is what matters.)
- **Qualitatively**, generated Turkish has gone from broken and repetitive to **grammatical and fluent**. A sample from step 2,500:

  > *"Türkiye'nin başkenti olan Antalya'ya komşu olduğunu ve bu ziyaretin nasıl geçtiğini... çok önemli olduğunu görüyoruz."*

  Note it is **confidently wrong about facts** (Antalya is not the capital) — which is exactly right for this stage. A model this size, this early, learns **fluency and grammar first**; factual reliability comes from scale and longer training. We show this honestly on the dashboard as "watch it learn."

The training dashboard streams the loss curve and these sample generations in real time.

## 7. From proof to product — the chatbot

The from-scratch model proves the pipeline. For an **interactive** Turkish assistant today, the repo also fine-tunes a strong open base model:

- **Base:** Qwen2.5-3B-Instruct (good Turkish, fits 8 GB under 4-bit quantization).
- **Method:** **QLoRA** (4-bit NF4 + LoRA adapters) → merge → GGUF → served via **Ollama** with a Turkish on-prem system persona.
- **Where it runs:** the second GPU on the same box, so the from-scratch model can keep training uninterrupted.

This "hybrid" gives both an impressive *interactive product now* and an honest *from-scratch proof* of sovereign capability.

## 8. How the pieces fit together

This model is the endpoint of a four-repository pipeline (each links to the next):

```
turkish-tokenizer  →  turkish-llm  →  turkish-corpus  →  turkish-llm-demo
(morphology +         (tokenizer +     (HPLT clean →      (THIS: live training,
 fertility metric)     arch spec)       dedup → blend)     chatbot, dashboard)
```

- **turkish-tokenizer** — the morphological analyzer and the fertility methodology.
- **turkish-llm** — trains/evaluates the tokenizer and defines the model architecture.
- **turkish-corpus** — the HPLT-anchored cleaning/dedup pipeline that produced the 39 B-token corpus.
- **turkish-llm-demo** *(here)* — trains the model on that corpus, serves the chatbot, and presents the story.

## 9. Honest limitations

Credibility requires stating what this is **not**:

- **It is small (~110 M).** It will not match GPT-4-class reasoning or factual recall. It is a *proof of sovereign capability*, sized to accessible hardware.
- **It is research-stage.** Training is ongoing; the from-scratch model is not production-tuned for chat or instruction-following yet.
- **Facts lag fluency.** As shown above, expect grammatical Turkish well before reliable facts.
- **Single-GPU constraints** shape every choice (model size, batch size, sequence length). More compute lifts all of these.
- **The interactive chatbot** is a fine-tune of an open base model, not the from-scratch model — clearly labelled as such throughout.

## 10. What's next

- Continue training toward larger token budgets and lower loss; publish checkpoint-by-checkpoint quality progression.
- Bespoke instruction/chat fine-tune of the **from-scratch** model once it matures.
- Scale the architecture (the ~350 M config) when multi-GPU compute is available.
- Formal evaluation on Turkish benchmarks (perplexity, downstream tasks) and tokenizer A/B (Unigram vs. morpheme-aware BPE).

---

*Built and operated entirely on-premise. Veri sizindir, model sizindir, karar sizindir — the data is yours, the model is yours, the decision is yours.*
