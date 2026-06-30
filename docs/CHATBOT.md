# The Chatbot — *Türkçe Asistan*

*What the interactive Turkish chatbot is, what it is **not**, what we actually built, and a plain explanation of QLoRA. Written for a technically-literate audience — and deliberately honest about provenance, because conflating this with the from-scratch model would be misleading.*

---

## The one-paragraph version

The chatbot is a **fine-tune of an existing open-weights model** (Alibaba's **Qwen2.5-3B-Instruct**), adapted with **QLoRA** to take on a Turkish on-prem assistant persona and served entirely on our own hardware via Ollama. It is **not** our from-scratch model, and it is **not** trained from scratch on our data. Its underlying language ability comes from Alibaba's pre-training; **what is ours** is the adaptation, the data curation, the 8 GB-fit training pipeline, the persona, and the fully on-premise serving. It exists to give the demo a **capable, interactive product today**, while the [from-scratch model](MODEL.md) is the **sovereign-capability proof** that matures over time.

## Why there are two models at all

This project is deliberately a **hybrid**:

| | [From-scratch model](MODEL.md) | **This chatbot** |
|---|---|---|
| Built from | random weights, **our** corpus + tokenizer | **Qwen2.5-3B-Instruct** (pre-trained by Alibaba) |
| What it proves | **sovereign capability** — we can build a Turkish LM from the data up | **a usable product now** — interactive, polished, on-prem |
| Provenance | fully ours | base is third-party; **adaptation** is ours |
| Maturity | research-stage, training for days | immediately useful |
| Size | ~110 M params | 3 B params |

Honest framing: the from-scratch model is the *integrity* of the project; the chatbot is the *usability*. We keep them clearly separate and never present the chatbot's fluency as evidence of the from-scratch model's ability.

## What it's built on

- **Base model: Qwen2.5-3B-Instruct** — a 3-billion-parameter, instruction-tuned, open-weights model released by Alibaba under the **Apache-2.0** license. It already has strong multilingual ability (including decent Turkish) and knows how to follow instructions and hold a conversation. We chose it because it is **genuinely open**, has good Turkish, and **fits 8 GB** under 4-bit quantization.
- Alternatives the same pipeline supports (swap one flag): Llama-3.2-3B-Instruct, Gemma-2-2B-it.

### What "building on" actually entails

"Building on a base model" does **not** mean we trained those 3 billion parameters — that would cost millions of dollars and enormous compute. It means:

1. We take the **frozen, pre-trained base** (it already speaks language).
2. We **adapt** it with a small amount of additional training (QLoRA, see below) so it reliably adopts our **Turkish on-prem assistant persona** and response style.
3. We **package and serve** it ourselves, on our own hardware, behind our own system prompt.

The base supplies *general* language and knowledge; our fine-tune supplies *behaviour and identity*. The vast majority of what the model "knows" comes from Alibaba's pre-training — that is the honest reality of any fine-tune, and we state it plainly.

## What is genuinely **ours**

- **The QLoRA fine-tuning pipeline** (`chatbot/finetune_qlora.py`): 4-bit loading, LoRA configuration, the trainer setup, and the engineering to make a 3 B model train on a single **8 GB** GPU — including the precision handling that lets it run on hardware as old as a Pascal Quadro P4000.
- **The training data** for the demo: an original, hand-written set of Turkish instruction/response pairs released here under **CC0** (deliberately license-clean, so nothing is borrowed from proprietary model outputs).
- **The Turkish on-prem persona** — the system prompt that defines a careful, honest, institution-internal Turkish assistant.
- **The full on-prem serving stack** — merge → quantize → Ollama, running disconnected from the internet, plus the chat UI and its integration into the dashboard.
- **The sovereignty wrapper** — the entire thing runs where the data lives.

What is **not** ours: the base model's weights, its pre-training corpus, or its core knowledge and reasoning. We are transparent about that.

## QLoRA in one minute

Fine-tuning a 3-billion-parameter model the naive way means updating all 3 billion weights — which needs far more than 8 GB of GPU memory. **QLoRA** (Quantized Low-Rank Adaptation) makes it fit on a single small GPU with two ideas:

1. **Quantize and freeze the base.** The 3 B base weights are compressed to **4-bit** (NF4) and **never updated**. This shrinks the model from ~6 GB to ~2 GB and removes it from the training math entirely.
2. **Train tiny adapters instead.** Small **low-rank** matrices (LoRA adapters) are injected next to the model's attention and MLP layers. *Only these* are trained — here, rank 16, which is a few million parameters versus the frozen 3 billion. The result is a **~60 MB adapter** that "steers" the frozen base.

So the entire fine-tune is: *freeze a 4-bit base, train a small steering layer on top.* That's why a 3 B model can be adapted on an 8 GB card in minutes. At serving time the adapter is **merged** back into the base and the whole thing is quantized again for efficient inference.

### Our specific run

- Base: Qwen2.5-3B-Instruct, 4-bit NF4 + double-quant.
- LoRA: rank 16, alpha 32, dropout 0.05, on the attention + MLP projections.
- Data: 20 CC0 Turkish instruction pairs, 3 epochs (a **persona/format** tune — small by design for the demo, not a deep capability transfer).
- Hardware: a single **NVIDIA Quadro RTX 4000** is the documented target; this run was executed on the box's second card, a **Pascal Quadro P4000**, in **fp32** (Pascal lacks fp16/bf16 tensor kernels, so fp32 is both correct and its fastest path).
- Result: a LoRA adapter (`train_loss ≈ 2.77`), merged and served via **Ollama** as `turkce-asistan`.

## What it **is**

- An **interactive Turkish chat assistant** you can talk to right now.
- **100% on-premise** — model, weights, and inference all stay on local hardware; no third-party API, no data leaving the building. This is the live demonstration of the KVKK/sovereignty story.
- A clean, swappable pipeline: change the base model or the data, re-run, redeploy.

## What it **is not**

- **Not the from-scratch sovereign model.** Its capabilities come from Alibaba's pre-training, not from our corpus. (See [MODEL.md](MODEL.md) for the model that *is* ours end-to-end.)
- **Not a from-scratch Turkish model**, and we never imply it is.
- **Not deeply re-trained.** The demo adapter used 20 examples — enough to set persona and format, not to add substantial new knowledge or skill.
- **Not fully sovereign in provenance.** The serving is sovereign; the base model's origins are not. For a *fully* sovereign chat assistant, the endpoint is the from-scratch model, once mature.
- **Not unconditionally licensed for commercial use.** Qwen2.5 is Apache-2.0 (permissive), but always verify the base model's license, and note that many *third-party* Turkish instruction datasets are distilled from proprietary models and are research-only — which is exactly why our demo data is original CC0.

## In short

The chatbot is the **honest product layer**: a real, on-prem, interactive Turkish assistant built by **adapting** a strong open base model with our own data and pipeline. The from-scratch model is the **honest proof layer**: smaller and younger, but **ours from the first token**. Presented together — and never confused — they tell the complete story.
