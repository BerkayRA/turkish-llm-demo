"""Checkpoint Time Machine: generate the SAME Turkish prompts from each milestone
checkpoint so the dashboard can show the from-scratch model improving step by step.
Runs on CPU (the model is only ~110M params). Writes JSON for the dashboard."""
import json
import os
import sys

import torch
import torch.nn.functional as F
from model import ModelConfig, TurkishLM, _top_p_filter

SP = "/opt/corpus/out/tokenizer/sp_unigram_32000.model"
RUN = "/opt/demo/train/runs/u32_124m"
OUT = "/opt/demo/dashboard/checkpoint_progression.json"
BOS, EOS = 1, 2
PROMPTS = [
    "Türkiye'nin başkenti ",
    "Yapay zeka ",
    "Bir zamanlar ",
    "Bilim, insanlığın ",
]
MAXTOK, TEMP, TOPP, SEED = 45, 0.7, 0.9, 1234
ORDER = ["ckpt_step0.pt", "ckpt_step100.pt", "ckpt_step1000.pt",
         "ckpt_step10000.pt", "ckpt_step20000.pt", "ckpt_step30000.pt",
         "ckpt_latest.pt"]

import sentencepiece as spm  # noqa: E402
sp = spm.SentencePieceProcessor()
sp.load(SP)


@torch.no_grad()
def gen(model, prompt):
    torch.manual_seed(SEED)
    ids = [BOS] + sp.encode(prompt, out_type=int)
    idx = torch.tensor([ids], dtype=torch.long)
    bs = model.cfg.block_size
    out = []
    for _ in range(MAXTOK):
        cond = idx if idx.size(1) <= bs else idx[:, -bs:]
        logits, _ = model(cond)
        logits = logits[:, -1, :] / TEMP
        logits = _top_p_filter(logits, TOPP)
        probs = F.softmax(logits, dim=-1)
        nx = torch.multinomial(probs, 1)
        t = int(nx)
        if t == EOS:
            break
        out.append(t)
        idx = torch.cat((idx, nx), dim=1)
    return prompt + sp.decode(out)


results = []
for name in ORDER:
    path = os.path.join(RUN, name)
    if not os.path.exists(path):
        continue
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = TurkishLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    step = int(ck.get("step", 0))
    outs = {p: gen(model, p) for p in PROMPTS}
    results.append({"step": step, "outputs": outs})
    del model
    # write incrementally so partial progress survives an interruption
    results.sort(key=lambda r: r["step"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"prompts": PROMPTS, "checkpoints": results}, f,
                  ensure_ascii=False, indent=1)
    print(f"done {name} step={step} (written {len(results)} ckpts)",
          file=sys.stderr, flush=True)

print("WROTE", OUT, "with", len(results), "checkpoints")
