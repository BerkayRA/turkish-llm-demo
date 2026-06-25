"""Llama-style decoder-only transformer for the Turkish LM A/B experiment.

Architecture (docs/EXPERIMENT_AB.md, Config A / "124m"):
  - RMSNorm (pre-norm), SwiGLU MLP, RoPE positional encoding, no biases.
  - Tied input/output embeddings (headline run).
  - GPT-2/nanoGPT init: std 0.02, residual projections scaled by 1/sqrt(2*n_layer).

Kept deliberately small and nanoGPT-flavoured. ~124M total params at vocab 32000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Decoder config. Defaults match Config A (~124M params) from the spec."""

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 32000
    block_size: int = 1024
    # SwiGLU intermediate dim. Spec Config A uses d_ff = 2048; if None we derive
    # a SwiGLU-idiomatic 8/3*d_model rounded to a multiple of 256.
    intermediate_size: int | None = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0  # pre-training: 0.0
    tie_embeddings: bool = True

    def ff_dim(self) -> int:
        if self.intermediate_size is not None:
            return self.intermediate_size
        hidden = int(8 * self.n_embd / 3)
        return 256 * ((hidden + 255) // 256)


# Presets. "124m" == Config A; "350m" is the optional scale-up (Config B-ish).
configs: dict[str, ModelConfig] = {
    "124m": ModelConfig(n_layer=12, n_head=12, n_embd=768, intermediate_size=2048,
                        block_size=1024),
    "350m": ModelConfig(n_layer=24, n_head=16, n_embd=1024, intermediate_size=2816,
                        block_size=2048),
}


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability under AMP, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Cast weight to the activation dtype so the output stays fp16 under AMP
        # (fp16 * fp32 would promote to fp32 and cascade through every block).
        return norm.to(dtype) * self.weight.to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device, dtype: torch.dtype):
    """Precompute RoPE cos/sin tables of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings. q,k: (B, n_head, T, head_dim); cos/sin: (T, head_dim)."""
    # Match the activation dtype: under fp16 AMP, fp16*fp32 would promote q/k to
    # fp32 and crash SDPA on the dtype mismatch with fp16 v (CRITICAL on Turing).
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        # Prefer fused SDPA (flash / mem-efficient) when available.
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # explicit fallback
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        ff = cfg.ff_dim()
        self.gate = nn.Linear(cfg.n_embd, ff, bias=False)
        self.up = nn.Linear(cfg.n_embd, ff, bias=False)
        self.down = nn.Linear(ff, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TurkishLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # weight tying

        # RoPE caches are buffers (not params); registered lazily / on device move.
        cos, sin = build_rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head,
                                    cfg.rope_theta, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual-projection weights (attn.proj, mlp.down) per GPT-2 init.
        scale = 1.0 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract the token-embedding table (also the tied head if tied).
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        # Inference path: only compute logits for the last position to save memory.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_p: float = 1.0,
                 eos_id: int | None = None) -> torch.Tensor:
        """Autoregressive sampling with temperature and nucleus (top-p) filtering."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab)

            if temperature <= 0.0:  # greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if 0.0 < top_p < 1.0:
                    logits = _top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Set logits outside the top-p nucleus to -inf (batched)."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens once cumulative prob exceeds top_p, but always keep the top-1.
    remove = cum_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back to original ordering.
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
