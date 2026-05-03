"""
Piece 5: The KV cache.

THE PROBLEM. Without a cache, generating one new token requires a *full
forward pass over the entire current sequence*. Step s recomputes K, V for
positions 0..s-1 even though they haven't changed since step 1. Generating
N tokens from a prompt of length P costs O((P+N)³) total — cubic in the
output length. Untenable for any real generation.

THE INSIGHT. K and V at position p depend only on the *input token* at p,
which once written never changes. Q depends on the current token only.
So:
    - Q is recomputed every step. It's small (one position) and cheap.
    - K and V are computed once per token and *stored* in a buffer.

THE PIPELINE. Two distinct phases:
    PREFILL — feed the entire prompt in one shot, populate the cache.
              Compute-bound (lots of FLOPs across many positions).
    DECODE  — feed exactly one new token, attend its Q against the cached
              K, V. Memory-bandwidth-bound (you read every cached vector
              from DRAM but only do a tiny amount of math per step).

These two phases live in completely different performance regimes. Real
inference engines optimize them separately.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64


class KVCache:
    """Pre-allocated K, V buffers, one slot per layer.

    Allocated once, mutated in place. The whole point is that decoding never
    calls the GPU allocator — every per-step write goes into the same buffers.
    """

    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        Dh = cfg.d_model // cfg.n_heads
        # Shape: (n_layers, B, H, max_seq_len, Dh). One big tensor for K, one for V.
        # `max_seq_len` defines the cache's capacity — generation can't exceed it.
        self.k = torch.zeros(cfg.n_layers, batch_size, cfg.n_heads, cfg.max_seq_len, Dh, device=device)
        self.v = torch.zeros_like(self.k)
        self.length = 0     # number of populated positions, shared across layers

    def write(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Append new K, V for this layer at positions [length, length + S_new)."""
        s_new = new_k.shape[-2]
        # Indexed assignment — in-place mutation of pre-allocated DRAM.
        self.k[layer_idx, :, :, self.length : self.length + s_new] = new_k
        self.v[layer_idx, :, :, self.length : self.length + s_new] = new_v

    def advance(self, s_new: int) -> None:
        """Advance the shared length pointer. Called once per forward pass."""
        self.length += s_new


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.layer_idx = layer_idx        # which slot of the KV cache to use
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim

        # Q, K, V for the *new* tokens only.
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)

        if kv_cache is not None:
            # Write the new K, V into the pre-allocated cache, then *read back*
            # the full sequence so far. K and V grow with each step; Q does not.
            pos_offset = kv_cache.length
            kv_cache.write(self.layer_idx, k, v)
            L_total = pos_offset + S
            k = kv_cache.k[self.layer_idx, :, :, :L_total]
            v = kv_cache.v[self.layer_idx, :, :, :L_total]
        else:
            pos_offset = 0
            L_total = S

        # Attention: Q is (B, H, S, Dh); K, V are (B, H, L_total, Dh).
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        # Mask: queries are at absolute positions [pos_offset, pos_offset+S);
        # keys are at [0, L_total). Slice the precomputed lower-triangular mask.
        scores = scores.masked_fill(~self.mask[pos_offset:pos_offset + S, :L_total], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        mlp_delta = self.mlp(self.norm2(x + attn_delta))
        return x + attn_delta + mlp_delta


class TransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        B, S = token_ids.shape
        # New tokens sit at absolute positions [pos_offset, pos_offset + S).
        pos_offset = kv_cache.length if kv_cache is not None else 0
        positions = torch.arange(pos_offset, pos_offset + S, device=token_ids.device)

        x = self.token_embed(token_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(S)
        x = self.final_norm(x)
        return self.lm_head(x)


# --- Demo: prove cached and uncached produce the same logits ---
torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()
model = TransformerLM(cfg).to(device).eval()

prompt = b"hello world"
ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
print(f"prompt: {prompt!r} (len={ids.shape[1]})")

# --- Path 1: feed everything in one shot, no cache ---
with torch.no_grad():
    naive_logits = model(ids)               # (1, 11, V)
naive_last = naive_logits[0, -1]            # logit for "what comes after the prompt"

# --- Path 2: prefill on prompt[:-1], then decode prompt[-1:] using the cache ---
cache = KVCache(cfg, batch_size=1, device=device)
with torch.no_grad():
    prefill_logits = model(ids[:, :-1], kv_cache=cache)
    print(f"after prefill on {ids.shape[1] - 1} tokens: cache.length = {cache.length}")

    decode_logits = model(ids[:, -1:], kv_cache=cache)
    print(f"after decode of  1 token:               cache.length = {cache.length}")

cached_last = decode_logits[0, -1]

# Both paths should produce the same logits for the position of the last token.
diff = (naive_last - cached_last).abs().max().item()
print(f"\nmax |naive - cached|:  {diff:.2e}")
print("✓ KV cache reproduces the naive output" if diff < 1e-4 else "✗ MISMATCH")

# --- Cache memory footprint ---
cache_bytes = cache.k.numel() * cache.k.element_size() + cache.v.numel() * cache.v.element_size()
print(f"\nKV cache buffer: {cache_bytes / 1024:.1f} KB allocated")
print(f"  shape per side: {tuple(cache.k.shape)} = {cache.k.numel():,} elements × {cache.k.element_size()} bytes")
print(f"  currently populated: {cache.length} / {cfg.max_seq_len} positions ({cache.length / cfg.max_seq_len * 100:.0f}%)")
