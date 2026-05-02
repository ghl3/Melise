"""
Piece 6: The inference loop.

Generation as a real, two-phase program:
  PREFILL — feed the prompt in one forward pass, populate the KV cache,
            keep the last position's logits.
  DECODE  — in a tight loop, sample one token from those logits, feed it
            back as a single-token forward pass, get fresh logits, repeat.

We time the two phases separately. You should see the asymmetry we've been
talking about: prefill works on many tokens at once and runs at high
throughput; decode works on one token at a time and runs at much lower
throughput because the GPU is bandwidth-bound on weight + KV reads per step.

Sampling here is greedy (argmax). Temperature/top-k/top-p would all live
in the same place — `sample()` — without changing anything else.
"""

from __future__ import annotations
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128


class KVCache:
    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        Dh = cfg.d_model // cfg.n_heads
        self.k = torch.zeros(cfg.n_layers, batch_size, cfg.n_heads, cfg.max_seq_len, Dh, device=device)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def write(self, layer_idx, new_k, new_v):
        s_new = new_k.shape[-2]
        self.k[layer_idx, :, :, self.length : self.length + s_new] = new_k
        self.v[layer_idx, :, :, self.length : self.length + s_new] = new_v

    def advance(self, s_new):
        self.length += s_new


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, kv_cache=None):
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)

        if kv_cache is not None:
            pos_offset = kv_cache.length
            kv_cache.write(self.layer_idx, k, v)
            L = pos_offset + S
            k = kv_cache.k[self.layer_idx, :, :, :L]
            v = kv_cache.v[self.layer_idx, :, :, :L]
        else:
            pos_offset, L = 0, S

        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores = scores.masked_fill(~self.mask[pos_offset:pos_offset + S, :L], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        mlp_delta = self.mlp(self.norm2(x + attn_delta))
        return x + attn_delta + mlp_delta


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids, kv_cache=None):
        B, S = token_ids.shape
        pos_offset = kv_cache.length if kv_cache is not None else 0
        positions = torch.arange(pos_offset, pos_offset + S, device=token_ids.device)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(S)
        return self.lm_head(self.final_norm(x))


# ----------------------------- Inference -----------------------------

def sample(logits: torch.Tensor) -> int:
    """Greedy: return the argmax index. .item() syncs the GPU here."""
    return logits.argmax().item()


@torch.no_grad()
def generate(model: TransformerLM, prompt_ids: torch.Tensor, max_new_tokens: int):
    """Run prefill then decode, returning new_tokens and per-phase timing."""
    device = prompt_ids.device
    cache = KVCache(model.cfg, batch_size=prompt_ids.shape[0], device=device)

    # Prefill: feed the whole prompt in one shot.
    torch.mps.synchronize()
    t0 = time.perf_counter()
    logits = model(prompt_ids, kv_cache=cache)
    next_id = sample(logits[0, -1])      # also syncs (via .item())
    torch.mps.synchronize()
    prefill_secs = time.perf_counter() - t0

    new_tokens = [next_id]

    # Decode: one token at a time, fed back into the model.
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if cache.length >= model.cfg.max_seq_len:
            break       # KV cache is full; we'd need a longer max_seq_len
        next_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
        logits = model(next_input, kv_cache=cache)
        next_id = sample(logits[0, -1])
        new_tokens.append(next_id)
    torch.mps.synchronize()
    decode_secs = time.perf_counter() - t0

    return new_tokens, prefill_secs, decode_secs


# ------------------------------ Demo ---------------------------------

torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()
model = TransformerLM(cfg).to(device).eval()

# Warmup. The first forward pass on MPS pays Metal compilation overhead;
# we don't want that included in our timing.
warmup_ids = torch.tensor([[0, 0, 0, 0]], device=device, dtype=torch.long)
with torch.no_grad():
    _ = model(warmup_ids)
torch.mps.synchronize()

prompt = b"hello world. once upon a time"
prompt_ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
P = prompt_ids.shape[1]
N_NEW = 60
print(f"prompt: {prompt!r}  ({P} tokens)")
print(f"generating {N_NEW} new tokens...\n")

new_tokens, prefill_secs, decode_secs = generate(model, prompt_ids, N_NEW)

# Decoded output (model is untrained → meaningless content, but a real pipeline).
out_bytes = bytes(b if 0 <= b < 256 else 0x3f for b in new_tokens)  # '?' for non-bytes
print(f"prompt + generated:")
print(f"  {prompt!r}")
print(f"  +{out_bytes!r}\n")

# The asymmetry, in numbers.
prefill_tok_per_s = P / prefill_secs
decode_tok_per_s  = len(new_tokens) / decode_secs
print(f"PREFILL   {P:>3} tokens in {prefill_secs * 1000:6.2f} ms  →  {prefill_tok_per_s:7.0f} tokens/s")
print(f"DECODE    {len(new_tokens):>3} tokens in {decode_secs * 1000:6.2f} ms  →  {decode_tok_per_s:7.0f} tokens/s")
print(f"\nratio: prefill is {prefill_tok_per_s / decode_tok_per_s:.1f}x faster per token than decode")
