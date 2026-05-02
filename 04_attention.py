"""
Piece 3b: Causal self-attention.

This is the *only* operator in a transformer that mixes information between
positions in the sequence. Embedding, MLP, layer norm — all per-token.
Attention is what lets position 4 look back at positions 0..3.

In one sentence:
  For each position i, produce an output vector that is a weighted sum of
  all *earlier* positions' "value" vectors. The weights come from how much
  i's "query" vector matches each earlier position's "key" vector.

Three derived vectors per token, called Q / K / V, all of shape D:
  Q (query): "what am I looking for?"
  K (key):   "what do I represent?"
  V (value): "what do I contribute, if attended to?"

All three are linear projections of the input vector for that token, learned.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    max_seq_len: int = 64


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        # Three (D, D) projections — Q, K, V — fused into one (D, 3D) matmul.
        # Functionally identical to three separate Linear layers, but one big
        # matmul is faster than three small ones.
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)

        # After the heads have done their thing, recombine into D again.
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # Causal mask: position i may attend to positions [0..i] only. A boolean
        # lower-triangular matrix. Registered as a *buffer* (moves with .to(device)
        # but not a learnable parameter). persistent=False so it isn't saved into
        # checkpoints — it's deterministic from max_seq_len.
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim

        # Project to Q, K, V in one matmul, then split into three (B, S, D) tensors.
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        # Reshape so the head axis comes *before* the sequence axis. This lets
        # the attention math run independently per head as a batched matmul.
        # (B, S, D) -> (B, S, H, Dh) -> (B, H, S, Dh)
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)

        # Attention scores: every query · every key, scaled by sqrt(Dh).
        # Shape (B, H, S, S). Entry [b, h, i, j] = how much position i wants
        # to attend to position j, in head h.
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)

        # Causal mask: future positions get -inf, becoming 0 after softmax.
        scores = scores.masked_fill(~self.mask[:S, :S], float("-inf"))

        # Normalize across "which positions to attend to" — the last axis.
        weights = torch.softmax(scores, dim=-1)            # (B, H, S, S)

        # Weighted sum of values.
        out = weights @ v                                   # (B, H, S, Dh)

        # Concatenate heads back into D, then mix them with the output projection.
        # The .contiguous() is needed because transpose returns a non-contiguous
        # view, and view() requires contiguous memory.
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


# --- Demo ---
device = torch.device("mps")
cfg = Config()
attn = CausalSelfAttention(cfg).to(device)

# A random "residual stream" tensor of the right shape — pretend it came from
# the embedding layer.
x = torch.randn(1, 5, cfg.d_model, device=device)
y = attn(x)
print(f"input  shape={tuple(x.shape)}")
print(f"output shape={tuple(y.shape)}   <- same shape, residual-stream invariant")

# Inspect the attention pattern for head 0. We re-run the math so we can grab
# the intermediate `weights` tensor that forward() doesn't return.
with torch.no_grad():
    H, Dh = cfg.n_heads, attn.head_dim
    q, k, _ = attn.qkv_proj(x).chunk(3, dim=-1)
    q = q.view(1, 5, H, Dh).transpose(1, 2)
    k = k.view(1, 5, H, Dh).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
    scores = scores.masked_fill(~attn.mask[:5, :5], float("-inf"))
    weights = torch.softmax(scores, dim=-1)[0, 0].cpu()  # head 0

print("\nattention pattern, head 0 (rows = queries, cols = keys):")
print("        " + "  ".join(f"k{j}  " for j in range(5)))
for i, row in enumerate(weights.tolist()):
    print(f"  q{i}:  " + "  ".join(f"{w:.2f}" for w in row))
