"""
Piece 3c: A full transformer block.

A block is two sublayers, each wrapped in a normalization and a residual:

    x = x + attention(norm(x))    # mixes information along S
    x = x + mlp(norm(x))          # mixes information along D

That's the entire architecture. Everything else is just stacking N of these.

Three design choices worth seeing in code:

  RESIDUAL: the sublayer's output is *added back* to its input rather than
  replacing it. The (B, S, D) "residual stream" is never overwritten — each
  layer can only refine what the previous layers wrote. Without residuals,
  deep stacks are untrainable (gradients vanish or explode through dozens of
  nested matmuls).

  PRE-NORM: we normalize *before* each sublayer (Pre-LN), not after. The
  output of the sublayer goes directly into the residual stream un-normalized.
  This is more stable than the original Transformer's Post-LN at depth, and
  it's what every modern model uses.

  RMSNORM, not LayerNorm: divide by RMS, no mean-subtraction, no bias. Cheaper
  than LayerNorm with no measurable accuracy loss. Llama, DeepSeek, etc. all
  use it.

The MLP here is a classic 2-layer feedforward with GELU. Modern models often
swap in SwiGLU (parallel gate projection); same role — mix information
along the D axis (attention is the only mixer along S).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    max_seq_len: int = 64


class RMSNorm(nn.Module):
    """Root-mean-square normalization. One learned per-dim scale, no shift."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-token RMS along the D axis: ||x_d|| / sqrt(D), with a tiny eps
        # to keep things well-defined when x is all zeros.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    """Same as in 04_attention.py — repeated so this file is self-contained."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores = scores.masked_fill(~self.mask[:S, :S], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class MLP(nn.Module):
    """Per-token feedforward. Up to 4D, GELU, back to D."""

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 4 * cfg.d_model      # standard expansion ratio
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """One transformer block: norm → attn → residual, then norm → mlp → residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_delta = self.attn(self.norm1(x))
        # MLP sees the *post-attention* residual stream, so attention's output
        # flows into MLP's input.
        mlp_delta = self.mlp(self.norm2(x + attn_delta))
        return x + attn_delta + mlp_delta


# --- Demo ---
device = torch.device("mps")
cfg = Config()
block = Block(cfg).to(device)

x = torch.randn(1, 5, cfg.d_model, device=device)
y = block(x)
print(f"input  shape={tuple(x.shape)}")
print(f"output shape={tuple(y.shape)}   <- same shape, residual-stream invariant")

print("\nparameters in one block:")
for name, p in block.named_parameters():
    print(f"  {name:<22} shape={str(tuple(p.shape)):<14} numel={p.numel():>8,}")
total = sum(p.numel() for p in block.parameters())
attn_n = sum(p.numel() for n, p in block.named_parameters() if n.startswith(("attn", "norm1")))
mlp_n  = sum(p.numel() for n, p in block.named_parameters() if n.startswith(("mlp", "norm2")))
print(f"\n  attention side: {attn_n:>8,}")
print(f"  mlp side:       {mlp_n:>8,}")
print(f"  total per block:{total:>8,}")
