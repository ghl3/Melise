"""Per-layer inference caches for the different attention types.

Each attention mechanism needs a different thing remembered between
decode steps:

    GQA  — the K and V vectors of every past token (a growing buffer).
    MLA  — only the small compressed latent per past token (plus the
           shared RoPE key slice when decoupled RoPE is enabled). K and V
           are re-expanded from the latent on the fly; this is the whole
           point of MLA.
    KDA  — a fixed-size recurrent state S per head (d_k × d_v), plus the
           tail of the short-convolution windows. Nothing grows with
           sequence length — this is the whole point of linear attention.

Each attention module knows which cache it needs: `attn.make_cache(B,
device)` returns the right layer cache with the right shapes. A model
collects one per attention layer into a `ModelCache`, which adds the
single shared `length` counter, advanced once per forward pass after all
layers have written.
"""

from __future__ import annotations

import torch


class KVLayerCache:
    """Growing K/V buffers for one GQA layer.

    Shape per side: (B, n_kv_heads, max_seq_len, head_dim). Note that
    n_kv_heads (not n_heads) is used — the shrinkage is exactly the GQA
    cache reduction.
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.k = torch.zeros(
            batch_size, n_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.v = torch.zeros_like(self.k)

    def write(self, new_k: torch.Tensor, new_v: torch.Tensor, offset: int) -> None:
        s = new_k.shape[-2]
        self.k[:, :, offset : offset + s] = new_k
        self.v[:, :, offset : offset + s] = new_v

    def read(self, total_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, :total_length], self.v[:, :, :total_length]


class LatentLayerCache:
    """Growing latent buffer for one MLA layer.

    Stores the per-token compressed KV latent c (B, max_seq_len, d_c) and,
    when decoupled RoPE is enabled, the single shared RoPE key slice
    (B, max_seq_len, d_rope). Compare against GQA's 2 * n_kv_heads *
    head_dim per token — the latent is the entire per-token memory here.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        d_latent: int,
        d_rope: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.c = torch.zeros(batch_size, max_seq_len, d_latent, device=device, dtype=dtype)
        self.k_rope = (
            torch.zeros(batch_size, max_seq_len, d_rope, device=device, dtype=dtype)
            if d_rope > 0
            else None
        )

    def write(self, new_c: torch.Tensor, new_k_rope: torch.Tensor | None, offset: int) -> None:
        s = new_c.shape[-2]
        self.c[:, offset : offset + s] = new_c
        if self.k_rope is not None:
            self.k_rope[:, offset : offset + s] = new_k_rope

    def read(self, total_length: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        k_rope = self.k_rope[:, :total_length] if self.k_rope is not None else None
        return self.c[:, :total_length], k_rope


class KDALayerCache:
    """Fixed-size recurrent state for one KDA layer.

    `state` is the per-head delta-rule memory S ∈ (B, H, d_k, d_v), kept in
    fp32 for numerical stability. `conv_q/k/v` hold the last (kernel - 1)
    inputs of each short-convolution window so decoding one token at a
    time produces the same convolution outputs as one long prefill.
    """

    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        conv_kernel: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.state = torch.zeros(batch_size, n_heads, d_k, d_v, device=device, dtype=torch.float32)
        tail = conv_kernel - 1
        self.conv_q = torch.zeros(batch_size, n_heads * d_k, tail, device=device, dtype=dtype)
        self.conv_k = torch.zeros(batch_size, n_heads * d_k, tail, device=device, dtype=dtype)
        self.conv_v = torch.zeros(batch_size, n_heads * d_v, tail, device=device, dtype=dtype)


class ModelCache:
    """One layer-cache per attention layer, plus the shared length counter.

    Models build this from their own attention modules:

        ModelCache([attn.make_cache(batch_size, device) for attn in ...])

    The model calls `advance(s)` once per forward pass after all layers
    have written their new entries.
    """

    def __init__(self, layers: list):
        self.layers = layers
        self.length = 0

    def advance(self, s: int) -> None:
        self.length += s
