"""Causal self-attention with GQA, RoPE, and SDPA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVLayerCache, ModelCache
from .rope import apply_rotary, precompute_rotary


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Components:
      - Grouped-query attention (GQA): n_q_heads queries, n_kv_heads keys/values.
        Each KV head is shared by n_heads / n_kv_heads query heads.
        With n_kv_heads == n_heads this is plain MHA (the vanilla model).
      - Rotary position embedding (RoPE): rotate Q and K by absolute position
        inside attention. V is left unrotated.
      - SDPA: PyTorch's fused attention kernel
        (FlashAttention or platform equivalent), with `enable_gqa=True` so the
        kernel handles the K/V repetition across query-head groups internally.

    Forward works for both training (no kv_cache) and inference (with kv_cache).
    `layer_idx` selects this layer's slot in the ModelCache.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        *,
        layer_idx: int = 0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "head_dim must be even (RoPE rotates pairs)"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx

        # Q at full n_heads; K and V at smaller n_kv_heads (GQA).
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE tables, stored fp32 (cast to the running dtype at apply time).
        cos, sin = precompute_rotary(head_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Causal bool mask, sliced per call to handle prefill and decode shapes.
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def make_cache(self, batch_size: int, device: torch.device) -> KVLayerCache:
        return KVLayerCache(
            batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim,
            device, self.q_proj.weight.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, H_kv, Dh = self.n_heads, self.n_kv_heads, self.head_dim

        # Project to Q, K, V. Q has H heads; K, V have H_kv (GQA).
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)         # (B, H,    S, Dh)
        kv = self.kv_proj(x).view(B, S, 2, H_kv, Dh)
        k = kv[:, :, 0].transpose(1, 2)                               # (B, H_kv, S, Dh)
        v = kv[:, :, 1].transpose(1, 2)

        # RoPE on Q and K only; use the absolute-position slice based on the
        # current cache length (0 during training, cache.length during decode).
        pos_offset = kv_cache.length if kv_cache is not None else 0
        cos = self.rope_cos[pos_offset : pos_offset + S]
        sin = self.rope_sin[pos_offset : pos_offset + S]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Cache write/read. K, V get extended; Q is for the new tokens only.
        if kv_cache is not None:
            layer = kv_cache.layers[self.layer_idx]
            layer.write(k, v, pos_offset)
            L = pos_offset + S
            k, v = layer.read(L)
        else:
            L = S

        # Build an additive mask for SDPA. Works for square (prefill) and
        # rectangular (decode) shapes uniformly.
        bool_mask = self.causal_mask[pos_offset : pos_offset + S, :L]
        attn_mask = torch.zeros(S, L, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(~bool_mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)
