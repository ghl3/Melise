"""Causal self-attention with GQA, RoPE, and SDPA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .config import Config
from .rope import apply_rotary, precompute_rotary


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Components:
      - Grouped-query attention (GQA): n_q_heads queries, n_kv_heads keys/values.
        Each KV head is shared by n_heads / n_kv_heads query heads.
      - Rotary position embedding (RoPE): rotate Q and K by absolute position
        inside attention. V is left unrotated.
      - SDPA: PyTorch's fused attention kernel
        (FlashAttention or platform equivalent), with `enable_gqa=True` so the
        kernel handles the K/V repetition across query-head groups internally.

    Forward works for both training (no kv_cache) and inference (with kv_cache).
    """

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        # Q at full n_heads; K and V at smaller n_kv_heads (GQA).
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # RoPE tables, cast to model dtype so multiplications stay in-precision.
        cos, sin = precompute_rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos.to(cfg.dtype), persistent=False)
        self.register_buffer("rope_sin", sin.to(cfg.dtype), persistent=False)

        # Causal bool mask, sliced per call to handle prefill and decode shapes.
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, H_kv, Dh = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim

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
            kv_cache.write(self.layer_idx, k, v)
            L = pos_offset + S
            k, v = kv_cache.read(self.layer_idx, L)
        else:
            L = S

        # Build an additive mask for SDPA. Works for square (prefill) and
        # rectangular (decode) shapes uniformly.
        bool_mask = self.causal_mask[pos_offset : pos_offset + S, :L]
        attn_mask = torch.zeros(S, L, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(~bool_mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)

        out = out.transpose(1, 2).contiguous().view(B, S, self.cfg.d_model)
        return self.o_proj(out)
