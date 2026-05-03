"""One transformer block: pre-norm attention + pre-norm MoE, with residuals."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .cache import KVCache
from .config import Config
from .moe import MoE
from .norm import RMSNorm


class Block(nn.Module):
    """Sequential Pre-RMSNorm transformer block.

        attn_delta = attn(norm1(x))
        moe_delta  = moe(norm2(x + attn_delta))
        return x + attn_delta + moe_delta

    The MoE sees the post-attention residual stream (sequential, not parallel),
    so attention's contribution flows into the MoE's input within the same block.
    """

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        moe_delta = self.moe(self.norm2(x + attn_delta))
        return x + attn_delta + moe_delta
