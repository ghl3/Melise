"""Vanilla decoder-only transformer: MHA + RoPE + dense SwiGLU.

    token embedding → N × [pre-norm MHA, pre-norm SwiGLU MLP] → norm → head

The GPT/Llama shape with no architectural tricks: full multi-head
attention (every query head has its own K/V head) and one dense gated
MLP per block. The baseline the other architectures are measured against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..attention import CausalSelfAttention, ModelCache
from ..ffn import GatedMLP
from ..norm import RMSNorm


@dataclass
class VanillaConfig:
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 4
    max_seq_len: int = 512
    n_heads: int = 8
    ffn_hidden: int = 2048
    dtype: torch.dtype = torch.bfloat16
    rope_base: float = 10000.0


class VanillaBlock(nn.Module):
    """Pre-norm attention + pre-norm MLP, plain residual stream."""

    def __init__(self, cfg: VanillaConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        # n_kv_heads == n_heads → plain MHA (no grouped-query sharing).
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.n_heads, cfg.max_seq_len,
            layer_idx=layer_idx, rope_base=cfg.rope_base,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, cfg.ffn_hidden, activation="swiglu")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache)
        return x + self.mlp(self.norm2(x))


class VanillaLM(nn.Module):
    """token embedding → VanillaBlocks → final norm → LM head."""

    def __init__(self, cfg: VanillaConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([VanillaBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(token_ids)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(token_ids.shape[1])
        return self.lm_head(self.final_norm(x))

    def new_cache(self, batch_size: int, device: torch.device | str) -> ModelCache:
        device = torch.device(device)
        return ModelCache([b.attn.make_cache(batch_size, device) for b in self.blocks])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
