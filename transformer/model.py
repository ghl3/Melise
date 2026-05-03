"""The full transformer language model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import Block
from .cache import KVCache
from .config import Config
from .norm import RMSNorm


class TransformerLM(nn.Module):
    """Decoder-only transformer language model.

    Architecture:
        token embedding → N transformer blocks → final norm → LM head

    No additive positional embeddings — RoPE is applied inside attention.

    Forward shape: (B, S) int token IDs → (B, S, vocab_size) logits.
    Same forward function for both training (no kv_cache) and inference
    (with kv_cache).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(token_ids)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(token_ids.shape[1])
        return self.lm_head(self.final_norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
