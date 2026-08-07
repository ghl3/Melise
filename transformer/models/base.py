"""The repo's original model: GQA + RoPE + Mixtral-style MoE.

    token embedding → N × [pre-norm GQA attention, pre-norm MoE] → norm → head

This is the architecture every pre-refactor checkpoint was trained with;
class name (TransformerLM), config (transformer.config.Config), and
parameter names (blocks.N.norm1 / attn / norm2 / moe) are unchanged, so
those checkpoints load as-is.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..attention import CausalSelfAttention, ModelCache
from ..config import Config
from ..ffn import MoE
from ..norm import RMSNorm


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
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len,
            layer_idx=layer_idx, rope_base=cfg.rope_base,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoE(
            cfg.d_model, cfg.n_experts, cfg.top_k,
            cfg.expert_hidden, cfg.shared_expert_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        moe_delta = self.moe(self.norm2(x + attn_delta))
        return x + attn_delta + moe_delta


class TransformerLM(nn.Module):
    """Decoder-only transformer language model.

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
