"""DeepSeek-V3 in miniature: MLA + sigmoid-routed MoE with bias balancing.

    token embedding
      → block 0:   [pre-norm MLA, pre-norm dense SwiGLU]      ← dense first layer
      → blocks 1+: [pre-norm MLA, pre-norm DeepSeekMoE]
      → norm → head

The two DeepSeek signatures:

  MLA with decoupled RoPE — the KV cache stores a small latent (plus one
    shared RoPE key slice) instead of per-head K/V; see attention/mla.py.

  DeepSeekMoE — sigmoid routing with an aux-loss-free balancing bias and
    always-on shared expert; see ffn/deepseek_moe.py. The first layer
    keeps a dense FFN (as DeepSeek-V3 and Kimi K2/K3 do) because routing
    right off the embedding is unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..attention import ModelCache, MultiLatentAttention
from ..ffn import DeepSeekMoE, GatedMLP
from ..norm import RMSNorm


@dataclass
class DeepSeekConfig:
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 6
    max_seq_len: int = 512
    # MLA. Cache per token: kv_lora_rank + qk_rope_head_dim floats
    # (vs 2 * n_kv_heads * head_dim = 256 for the base model's GQA).
    n_heads: int = 16
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 32
    qk_rope_head_dim: int = 16
    v_head_dim: int = 32
    # FFN / MoE.
    dense_hidden: int = 2048        # layer 0's dense FFN
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    n_shared_experts: int = 1
    shared_expert_hidden: int = 1024
    bias_update_rate: float = 1e-3  # balancing-bias step γ
    dtype: torch.dtype = torch.bfloat16

    # Which tokenizer this model's ids come from: "bytes" (the 256
    # byte values) or a trained artifact name like "bpe4k"
    # (transformer.tokenizer.load_tokenizer resolves it). Serialized into
    # checkpoints via the config; read with getattr(cfg, "tokenizer",
    # "bytes") for pre-field checkpoints.
    tokenizer: str = "bytes"
    rope_base: float = 10000.0


class DeepSeekBlock(nn.Module):
    """Pre-norm MLA + pre-norm FFN (dense on layer 0, MoE elsewhere)."""

    def __init__(self, cfg: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiLatentAttention(
            cfg.d_model, cfg.n_heads, cfg.kv_lora_rank,
            cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim,
            cfg.max_seq_len,
            layer_idx=layer_idx, rope=True, gated=False, rope_base=cfg.rope_base,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        if layer_idx == 0:
            self.ffn = GatedMLP(cfg.d_model, cfg.dense_hidden, activation="swiglu")
        else:
            self.ffn = DeepSeekMoE(
                cfg.d_model, cfg.n_experts, cfg.top_k, cfg.expert_hidden,
                cfg.n_shared_experts, cfg.shared_expert_hidden,
                activation="swiglu", bias_update_rate=cfg.bias_update_rate,
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache)
        return x + self.ffn(self.norm2(x))


class DeepSeekLM(nn.Module):
    """token embedding → DeepSeekBlocks → final norm → LM head."""

    def __init__(self, cfg: DeepSeekConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([DeepSeekBlock(cfg, i) for i in range(cfg.n_layers)])
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
