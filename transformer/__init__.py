"""Transformer language model package.

A component library plus explicit model definitions:

    attention/  CausalSelfAttention (GQA) | MultiLatentAttention (MLA) |
                KimiDeltaAttention (KDA), RoPE, per-layer caches
    ffn/        GatedMLP (SwiGLU / SiTU-GLU) | MoE | DeepSeekMoE |
                StableLatentMoE
    norm/       RMSNorm
    residual/   AttentionResiduals (Kimi K3 depth attention)
    models/     one file per architecture, each laying out its stack
                explicitly: base (GQA + MoE), vanilla (MHA + dense),
                deepseek (MLA + DeepSeekMoE), kimi3 (KDA/MLA hybrid +
                AttnRes + LatentMoE)

Every model exposes the same interface: `forward(token_ids, kv_cache)`,
`new_cache(batch_size, device)`, `num_parameters()`, and `.cfg` — so
generate() and the training script work with all of them.

Quick start:

    import torch
    from transformer import Kimi3Config, Kimi3LM, generate

    cfg = Kimi3Config()
    model = Kimi3LM(cfg).to("mps").to(cfg.dtype)
    ids = torch.tensor([[ord(c) for c in "hello"]])
    out = generate(model, ids.to("mps"), max_new_tokens=20)
"""

from .attention import (
    CausalSelfAttention,
    KimiDeltaAttention,
    ModelCache,
    MultiLatentAttention,
    apply_rotary,
    precompute_rotary,
)
from .config import Config
from .ffn import DeepSeekMoE, GatedMLP, MoE, StableLatentMoE, SwiGLUExpert
from .generate import generate, greedy_sample
from .models import (
    MODELS,
    DeepSeekConfig,
    DeepSeekLM,
    Kimi3Config,
    Kimi3LM,
    TransformerLM,
    VanillaConfig,
    VanillaLM,
    build_model,
)
from .norm import RMSNorm
from .residual import AttentionResiduals

# Pre-refactor name for the inference cache container.
KVCache = ModelCache

__all__ = [
    "AttentionResiduals",
    "CausalSelfAttention",
    "Config",
    "DeepSeekConfig",
    "DeepSeekLM",
    "DeepSeekMoE",
    "GatedMLP",
    "KVCache",
    "Kimi3Config",
    "Kimi3LM",
    "KimiDeltaAttention",
    "MODELS",
    "MoE",
    "ModelCache",
    "MultiLatentAttention",
    "RMSNorm",
    "StableLatentMoE",
    "SwiGLUExpert",
    "TransformerLM",
    "VanillaConfig",
    "VanillaLM",
    "apply_rotary",
    "build_model",
    "generate",
    "greedy_sample",
    "precompute_rotary",
]
