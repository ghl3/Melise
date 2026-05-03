"""Transformer language model package.

A clean, reusable decoder-only transformer with GQA, RoPE, FlashAttention
(via torch's SDPA), and Mixtral-style MoE with a shared expert.

The same `TransformerLM.forward` is used for both training (no kv_cache,
loss computed externally) and inference (with kv_cache, generation loop).

Quick start:

    from transformer import Config, TransformerLM, generate

    cfg = Config()
    model = TransformerLM(cfg).to("mps").to(cfg.dtype)
    ids = torch.tensor([[ord(c) for c in "hello"]])
    out = generate(model, ids.to("mps"), max_new_tokens=20)
"""

from .attention import CausalSelfAttention
from .block import Block
from .cache import KVCache
from .config import Config
from .generate import generate, greedy_sample
from .model import TransformerLM
from .moe import MoE, SwiGLUExpert
from .norm import RMSNorm
from .rope import apply_rotary, precompute_rotary

__all__ = [
    "Block",
    "CausalSelfAttention",
    "Config",
    "KVCache",
    "MoE",
    "RMSNorm",
    "SwiGLUExpert",
    "TransformerLM",
    "apply_rotary",
    "generate",
    "greedy_sample",
    "precompute_rotary",
]
