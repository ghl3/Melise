"""Attention mechanisms, their caches, and RoPE.

Interchangeable attention modules — plain nn.Modules with explicit
constructor arguments:

    CausalSelfAttention — grouped-query attention + RoPE + SDPA (plain MHA
                          when n_kv_heads == n_heads). Cache: per-token K/V.
    MultiLatentAttention — MLA (DeepSeek-V2/V3, Kimi K2/K3), with decoupled
                          RoPE (DeepSeek) or NoPE + output gate (Kimi K3).
                          Cache: per-token latent.
    KimiDeltaAttention  — KDA (Kimi Linear / K3): channel-wise gated
                          delta-rule linear attention. Cache: fixed-size
                          recurrent state.

Each module builds its own layer cache via `make_cache(batch_size,
device)`; a model collects those into a ModelCache (see cache.py).
"""

from .cache import KDALayerCache, KVLayerCache, LatentLayerCache, ModelCache
from .gqa import CausalSelfAttention
from .kda import KimiDeltaAttention
from .mla import MultiLatentAttention
from .rope import apply_rotary, precompute_rotary

__all__ = [
    "CausalSelfAttention",
    "KimiDeltaAttention",
    "MultiLatentAttention",
    "KDALayerCache",
    "KVLayerCache",
    "LatentLayerCache",
    "ModelCache",
    "apply_rotary",
    "precompute_rotary",
]
