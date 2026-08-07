"""Feedforward networks: dense gated MLPs and MoE variants.

Interchangeable FFN modules — plain nn.Modules with explicit constructor
arguments:

    GatedMLP        — one gated MLP (SwiGLU or SiTU-GLU); the dense FFN.
    MoE             — Mixtral-style: softmax top-k + 1 shared expert.
    DeepSeekMoE     — DeepSeek-V3 / Kimi K2: sigmoid routing, aux-loss-free
                      bias (sign update), shared experts.
    StableLatentMoE — Kimi K3: latent-width routed experts, normalized
                      aggregation, Quantile Balancing, SiTU-GLU.
"""

from .deepseek_moe import DeepSeekMoE
from .dense import GatedMLP, SwiGLUExpert
from .latent_moe import StableLatentMoE
from .moe import MoE
from .routing import run_topk_experts

__all__ = [
    "DeepSeekMoE",
    "GatedMLP",
    "MoE",
    "StableLatentMoE",
    "SwiGLUExpert",
    "run_topk_experts",
]
