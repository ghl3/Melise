"""Model configuration for the base model (models/base.py).

This is the repo's original Config, unchanged: it describes the GQA +
RoPE + Mixtral-style-MoE architecture that existing checkpoints were
trained with. The other architectures each carry their own config
dataclass in their model file (models/vanilla.py, models/deepseek.py,
models/kimi3.py) — a config here is just the knobs one architecture
exposes, not a description of which components to assemble.
"""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    """All hyperparameters for a TransformerLM.

    Defaults match the design we settled on in tutorial/14_final_model.py:
    GQA + RoPE + Mixtral-style MoE with a shared expert + bf16.
    """

    # Vocabulary and shape.
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 4
    max_seq_len: int = 512

    # Attention (GQA: n_q_heads = n_heads, n_kv_heads <= n_heads).
    n_heads: int = 16
    n_kv_heads: int = 4

    # MoE.
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    shared_expert_hidden: int = 1024

    # Storage / compute dtype for weights and activations.
    dtype: torch.dtype = torch.bfloat16

    # RoPE frequency base.
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even (RoPE rotates pairs)"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
