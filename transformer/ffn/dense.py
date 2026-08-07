"""Gated MLPs: SwiGLU and SiTU-GLU.

Every feedforward in this repo — the dense FFN, MoE routed experts, MoE
shared experts — is a gated MLP with three linears (gate, up, down) and
one of two activations:

  SwiGLU (Llama, DeepSeek, Kimi K2):
      down( silu(gate(x)) * up(x) )
      where silu(g) = g * sigmoid(g) (a.k.a. Swish).

  SiTU-GLU (Kimi K3, §2.3.2): both multiplicative factors in SwiGLU are
  unbounded, so coincident large coordinates can overflow low-precision
  arithmetic. SiTU-GLU soft-caps the *linear* factor of the Swish gate
  and the up branch with softcap(x, β) = β·tanh(x/β):
      down( [β₁·tanh(gate(x)/β₁) * sigmoid(gate(x))] * [β₂·tanh(up(x)/β₂)] )
  Near the origin tanh is ~identity, so SiTU-GLU behaves like SwiGLU; for
  large inputs the product is bounded by β₁β₂. K3 uses β₁ = 4, β₂ = 25.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMLP(nn.Module):
    """Three-linear gated MLP; `activation` picks SwiGLU or SiTU-GLU.

    `d_out` defaults to `d_in` — MoE latent experts pass the latent width
    for both.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int,
        activation: str = "swiglu",
        d_out: int | None = None,
        situ_beta_gate: float = 4.0,
        situ_beta_up: float = 25.0,
    ):
        super().__init__()
        assert activation in ("swiglu", "situ"), f"unknown activation {activation!r}"
        d_out = d_in if d_out is None else d_out
        self.gate = nn.Linear(d_in, hidden, bias=False)
        self.up = nn.Linear(d_in, hidden, bias=False)
        self.down = nn.Linear(hidden, d_out, bias=False)
        self.activation = activation
        self.beta_gate = situ_beta_gate
        self.beta_up = situ_beta_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        u = self.up(x)
        if self.activation == "swiglu":
            return self.down(F.silu(g) * u)
        b1, b2 = self.beta_gate, self.beta_up
        gate_branch = b1 * torch.tanh(g / b1) * torch.sigmoid(g)   # soft-capped Swish
        up_branch = b2 * torch.tanh(u / b2)                        # soft-capped linear
        return self.down(gate_branch * up_branch)


class SwiGLUExpert(GatedMLP):
    """One SwiGLU MLP with the classic name — kept for checkpoint and API
    compatibility with the original moe.py (same gate/up/down parameters)."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__(d_model, hidden, activation="swiglu")
