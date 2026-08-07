"""DeepSeek-V3-style MoE (also Kimi K2): sigmoid routing, aux-loss-free
bias balancing, shared expert(s).

Differences from the Mixtral-style MoE:

  Sigmoid scoring — each expert gets an independent affinity in (0,1)
    instead of competing in one softmax. Top-k weights are the selected
    scores renormalized to sum to 1 (times an optional route_scale).

  Aux-loss-free load balancing — a per-expert bias b is ADDED FOR
    SELECTION ONLY (never in the mixture weights, so it doesn't touch the
    gradient path). After each training batch the bias takes a fixed-size
    step: overloaded experts get pushed down, underloaded ones up:
        b_j ← b_j + γ · sign(target_load − load_j)
    DeepSeek-V3 introduced this to replace auxiliary balancing losses,
    which trade balance against model quality. The bias is a persistent
    buffer (saved in checkpoints) and frozen at inference.

Kimi K3's StableLatentMoE keeps this selection rule but replaces the
sign-update with Quantile Balancing — see latent_moe.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dense import GatedMLP
from .routing import run_topk_experts


class DeepSeekMoE(nn.Module):
    """Sigmoid top-k routing with bias-based aux-loss-free balancing."""

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        n_shared_experts: int = 1,
        shared_expert_hidden: int = 1024,
        *,
        activation: str = "swiglu",
        route_scale: float = 1.0,
        bias_update_rate: float = 1e-3,
    ):
        super().__init__()
        assert top_k < n_experts, "top_k must be < n_experts"
        self.top_k = top_k
        self.n_experts = n_experts
        self.route_scale = route_scale
        self.bias_update_rate = bias_update_rate
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            GatedMLP(d_model, expert_hidden, activation) for _ in range(n_experts)
        ])
        self.shared_experts = nn.ModuleList([
            GatedMLP(d_model, shared_expert_hidden, activation)
            for _ in range(n_shared_experts)
        ])
        # Selection-only bias; fp32 buffer so it checkpoints and resumes.
        self.register_buffer("route_bias", torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        scores = torch.sigmoid(self.router(x_flat).float())        # (T, E)
        # Bias steers WHICH experts are chosen ...
        topk_idx = (scores + self.route_bias).topk(self.top_k, dim=-1).indices
        # ... but weights come from the raw scores only.
        topk_scores = scores.gather(-1, topk_idx)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = (topk_weights * self.route_scale).to(x.dtype)

        out = run_topk_experts(x_flat, self.experts, topk_idx, topk_weights, D)
        for shared in self.shared_experts:
            out = out + shared(x_flat)

        if self.training and self.bias_update_rate > 0:
            self._update_bias(topk_idx)

        return out.view(B, S, D)

    @torch.no_grad()
    def _update_bias(self, topk_idx: torch.Tensor) -> None:
        """Fixed-step sign update toward uniform expert load (DeepSeek-V3)."""
        counts = torch.zeros(self.n_experts, device=topk_idx.device)
        counts.scatter_add_(
            0, topk_idx.reshape(-1),
            torch.ones_like(topk_idx.reshape(-1), dtype=counts.dtype),
        )
        load = counts / counts.sum()
        target = 1.0 / self.n_experts
        self.route_bias += self.bias_update_rate * torch.sign(target - load)
