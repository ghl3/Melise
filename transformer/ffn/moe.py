"""Mixtral-style Mixture of Experts (the repo's original MoE).

Routed experts with softmax top-k selection plus a single always-on
shared expert. Each expert is a SwiGLU feedforward. Parameter names
(router / experts / shared_expert) match the original flat moe.py so old
checkpoints keep loading.
"""

import torch
import torch.nn as nn

from .dense import SwiGLUExpert
from .routing import run_topk_experts


class MoE(nn.Module):
    """Top-k routed experts plus one always-on shared expert.

    For each token:
      1. Router scores all routed experts.
      2. Select top-k. Softmax-normalize their logits into weights.
      3. Run each chosen expert; sum weighted outputs.
      4. Add the shared expert's output (run unconditionally on every token).

    Softmax-over-top-k routing with no explicit load balancing — compare
    DeepSeekMoE (sigmoid + bias sign-update) and StableLatentMoE
    (sigmoid + Quantile Balancing).
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        shared_expert_hidden: int,
    ):
        super().__init__()
        assert top_k < n_experts, "top_k must be < n_experts"
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(d_model, expert_hidden) for _ in range(n_experts)
        ])
        self.shared_expert = SwiGLUExpert(d_model, shared_expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Routed experts. Run the router in fp32 for stable softmax even when
        # the model dtype is bf16; cast weights back to model dtype before use.
        router_logits = self.router(x_flat).float()
        topk_logits, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1).to(x.dtype)

        out = run_topk_experts(x_flat, self.experts, topk_idx, topk_weights, D)

        # Shared expert: always on, applied to every token.
        out = out + self.shared_expert(x_flat)

        return out.view(B, S, D)
