"""Mixture of Experts.

Routed experts with top-k selection plus a single always-on shared expert.
Each expert is a SwiGLU feedforward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class SwiGLUExpert(nn.Module):
    """One MoE expert: a small SwiGLU MLP.

        y = down( silu(gate(x)) * up(x) )
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Top-k routed experts plus one always-on shared expert.

    For each token:
      1. Router scores all routed experts.
      2. Select top-k. Softmax-normalize their logits into weights.
      3. Run each chosen expert; sum weighted outputs.
      4. Add the shared expert's output (run unconditionally on every token).

    Forward pattern: loop over experts (not over tokens) and batch the
    selecting tokens through each expert. This is the standard
    vectorization used in real MoE implementations.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(cfg.d_model, cfg.expert_hidden)
            for _ in range(cfg.n_experts)
        ])
        self.shared_expert = SwiGLUExpert(cfg.d_model, cfg.shared_expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Routed experts. Run the router in fp32 for stable softmax even when
        # the model dtype is bf16; cast weights back to model dtype before use.
        router_logits = self.router(x_flat).float()
        topk_logits, topk_idx = torch.topk(router_logits, k=self.cfg.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1).to(x.dtype)

        out = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            chosen = (topk_idx == expert_idx)               # (T, K) bool
            token_has_expert = chosen.any(dim=-1)           # (T,) bool
            if not token_has_expert.any():
                continue
            token_ids = token_has_expert.nonzero(as_tuple=True)[0]
            # Per-token weight on this expert (sum across top-k slots in case
            # the same expert appears twice — rare but possible).
            w = (chosen[token_ids].to(x.dtype) * topk_weights[token_ids]).sum(dim=-1)
            expert_out = expert(x_flat[token_ids])
            out[token_ids] += w.unsqueeze(-1) * expert_out

        # Shared expert: always on, applied to every token.
        out = out + self.shared_expert(x_flat)

        return out.view(B, S, D)
