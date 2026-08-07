"""Stable LatentMoE — Kimi K3 (§2.3).

K3 scales to 896 routed experts with 16 active by making three changes to
the DeepSeek-style MoE:

  Latent routed experts — routed experts don't operate at full model
    width d. The token is first projected into a latent z = W↓x ∈ R^ℓ
    (K3: ℓ = d/2); routed experts are small GLU MLPs ℓ → hidden → ℓ; the
    aggregate is mapped back with W↑. Dispatch traffic and per-expert
    parameters scale with ℓ instead of d, which is what makes the huge
    expert pool affordable. Shared experts keep a full-width path for
    common transformations:
        u = Σ_{i ∈ topk} p_i · E_i(W↓ x)
        y = Σ_j E_j_shared(x) + W↑ RMSNorm(u)

  Normalized aggregation — the RMSNorm between expert aggregation and
    up-projection ("Normalized LatentMoE"). The routed sum's scale varies
    with which experts fire and their weights; without the norm, W↓ /
    experts / W↑ form a chain of near-consecutive matmuls whose
    activations exploded at K3's scale.

  Quantile Balancing (QB) — aux-loss-free routing bias like DeepSeek's,
    but instead of nudging each bias by a fixed γ·sign(error) step, QB
    *solves* for the bias that gives each expert exactly its target load,
    using one quantile per expert over the batch's routing margins:
        margin_ij = s_ij − cutoff_i   (cutoff_i = (k+1)-th largest biased
                                       score of token i — what expert j
                                       must beat to enter token i's top-k)
        b_j ← −quantile_{1−k/E}(margin_:,j),  then center b to mean 0.
    Setting the bias at the (1−k/E) quantile means exactly a k/E fraction
    of tokens clear the bar for each expert — perfectly balanced load in
    one step, no slow-adaptation/oscillation trade-off. Selection uses the
    PREVIOUS step's bias (a batch is never routed with a bias derived from
    itself); the bias is frozen at inference.

Experts use SiTU-GLU activations by default (see dense.py) — the third
stability component.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..norm import RMSNorm
from .dense import GatedMLP
from .routing import run_topk_experts


class StableLatentMoE(nn.Module):
    """Latent routed experts + normalized aggregation + Quantile Balancing."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        n_shared_experts: int = 2,
        shared_expert_hidden: int = 1024,
        *,
        activation: str = "situ",
        route_scale: float = 1.0,
    ):
        super().__init__()
        assert top_k < n_experts, "top_k must be < n_experts"
        self.top_k = top_k
        self.n_experts = n_experts
        self.route_scale = route_scale
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w_down = nn.Linear(d_model, latent_dim, bias=False)
        self.w_up = nn.Linear(latent_dim, d_model, bias=False)
        self.u_norm = RMSNorm(latent_dim)
        self.experts = nn.ModuleList([
            GatedMLP(latent_dim, expert_hidden, activation) for _ in range(n_experts)
        ])
        self.shared_experts = nn.ModuleList([
            GatedMLP(d_model, shared_expert_hidden, activation)
            for _ in range(n_shared_experts)
        ])
        self.register_buffer("route_bias", torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        scores = torch.sigmoid(self.router(x_flat).float())        # (T, E)
        topk_idx = (scores + self.route_bias).topk(self.top_k, dim=-1).indices
        topk_scores = scores.gather(-1, topk_idx)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = (topk_weights * self.route_scale).to(x.dtype)

        # Routed path in the latent space.
        z = self.w_down(x_flat)                                    # (T, ℓ)
        u = run_topk_experts(z, self.experts, topk_idx, topk_weights, z.shape[-1])
        out = self.w_up(self.u_norm(u))

        # Full-width shared path.
        for shared in self.shared_experts:
            out = out + shared(x_flat)

        if self.training:
            self._quantile_balance(scores)

        return out.view(B, S, D)

    @torch.no_grad()
    def _quantile_balance(self, scores: torch.Tensor) -> None:
        """One-shot bias solve from this batch's routing margins (K3 Eq. 14)."""
        k, E = self.top_k, self.n_experts
        # Cutoff per token: the (k+1)-th largest biased score — the bar an
        # expert must clear to enter this token's top-k.
        biased = scores + self.route_bias
        cutoff = biased.topk(k + 1, dim=-1).values[:, -1]          # (T,)
        margins = scores - cutoff.unsqueeze(-1)                    # (T, E)
        new_bias = -torch.quantile(margins, 1.0 - k / E, dim=0)    # (E,)
        self.route_bias.copy_(new_bias - new_bias.mean())
