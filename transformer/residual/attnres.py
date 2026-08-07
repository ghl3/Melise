"""Attention Residuals (AttnRes) — Kimi K3 (§2.2).

A standard pre-norm residual stream accumulates every module's output
into one running sum:

    h_l = embedding + f_1 + f_2 + ... + f_{l−1}

— all prior information compressed into a single state, weighted
uniformly, a bottleneck over DEPTH reminiscent of what RNNs were over
time. AttnRes applies the transformer's own medicine to depth: each
module's input becomes an attention-weighted combination over the
embedding and all preceding module outputs (K3 Eq. 8–9):

    α_{i→l} = softmax_i( q_lᵀ · rmsnorm(v_i) )        v_0 = embedding,
    h_l     = Σ_{i<l} α_{i→l} · v_i                    v_i = f_i(h_i)

where q_l is a learnable per-module "pseudo-query" (one d-vector per
module — the α are per-token scalars, since the keys v_i are per-token).
The rmsnorm on keys stops modules with large-magnitude outputs from
dominating selection; values are left un-normalized.

This is the FULL form of AttnRes — O(L²·d) arithmetic over module count
L, affordable at small depth (K3 itself partitions 93 layers into 8
blocks of summed outputs to cut the overhead; we don't need that here).
Following K3's Figure 2, every module — each attention layer AND each
FFN — is a stage with its own pseudo-query, and one final pseudo-query
aggregates all outputs for the LM head.

Depth-attention is per-token, so it composes with any KV cache without
extra state: each decoded token mixes its own stage outputs.

Pseudo-queries init at zero → uniform α → each module starts by seeing
the plain AVERAGE of prior outputs and learns to specialize from there.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AttentionResiduals(nn.Module):
    """Learnable pseudo-queries over module outputs (full AttnRes).

    Memory note: the mix stacks all preceding outputs, and doing that at
    every stage would retain O(L²·B·S·D) of autograd intermediates — at
    depth 35 that is tens of GB. The mixing math is trivially cheap, so
    during training we gradient-checkpoint it: backward recomputes the
    stack/normalize/softmax instead of storing them, keeping the true
    cost at the O(L·B·S·D) of the stage outputs themselves (which the
    architecture needs alive regardless).
    """

    def __init__(self, d_model: int, n_stages: int):
        super().__init__()
        # Row l mixes the inputs for stage l; row n_stages builds the final
        # representation fed to the LM head.
        self.queries = nn.Parameter(torch.zeros(n_stages + 1, d_model))

    def forward(self, stage_idx: int, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Mix `outputs` (embedding + module outputs so far) for one stage.

        outputs: list of (B, S, D). Returns (B, S, D).
        """
        if torch.is_grad_enabled() and self.training:
            return checkpoint(self._mix, stage_idx, *outputs, use_reentrant=False)
        return self._mix(stage_idx, *outputs)

    def _mix(self, stage_idx: int, *outputs: torch.Tensor) -> torch.Tensor:
        stack = torch.stack(outputs)                               # (n, B, S, D)
        q = self.queries[stage_idx].float()

        # Softmax kernel over depth, computed in fp32: keys are the
        # rms-normalized outputs, values the raw outputs.
        x32 = stack.float()
        keys = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        logits = (keys * q).sum(dim=-1)                            # (n, B, S)
        alpha = torch.softmax(logits, dim=0)

        return (alpha.unsqueeze(-1) * x32).sum(dim=0).to(stack.dtype)
