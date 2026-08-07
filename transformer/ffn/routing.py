"""Shared MoE dispatch: run each token through its selected experts.

All three MoE variants route the same way once (indices, weights) are
chosen: loop over experts (not over tokens) and batch each expert's
selected tokens through it in one matmul. This is the standard
vectorization used in real MoE implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def run_topk_experts(
    x: torch.Tensor,          # (T, d_in) — flattened tokens (or latents)
    experts: nn.ModuleList,   # each: (n, d_in) -> (n, d_out)
    topk_idx: torch.Tensor,   # (T, K) int — chosen expert per slot
    topk_weights: torch.Tensor,  # (T, K) — weight per slot, in x.dtype
    d_out: int,
) -> torch.Tensor:
    """Weighted sum of expert outputs per token, shape (T, d_out)."""
    out = x.new_zeros(x.shape[0], d_out)
    for expert_idx, expert in enumerate(experts):
        chosen = topk_idx == expert_idx                 # (T, K) bool
        token_has_expert = chosen.any(dim=-1)           # (T,) bool
        if not token_has_expert.any():
            continue
        token_ids = token_has_expert.nonzero(as_tuple=True)[0]
        # Per-token weight on this expert (sum across top-k slots in case
        # the same expert appears twice — rare but possible).
        w = (chosen[token_ids].to(x.dtype) * topk_weights[token_ids]).sum(dim=-1)
        out[token_ids] += w.unsqueeze(-1) * expert(x[token_ids])
    return out
