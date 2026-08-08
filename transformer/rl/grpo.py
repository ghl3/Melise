"""GRPO math: group advantages, rollout tensorization, clipped loss.

Group Relative Policy Optimization (DeepSeekMath, arXiv:2402.03300).
The three pieces a training step composes, in order:

    adv  = group_advantages(rewards)          # (P, G) -> (P·G,)
    batch = pad_rollouts(seqs)                # ragged rollouts -> tensors
    loss, stats = grpo_loss(new_lp, batch.old_lp, ref_lp, adv, ...)

The group baseline (z-score within each prompt's G completions) replaces
PPO's learned value network. The KL term uses the k3 estimator
exp(d) − d − 1 with d = ref_lp − new_lp: unbiased, always ≥ 0, and
low-variance — the standard choice for RLHF-style KL penalties.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """(P, G) rewards -> (P·G,) advantages, z-scored within each group.

    A group with identical rewards (all failed, or all perfect) gets
    advantage ≈ 0 everywhere — no gradient, by design: GRPO only learns
    from within-group contrast.
    """
    adv = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, keepdim=True) + eps
    )
    return adv.reshape(-1)


@dataclass
class RolloutBatch:
    """Padded tensors for one step's rollouts (all CPU; move as needed).

    ids     (N, L)  prompt+completion token ids, right-padded with 0
    tok     (N, T)  completion tokens only
    pos     (N, T)  index into ids of the logit that predicts tok[:, t]
    mask    (N, T)  True where t < completion length
    old_lp  (N, T)  rollout-time log-probs of tok
    """

    ids: torch.Tensor
    tok: torch.Tensor
    pos: torch.Tensor
    mask: torch.Tensor
    old_lp: torch.Tensor

    @property
    def total_tokens(self) -> int:
        return int(self.mask.sum().item())

    def to(self, device) -> "RolloutBatch":
        return RolloutBatch(*(t.to(device) for t in
                              (self.ids, self.tok, self.pos, self.mask, self.old_lp)))


def pad_rollouts(seqs) -> RolloutBatch:
    """seqs: list of (prompt_bytes, completion_ids, old_lp_row) — one entry
    per rollout, groups flattened in order (grpo_loss's advantage vector
    must use the same order)."""
    N = len(seqs)
    pls = [len(p) for p, _, _ in seqs]
    cls = [len(c) for _, c, _ in seqs]
    Lmax, Tmax = max(p + c for p, c in zip(pls, cls)), max(cls)
    ids = torch.zeros(N, Lmax, dtype=torch.long)
    tok = torch.zeros(N, Tmax, dtype=torch.long)
    pos = torch.zeros(N, Tmax, dtype=torch.long)
    mask = torch.zeros(N, Tmax, dtype=torch.bool)
    old_lp = torch.zeros(N, Tmax)
    for i, (p, c, lp) in enumerate(seqs):
        pl, cl = pls[i], cls[i]
        ids[i, :pl] = torch.tensor(list(p))
        ids[i, pl : pl + cl] = torch.tensor(c)
        tok[i, :cl] = torch.tensor(c)
        pos[i, :cl] = torch.arange(pl - 1, pl - 1 + cl)
        mask[i, :cl] = True
        old_lp[i, :cl] = lp
    return RolloutBatch(ids, tok, pos, mask, old_lp)


def grpo_loss(new_lp, old_lp, ref_lp, adv, mask, *, clip_eps: float,
              kl_coef: float, total_tokens: int):
    """Clipped surrogate + KL-to-reference for one microbatch.

    Normalized by the STEP's total completion tokens (not the
    microbatch's), so microbatched backward passes accumulate to exactly
    the full-batch gradient. Returns (loss, stats) with stats holding
    the microbatch's summed kl and clipped-token count.
    """
    ratio = torch.exp(new_lp - old_lp)
    a = adv.unsqueeze(1)
    surr = torch.minimum(
        ratio * a,
        ratio.clamp(1 - clip_eps, 1 + clip_eps) * a,
    )
    d = ref_lp - new_lp
    kl = torch.exp(d) - d - 1  # k3 estimator, always ≥ 0
    loss = -((surr - kl_coef * kl) * mask).sum() / total_tokens
    with torch.no_grad():
        stats = {
            "kl": float((kl * mask).sum().item()),
            "clipped": float(((ratio - 1).abs() > clip_eps)[mask].sum().item()),
        }
    return loss, stats
