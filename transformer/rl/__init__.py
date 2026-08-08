"""Reinforcement learning on verifiable rewards (RLVR).

    tasks.py    the environment: task generators with deterministic
                scorers and canonical answers (no reward model)
    rollout.py  batched KV-cache policy rollouts + log-prob extraction,
                greedy eval over fixed task sets
    grpo.py     the GRPO math: group advantages, rollout tensorization,
                clipped surrogate + k3 KL loss

scripts/grpo.py composes these into a training run; scripts/gen_task_sft.py
turns the same tasks into cold-start SFT data.
"""

from .grpo import RolloutBatch, group_advantages, grpo_loss, pad_rollouts
from .rollout import eval_rewards, gather_completion_logprobs, rollout_group
from .tasks import TASKS, Task, sample_tasks

__all__ = [
    "RolloutBatch",
    "TASKS",
    "Task",
    "eval_rewards",
    "gather_completion_logprobs",
    "group_advantages",
    "grpo_loss",
    "pad_rollouts",
    "rollout_group",
    "sample_tasks",
]
