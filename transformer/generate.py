"""Inference helpers — greedy or custom-sampled autoregressive generation."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def greedy_sample(logits: torch.Tensor) -> int:
    """Default sampler: argmax."""
    return int(logits.argmax().item())


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    sample_fn: Callable[[torch.Tensor], int] = greedy_sample,
) -> list[int]:
    """Run prefill on `prompt_ids`, then decode up to `max_new_tokens` tokens.

    Args:
        model: any model from transformer.models (they all expose
            `new_cache` and `cfg.max_seq_len`). Should be in eval mode for
            stable behavior; we don't toggle it here.
        prompt_ids: (B, S) int64 token IDs. B is typically 1.
        max_new_tokens: maximum tokens to generate (cap).
        sample_fn: maps a (vocab_size,) logits tensor to a token id.
            Default is greedy argmax. Pass your own for temperature/top-k/top-p.

    Returns: list of generated token ids (length ≤ max_new_tokens).
    Stops early if the cache fills to `cfg.max_seq_len`. (KDA layers have
    no length limit — their state is O(1) — but GQA/MLA buffers and the
    causal masks are sized to max_seq_len, so the cap applies to all.)
    """
    cache = model.new_cache(prompt_ids.shape[0], prompt_ids.device)

    # Prefill: process the whole prompt in one forward pass.
    logits = model(prompt_ids, kv_cache=cache)
    next_id = sample_fn(logits[0, -1])
    new_tokens = [next_id]

    # Decode: one new token per step, fed back through the model.
    for _ in range(max_new_tokens - 1):
        if cache.length >= model.cfg.max_seq_len:
            break
        next_input = torch.tensor([[next_id]], device=prompt_ids.device, dtype=torch.long)
        logits = model(next_input, kv_cache=cache)
        next_id = sample_fn(logits[0, -1])
        new_tokens.append(next_id)

    return new_tokens
