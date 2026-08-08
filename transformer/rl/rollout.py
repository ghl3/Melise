"""Policy rollouts and log-prob extraction for RL training.

rollout_group() decodes a whole GRPO group (n completions of one prompt)
as a single KV-cache batch. eval_rewards() is the deterministic
counterpart: greedy decoding over a fixed seeded task set, used for
best-checkpoint selection.
"""

from __future__ import annotations

import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from ..chat import END_TURN, make_prompt_ids
from ..tokenizer import ByteTokenizer
from .tasks import sample_tasks


@torch.no_grad()
def rollout_group(model, prompt, n: int, max_new: int, temperature: float,
                  device, greedy: bool = False, stop_id: int = END_TURN):
    """Decode `n` completions of one prompt (a token-id sequence) in a
    single batch.

    Returns (completions, old_lp, lengths):
      completions  list of n token-id lists, each cut at its stop_id
                   (inclusive) or max_new
      old_lp       (n, T) log-probs of the sampled tokens under the
                   policy that sampled them (raw logits, untempered)
      lengths      (n,) completion lengths

    stop_id defaults to the end-turn id, which is 3 under every
    tokenizer (byte template and BPE specials are pinned identically).
    """
    max_new = min(max_new, model.cfg.max_seq_len - len(prompt))
    ids = torch.tensor([list(prompt)] * n, device=device, dtype=torch.long)
    cache = model.new_cache(n, device)
    logits = model(ids, kv_cache=cache)[:, -1]  # (n, V)

    toks, lps = [], []
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new):
        lp_all = F.log_softmax(logits, dim=-1)
        if greedy:
            tok = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)
        toks.append(tok.squeeze(1))
        lps.append(lp_all.gather(1, tok).squeeze(1))
        done |= tok.squeeze(1) == stop_id
        if bool(done.all()):
            break
        logits = model(tok, kv_cache=cache)[:, -1]

    toks = torch.stack(toks, dim=1)  # (n, T)
    lps = torch.stack(lps, dim=1)
    completions, lengths = [], []
    for row in toks.tolist():
        cut = row.index(stop_id) + 1 if stop_id in row else len(row)
        completions.append(row[:cut])
        lengths.append(cut)
    return completions, lps, lengths


def gather_completion_logprobs(model, ids, pos, tok):
    """Log-probs the model assigns to completion tokens.

    ids (N, L) padded full sequences; pos (N, T) index of the logit that
    predicts each completion token; tok (N, T) the completion tokens.
    Differentiable — run under no_grad for reference models.
    """
    logits = model(ids)  # (N, L, V)
    at = logits.gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
    return F.log_softmax(at, dim=-1).gather(2, tok.unsqueeze(-1)).squeeze(-1)


def eval_rewards(model, task_names, n_prompts: int, seed, max_new: int, device,
                 tok=None):
    """Greedy decode a fixed task set; mean reward overall and per kind."""
    tok = tok or ByteTokenizer()
    was_training = model.training
    model.eval()
    try:
        tasks = sample_tasks(task_names, n_prompts, random.Random(f"{seed}-eval"))
        by_kind = defaultdict(list)
        for task in tasks:
            comps, _, _ = rollout_group(
                model, make_prompt_ids(task.prompt, tok), 1, max_new, 1.0,
                device, greedy=True, stop_id=tok.end_turn_id,
            )
            text = tok.decode(comps[0]).strip()
            by_kind[task.kind].append(task.score(text))
        all_scores = [s for v in by_kind.values() for s in v]
        return (sum(all_scores) / len(all_scores),
                {k: sum(v) / len(v) for k, v in sorted(by_kind.items())})
    finally:
        if was_training:
            model.train()
