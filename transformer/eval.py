"""Model evaluation: exact bpb over byte slices, masked chat loss.

slice_nll() is the offline benchmark evaluator (scripts/eval_checkpoint.py):
a deterministic full pass over a contiguous byte slice in back-to-back
windows, every byte after the first predicted exactly once, summed —
not averaged over sampled batches — so results are exact and
reproducible.

masked_conversation_loss() is SFT validation: mean cross-entropy over
assistant-byte targets only, on a fixed prefix of the val conversations.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .data import conversation_batch

LN2 = math.log(2.0)


@torch.no_grad()
def slice_nll(model, data: torch.Tensor, seq_len: int, batch_size: int,
              device: torch.device, verbose: bool = True) -> tuple[float, int]:
    """Total cross-entropy (nats) over a contiguous byte slice.

    The slice is cut into back-to-back windows: window k holds inputs
    data[k·L : k·L+L] and targets shifted one byte right, so every byte
    except data[0] is predicted exactly once. Returns (nll_sum, n_pred);
    bpb = nll_sum / n_pred / ln 2.
    """
    n = data.shape[0]
    starts = list(range(0, n - 1, seq_len))
    nll_sum = 0.0
    n_pred = 0
    t0 = time.perf_counter()
    for i in range(0, len(starts), batch_size):
        batch = starts[i : i + batch_size]
        # The final window may be short; length-group so we can stack.
        lengths = [min(seq_len, n - 1 - s) for s in batch]
        full = [s for s, l in zip(batch, lengths) if l == seq_len]
        ragged = [(s, l) for s, l in zip(batch, lengths) if l < seq_len]
        chunks = []
        if full:
            chunks.append(
                (torch.stack([data[s : s + seq_len] for s in full]),
                 torch.stack([data[s + 1 : s + 1 + seq_len] for s in full]))
            )
        for s, l in ragged:
            chunks.append((data[s : s + l].unsqueeze(0),
                           data[s + 1 : s + 1 + l].unsqueeze(0)))
        for inputs, targets in chunks:
            inputs = inputs.long().to(device)
            targets = targets.long().to(device)
            logits = model(inputs)
            nll = F.cross_entropy(
                logits.view(-1, model.cfg.vocab_size),
                targets.view(-1),
                reduction="sum",
            )
            nll_sum += float(nll.item())
            n_pred += targets.numel()
        if verbose and (i // batch_size) % 50 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            tps = n_pred / elapsed
            eta = (n - 1 - n_pred) / tps
            print(
                f"    {n_pred:>10,}/{n - 1:,} bytes  "
                f"bpb so far {nll_sum / n_pred / LN2:.3f}  "
                f"({tps:,.0f} B/s, ETA {eta / 60:.0f}m)"
            )
    return nll_sum, n_pred


@torch.no_grad()
def sampled_val_loss(model, val_datasets, weights, batch_size: int,
                     seq_len: int, n_batches: int) -> float:
    """Pretraining validation: mean cross-entropy over randomly sampled
    windows from the val slices (fast, approximate — the training-loop
    eval). For exact numbers use slice_nll."""
    from .data import get_batch

    was_training = model.training
    model.eval()
    try:
        total = 0.0
        for _ in range(n_batches):
            inputs, targets = get_batch(val_datasets, weights, batch_size, seq_len)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, model.cfg.vocab_size), targets.view(-1)
            )
            total += loss.item()
        return total / n_batches
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def masked_conversation_loss(model, val_convs, batch_size: int, seq_len: int,
                             n_batches: int, device, tok=None) -> float:
    """Deterministic masked loss over a fixed prefix of the val set
    (nats per assistant token)."""
    was_training = model.training
    model.eval()
    try:
        total, count = 0.0, 0
        for b in range(n_batches):
            idx = list(range(b * batch_size, min((b + 1) * batch_size, len(val_convs))))
            if not idx:
                break
            inputs, targets, mask = conversation_batch(
                val_convs, idx, seq_len, device, tok=tok)
            logits = model(inputs)
            nll = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
            total += float(nll.item())
            count += int(mask.sum().item())
        return total / max(count, 1)
    finally:
        if was_training:
            model.train()
