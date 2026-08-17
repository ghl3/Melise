"""Model evaluation: exact bpb over byte slices, masked chat loss.

slice_nll() is the offline benchmark evaluator (scripts/eval_checkpoint.py):
a deterministic full pass over a contiguous byte slice in back-to-back
windows, every byte after the first predicted exactly once, summed —
not averaged over sampled batches — so results are exact and
reproducible.

pin_val_windows() + fixed_window_eval() are the gen-4 training-loop
evaluator: a PINNED set of windows per val slice, scored identically at
every eval, with accuracy/top-5/entropy alongside bpb. Replaces the
sampled eval whose mixture-draw noise (±0.04 bpb) let a lucky draw hold
best.pt for 53k steps in gen-3, and whose coverage varied 4×–0.07×
across domains.

masked_conversation_loss() is SFT validation: mean cross-entropy over
assistant-byte targets only, on a fixed prefix of the val conversations.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import conversation_batch

LN2 = math.log(2.0)

# Metric keys every fixed_window_eval domain entry carries.
EVAL_METRICS = ("loss", "bpb", "acc", "top5", "ent")


def pin_val_windows(val_datasets, val_paths, seq_len: int, n_windows: int,
                    seed: int = 0):
    """Deterministic window starts per val slice: every eval scores the
    SAME windows, so curves are comparable step-to-step (no draw noise)
    and every domain gets equal coverage regardless of slice size.
    Seeded by (seed, file stem) — stable across resumes and runs of the
    same recipe. Slices shorter than one window get no starts (and are
    skipped by fixed_window_eval)."""
    pinned = []
    for path, vd in zip(val_paths, val_datasets):
        hi = vd.shape[0] - seq_len - 1
        if hi <= 0:
            pinned.append([])
            continue
        rng = random.Random(f"fixedval:{seed}:{Path(path).stem}")
        pinned.append(sorted(rng.randrange(hi) for _ in range(n_windows)))
    return pinned


@torch.no_grad()
def fixed_window_eval(model, val_datasets, val_paths, pinned, seq_len: int,
                      batch_size: int, byte_lens=None, agg_weights=None):
    """Score the pinned windows. Returns (agg, domains):

    domains[stem] = {loss (nats/token), bpb (bits/RAW byte), acc (top-1),
    top5, ent (mean predictive entropy, bits)} — the triple that
    decomposes bpb moves in realtime: bpb↑ + acc flat + ent↓ is
    confidence misallocation (gen-3's war_and_peace mystery, diagnosable
    live); bpb↑ + acc↓ is rank loss. NB acc/top5/ent are per-TOKEN, not
    byte-true — within-run diagnostics, not cross-tokenizer comparable.

    agg is the val-slice-size-weighted aggregate over domains
    (weights = `agg_weights`, aligned to val_paths; uniform if omitted)
    — deterministic, so best.pt selection sees zero draw noise."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    try:
        domains: dict[str, dict[str, float]] = {}
        for path, vd, starts in zip(val_paths, val_datasets, pinned):
            if not starts:
                continue
            nll = n_tok = n_bytes = top1 = top5 = ent_bits = 0.0
            for lo in range(0, len(starts), batch_size):
                chunk = starts[lo:lo + batch_size]
                # Windows gather on the corpus's device (CPU for gen-4 —
                # see pretrain.py) and move to the model's.
                inputs = torch.stack(
                    [vd[s:s + seq_len] for s in chunk]).long().to(dev)
                targets = torch.stack(
                    [vd[s + 1:s + 1 + seq_len] for s in chunk]).long().to(dev)
                logits = model(inputs)
                lsm = F.log_softmax(logits.float(), dim=-1)
                nll -= float(lsm.gather(-1, targets.unsqueeze(-1)).sum())
                tk = logits.topk(5, dim=-1).indices
                eq = tk == targets.unsqueeze(-1)
                top1 += float(eq[..., 0].sum())
                top5 += float(eq.any(-1).sum())
                ent_bits += float((-lsm.exp() * lsm).sum(-1).sum()) / LN2
                n_tok += targets.numel()
                n_bytes += (float(byte_lens[targets].sum())
                            if byte_lens is not None else targets.numel())
            domains[Path(path).stem] = {
                "loss": nll / n_tok,
                "bpb": nll / LN2 / n_bytes,
                "acc": top1 / n_tok,
                "top5": top5 / n_tok,
                "ent": ent_bits / n_tok,
            }
        stems = list(domains)
        if agg_weights is not None:
            w = {Path(p).stem: float(x) for p, x in zip(val_paths, agg_weights)}
        else:
            w = {s: 1.0 for s in stems}
        total_w = sum(w[s] for s in stems) or 1.0
        agg = {k: sum(domains[s][k] * w[s] for s in stems) / total_w
               for k in EVAL_METRICS}
        return agg, domains
    finally:
        if was_training:
            model.train()


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
                     seq_len: int, n_batches: int,
                     byte_lens=None) -> tuple[float, float]:
    """Pretraining validation: (mean loss nats/token, bpb) over randomly
    sampled windows from the val slices (fast, approximate — the
    training-loop eval). bpb divides bits by the targets' RAW byte
    count via `byte_lens` (a per-token-id byte-length tensor), so the
    number is bits-per-byte under any tokenizer; without byte_lens it
    falls back to per-token bits (only honest for byte models). For
    exact numbers use slice_nll."""
    from .data import get_batch

    was_training = model.training
    model.eval()
    try:
        total, bits, n_bytes = 0.0, 0.0, 0.0
        for _ in range(n_batches):
            inputs, targets = get_batch(val_datasets, weights, batch_size, seq_len)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, model.cfg.vocab_size), targets.view(-1)
            )
            total += loss.item()
            bits += loss.item() * targets.numel() / LN2
            n_bytes += (float(byte_lens[targets].sum()) if byte_lens is not None
                        else targets.numel())
        return total / n_batches, bits / n_bytes
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def masked_conversation_loss(model, val_convs, batch_size: int, seq_len: int,
                             n_batches: int, device, tok=None,
                             byte_lens=None) -> tuple[float, float]:
    """Deterministic masked loss over a fixed prefix of the val set:
    (nats per assistant token, bits per assistant BYTE). The byte
    normalization uses `byte_lens` (per-token-id raw byte lengths) so
    the bpb number is tokenizer-independent; without it the second
    value is per-token bits (honest for byte models only)."""
    was_training = model.training
    model.eval()
    try:
        total, count, n_bytes = 0.0, 0, 0.0
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
            n_bytes += (float(byte_lens[targets[mask]].sum())
                        if byte_lens is not None else int(mask.sum().item()))
        return total / max(count, 1), total / LN2 / max(n_bytes, 1.0)
    finally:
        if was_training:
            model.train()
