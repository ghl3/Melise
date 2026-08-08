"""Corpus and conversation loading/batching for all training stages.

Two data models live here:

  Byte-stream corpora (pretraining): each file is one long uint8 tensor;
  batches are random fixed-length windows. load_data_mix() reads the
  mixture config (which files, sampling multipliers, train/val/test
  splits); load_data() materializes the train/val slices — the test
  remainder is NEVER loaded here, it stays virgin for offline evals
  (transformer.eval.slice_nll via scripts/eval_checkpoint.py).

  Conversations (SFT): each example is one whole conversation in the
  byte chat template (transformer.chat); batches are padded rows with a
  per-position mask selecting assistant-byte targets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .chat import assistant_mask, split_conversations

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- Byte-stream corpora (pretraining) ----------


def load_data(paths, device, splits):
    """Load each corpus as a uint8 tensor (1 byte per token). Batches are
    cast to int64 on the fly in get_batch — storing the corpora themselves
    as int64 would be 8× the memory (WikiText-103 alone would be ~4.3 GB).

    `splits` maps path → (train_frac, val_frac). Files without an entry
    train on 100% of their bytes and contribute nothing to validation.
    Any remainder after train+val (e.g. enwik8's canonical last-5% test
    split) is NEVER loaded here — it stays untouched for offline evals.
    """
    train_list, byte_counts = [], []
    val_list, val_bytes = [], []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/download_data.py first."
            )
        raw = path.read_bytes()
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
        train_frac, val_frac = splits.get(path, (1.0, 0.0))
        n = data.shape[0]
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        train_list.append(data[:train_end])
        if val_end > train_end:
            val_list.append(data[train_end:val_end])
            val_bytes.append(val_end - train_end)
        byte_counts.append(len(raw))
    return train_list, val_list, val_bytes, byte_counts


def get_batch(datasets, weights, batch_size, seq_len):
    """Random windows from a weighted mixture of byte streams. RNG use is
    deliberate and stable (device RNG for single-corpus starts, CPU RNG
    for mixture choice) — exact-resume depends on it."""
    if len(datasets) == 1:
        d = datasets[0]
        starts = torch.randint(
            0, d.shape[0] - seq_len - 1, (batch_size,), device=d.device
        )
        inputs = torch.stack([d[s : s + seq_len] for s in starts])
        targets = torch.stack([d[s + 1 : s + 1 + seq_len] for s in starts])
        return inputs.long(), targets.long()

    chosen = torch.multinomial(weights, batch_size, replacement=True).tolist()
    inputs, targets = [], []
    for d_idx in chosen:
        d = datasets[d_idx]
        s = int(torch.randint(0, d.shape[0] - seq_len - 1, (1,)).item())
        inputs.append(d[s : s + seq_len])
        targets.append(d[s + 1 : s + 1 + seq_len])
    return torch.stack(inputs).long(), torch.stack(targets).long()


def parse_weights(weights_str, byte_counts):
    if weights_str is None:
        return torch.tensor([float(b) for b in byte_counts])
    weights = [float(w) for w in weights_str.split(",")]
    if len(weights) != len(byte_counts):
        raise SystemExit(
            f"--data-weights has {len(weights)} entries; --data has {len(byte_counts)}."
        )
    return torch.tensor(weights)


def load_data_mix(mix_path):
    """Read a mixture config: which files to train on, per-file
    sampling-weight multipliers, and per-file train/val/test splits.

        {"include": "data/*.txt",                     # glob or list of globs
         "exclude": "data/chat_*.txt",                # glob(s) removed after include
         "multipliers": {"data/enwik8.txt": 0.1},     # default 1.0
         "splits": {                                  # default: 100% train
            "data/enwik8.txt": {"train": 0.90, "val": 0.05, "test": 0.05}}}

    Files without a "splits" entry train on all of their bytes. The test
    fraction is reserved — pretraining never loads it (offline evals do).
    "exclude" exists so stage-specific corpora sharing data/ (the SFT
    chat files) can't silently leak into pretraining.

    Returns (paths, multipliers, splits) with multipliers aligned to
    paths and splits keyed by path. Keys may be project-relative paths or
    bare filenames.
    """
    mix = json.loads(mix_path.read_text())
    includes = mix.get("include", "data/*.txt")
    if isinstance(includes, str):
        includes = [includes]
    excludes = mix.get("exclude", [])
    if isinstance(excludes, str):
        excludes = [excludes]
    excluded = {p for g in excludes for p in PROJECT_ROOT.glob(g)}
    paths = sorted({p for g in includes for p in PROJECT_ROOT.glob(g)} - excluded)
    if not paths:
        raise SystemExit(f"--data-mix: include patterns {includes} matched no files")

    def match(table, matched):
        """Look up a per-file entry by relative path or filename."""
        def lookup(p):
            rel = str(p.relative_to(PROJECT_ROOT))
            for key in (rel, p.name):
                if key in table:
                    matched.add(key)
                    return table[key]
            return None
        return lookup

    raw_mults = dict(mix.get("multipliers", {}))
    raw_splits = dict(mix.get("splits", {}))
    matched = set()
    mult_of = match(raw_mults, matched)
    split_of = match(raw_splits, matched)

    mults, splits = [], {}
    for p in paths:
        m = mult_of(p)
        mults.append(1.0 if m is None else float(m))
        s = split_of(p)
        if s is not None:
            train, val = float(s.get("train", 1.0)), float(s.get("val", 0.0))
            test = float(s.get("test", 0.0))
            if train + val + test > 1.0 + 1e-9:
                raise SystemExit(f"--data-mix: splits for {p.name} sum to > 1")
            splits[p] = (train, val)

    unmatched = (set(raw_mults) | set(raw_splits)) - matched
    if unmatched:
        raise SystemExit(
            f"--data-mix: keys match no included file: {sorted(unmatched)}"
        )
    return paths, mults, splits


# ---------- Conversations (SFT) ----------


def load_conversations(paths, seed: int, val_frac: float):
    """Read chat corpora, split into conversations, and carve off a
    deterministic validation set. The shuffle uses its own RNG seeded by
    `seed`, so resumes see the identical train/val split."""
    convs = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path} not found — run scripts/prep_chat_data.py")
        convs.extend(split_conversations(path.read_bytes()))
    random.Random(seed).shuffle(convs)
    n_val = max(int(len(convs) * val_frac), 64)
    return convs[n_val:], convs[:n_val]


def conversation_batch(convs, idx, seq_len: int, device):
    """Tensorize conversations: inputs (B, L) int64, targets (B, L) int64,
    mask (B, L) bool marking positions whose *target* is an assistant byte.
    Conversations are truncated to seq_len+1 bytes and padded with 0x00
    (pad positions are masked out, so the pad byte is never a target)."""
    B, L = len(idx), seq_len
    tokens = torch.zeros((B, L + 1), dtype=torch.uint8)
    mask = torch.zeros((B, L), dtype=torch.bool)
    for row, i in enumerate(idx):
        conv = convs[i][: L + 1]
        tokens[row, : len(conv)] = torch.frombuffer(bytearray(conv), dtype=torch.uint8)
        amask = assistant_mask(conv)
        for t in range(len(conv) - 1):
            mask[row, t] = amask[t + 1]
    tokens = tokens.long().to(device)
    return tokens[:, :-1], tokens[:, 1:], mask.to(device)
