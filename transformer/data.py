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

import numpy as np
import torch

from .chat import (
    assemble,
    assistant_mask,
    assistant_mask_ids,
    encode_ids,
    parse_turns,
    split_conversations,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_bytes(tok) -> bool:
    return tok is None or tok.name == "bytes"


def _encode_slice_cached(path: Path, lo: int, hi: int, tok) -> np.ndarray:
    """Token ids (uint16) for bytes lo:hi of a corpus file, disk-cached
    per (file, byte-range, tokenizer). Byte ranges come from the split
    fractions, so slice boundaries are byte-identical across encodings —
    the enwik8 test slice stays the same 5 MB no matter the tokenizer."""
    cache = PROJECT_ROOT / "data" / ".tokcache"
    cache.mkdir(exist_ok=True)
    f = cache / f"{path.stem}.{lo}-{hi}.{tok.name}.npy"
    if f.exists():
        return np.load(f)
    data = path.read_bytes()[lo:hi]
    ids: list[int] = []
    start = 0
    while start < len(data):  # chunk at newlines to bound encoder memory
        end = min(start + (8 << 20), len(data))
        if end < len(data):
            nl = data.rfind(b"\n", start, end)
            if nl > start:
                end = nl + 1
        ids.extend(tok.encode(data[start:end]))
        start = end
    arr = np.asarray(ids, dtype=np.uint16)
    np.save(f, arr)
    print(f"  tokenized {path.name}[{lo}:{hi}]: {len(data) / max(len(arr), 1):.2f} "
          f"bytes/token ({len(arr):,} tokens, cached)")
    return arr


# ---------- Byte-stream corpora (pretraining) ----------


def load_data(paths, device, splits, tok=None):
    """Load each corpus as a token tensor: uint8 bytes for the byte
    tokenizer (int64 batches are cast on the fly in get_batch — storing
    corpora as int64 would be 8× the memory), int32 token ids for BPE
    (disk-cached; see _encode_slice_cached).

    `splits` maps path → (train_frac, val_frac) as BYTE fractions
    regardless of tokenizer — slicing happens before encoding. Files
    without an entry train on 100% of their bytes and contribute nothing
    to validation. Any remainder after train+val (e.g. enwik8's
    canonical last-5% test split) is NEVER loaded here — it stays
    untouched for offline evals. Returned byte counts are raw byte
    sizes, so mixture weighting is encoding-independent.

    val_list/val_bytes/val_paths are compact and parallel to each
    other, NOT to `paths` — only files with a val slice appear.
    """
    train_list, byte_counts = [], []
    val_list, val_bytes, val_paths = [], [], []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/download_data.py first."
            )
        n = path.stat().st_size
        train_frac, val_frac = splits.get(path, (1.0, 0.0))
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)
        if _is_bytes(tok):
            raw = path.read_bytes()
            data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
            train, val = data[:train_end], data[train_end:val_end]
        else:
            train = torch.from_numpy(
                _encode_slice_cached(path, 0, train_end, tok).astype(np.int32)
            ).to(device)
            val = torch.from_numpy(
                _encode_slice_cached(path, train_end, val_end, tok).astype(np.int32)
            ).to(device) if val_end > train_end else None
        train_list.append(train)
        if val_end > train_end:
            val_list.append(val)
            val_bytes.append(val_end - train_end)
            val_paths.append(path)
        byte_counts.append(n)
    return train_list, val_list, val_bytes, val_paths, byte_counts


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
         "groups": {                                  # optional, see below
            "fiction": {"include": "data/books/*.txt", "share": 0.10}},
         "splits": {                                  # default: 100% train
            "data/enwik8.txt": {"train": 0.90, "val": 0.05, "test": 0.05}}}

    Files without a "splits" entry train on all of their bytes. The test
    fraction is reserved — pretraining never loads it (offline evals do).
    "exclude" exists so stage-specific corpora sharing data/ (the SFT
    chat files) can't silently leak into pretraining.

    "groups" treats a set of files as one coherent corpus with a target
    share of the total sampling weight. Each group's files get a common
    multiplier m_g solved so that m_g·bytes_g / total_weight = share —
    so within a group, files are sampled by byte size (uniform effective
    epochs), and the group's gradient share is the single knob. Grouped
    files may not also appear in "multipliers" (ambiguous), nor in two
    groups. Ungrouped files keep their explicit/default multipliers;
    group shares are fractions of the SAME total, so with ungrouped
    files present, shares must sum to < 1 (the remainder is the
    ungrouped files' share). If every file is grouped, shares are the
    weights directly and should sum to 1.

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

    groups = mix.get("groups", {})
    if groups:
        path_idx = {p: i for i, p in enumerate(paths)}
        grouped: dict = {}  # path -> group name
        members: dict = {}  # group name -> [paths]
        for gname, spec in groups.items():
            gincludes = spec["include"]
            if isinstance(gincludes, str):
                gincludes = [gincludes]
            gpaths = sorted({p for g in gincludes for p in PROJECT_ROOT.glob(g)} & set(paths))
            if not gpaths:
                raise SystemExit(f"--data-mix: group {gname!r} matched no included file")
            for p in gpaths:
                if p in grouped:
                    raise SystemExit(
                        f"--data-mix: {p.name} is in groups {grouped[p]!r} and {gname!r}")
                if mult_of(p) is not None:
                    raise SystemExit(
                        f"--data-mix: {p.name} is in group {gname!r} AND has an "
                        "explicit multiplier — use one or the other")
                grouped[p] = gname
            members[gname] = gpaths
        shares = {g: float(spec["share"]) for g, spec in groups.items()}
        total_share = sum(shares.values())
        fixed_weight = sum(
            p.stat().st_size * mults[path_idx[p]] for p in paths if p not in grouped)
        if fixed_weight > 0:
            if total_share >= 1.0 - 1e-9:
                raise SystemExit(
                    f"--data-mix: group shares sum to {total_share:.3f} but ungrouped "
                    "files exist — shares must sum to < 1 to leave them room")
            total_weight = fixed_weight / (1.0 - total_share)
        else:
            total_weight = 1.0  # all files grouped: shares ARE the (relative) weights
        for gname, gpaths in members.items():
            group_bytes = sum(p.stat().st_size for p in gpaths)
            m_g = shares[gname] * total_weight / group_bytes
            for p in gpaths:
                mults[path_idx[p]] = m_g

    return paths, mults, splits


# ---------- Conversations (SFT) ----------


def split_conversation(conv: bytes, max_len: int, tok=None) -> list[bytes]:
    """Split one conversation at turn boundaries into chunks that fit
    max_len TOKENS under `tok` (bytes for the byte tokenizer).

    SFT truncates each example to seq_len+1 tokens, so without
    splitting, everything past the cap in a long conversation is
    silently wasted (~89% of the SmolTalk mix exceeds 512 bytes).
    Chunks keep whole turns; a chunk must contain an assistant turn to
    be a training example (user-only tails are dropped). A single turn
    longer than max_len stays alone and is truncated downstream."""
    if _is_bytes(tok):
        if len(conv) <= max_len:
            return [conv]
        cost = lambda content: len(content) + 2  # marker + END_TURN
    else:
        cost = lambda content: len(tok.encode(content)) + 2
    turns = parse_turns(conv)
    lens = [cost(content) for _, content in turns]
    if sum(lens) + 1 <= max_len:
        return [conv]
    chunks: list[bytes] = []
    cur: list[tuple[str, bytes]] = []
    cur_len = 1  # END_CONV

    def flush():
        if any(role == "assistant" for role, _ in cur):
            chunks.append(assemble(cur))

    for (role, content), turn_len in zip(turns, lens):
        if cur and cur_len + turn_len > max_len:
            flush()
            cur, cur_len = [], 1
        cur.append((role, content))
        cur_len += turn_len
    flush()
    return chunks or [conv]


def load_conversations(paths, seed: int, val_frac: float,
                       seq_len: int | None = None, tok=None):
    """Read chat corpora, split into conversations (further split at turn
    boundaries to fit seq_len tokens when given), and carve off a
    deterministic validation set. The shuffle uses its own RNG seeded by
    `seed`, so resumes see the identical train/val split."""
    convs = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path} not found — run scripts/prep_chat_data.py")
        cs = split_conversations(path.read_bytes())
        if seq_len is not None:
            cs = [chunk for c in cs for chunk in split_conversation(c, seq_len + 1, tok)]
        convs.extend(cs)
    random.Random(seed).shuffle(convs)
    n_val = max(int(len(convs) * val_frac), 64)
    return convs[n_val:], convs[:n_val]


def conversation_batch(convs, idx, seq_len: int, device, tok=None):
    """Tensorize conversations: inputs (B, L) int64, targets (B, L) int64,
    mask (B, L) bool marking positions whose *target* is an assistant
    token. Conversations are truncated to seq_len+1 tokens and padded
    with the pad id (0 in every tokenizer; pad positions are masked out,
    so pad is never a target)."""
    B, L = len(idx), seq_len
    mask = torch.zeros((B, L), dtype=torch.bool)
    if _is_bytes(tok):
        tokens = torch.zeros((B, L + 1), dtype=torch.uint8)
        for row, i in enumerate(idx):
            conv = convs[i][: L + 1]
            tokens[row, : len(conv)] = torch.frombuffer(
                bytearray(conv), dtype=torch.uint8)
            amask = assistant_mask(conv)
            for t in range(len(conv) - 1):
                mask[row, t] = amask[t + 1]
    else:
        tokens = torch.zeros((B, L + 1), dtype=torch.long)
        for row, i in enumerate(idx):
            ids = encode_ids(convs[i], tok)[: L + 1]
            tokens[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            amask = assistant_mask_ids(ids, tok)
            for t in range(len(ids) - 1):
                mask[row, t] = amask[t + 1]
    tokens = tokens.long().to(device)
    return tokens[:, :-1], tokens[:, 1:], mask.to(device)
