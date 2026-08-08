"""Evaluate checkpoints on a reserved data split (bits per byte).

Examples:

    All best.pt checkpoints on the enwik8 test slice (the default):
        .venv/bin/python scripts/eval_checkpoint.py

    Specific checkpoints:
        .venv/bin/python scripts/eval_checkpoint.py \\
            checkpoints/kimi3-small-17M-crisp-harbor-*/best.pt

    Sanity-check against training numbers (val slice instead of test):
        .venv/bin/python scripts/eval_checkpoint.py --split val

    Quick smoke test on the first 200 kB of the slice:
        .venv/bin/python scripts/eval_checkpoint.py --max-bytes 200000

The split geometry comes from the --data-mix config (same file train.py
trains from), so the slice boundaries are byte-for-byte the ones training
used: train.py loads train+val and never touches the remainder — that
remainder is the test slice this script reads. For enwik8 with the
canonical 90/5/5 split, the test slice is the final 5 MB, which no
training or model-selection decision has ever seen.

Unlike training's eval (random windows from the val slice), this is a
deterministic full pass: the slice is cut into contiguous seq-len windows
and every byte after the first is predicted exactly once. bpb is the
exact total: sum of per-byte cross-entropy over the slice / (ln 2 × bytes
predicted). Each window predicts with only in-window context, so bytes
early in a window see short context — slightly pessimistic vs a
sliding-window eval, but identical methodology across checkpoints, which
is what matters for comparing runs. Results print as a table sorted by
bpb; --json appends machine-readable lines.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from transformer import build_model

# Sibling script import: scripts/ is sys.path[0] when running this file.
# load_data_mix is the single source of truth for split geometry.
from train import load_data_mix

LN2 = math.log(2.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate checkpoints on a reserved data split (bits per byte).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "checkpoints",
        type=Path,
        nargs="*",
        help="Checkpoints saved by scripts/train.py. Default: every "
        "checkpoints/*/best.pt",
    )
    p.add_argument(
        "--data-mix",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mix-downweight-wiki.json",
        help="Mixture config whose per-file splits define the slice boundaries",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "enwik8.txt",
        help="Corpus file to evaluate on (must have a splits entry in --data-mix)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("val", "test"),
        help="Which reserved slice to evaluate",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Window length. Default: the checkpoint config's max_seq_len",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Windows per batch")
    p.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="Evaluate only the first N bytes of the slice (0 = whole slice)",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Append one JSON line per checkpoint to this file",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto picks cuda > mps > cpu",
    )
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_slice(args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    """Return (slice bytes as a CPU uint8 tensor, description).

    Mirrors train.py's load_data arithmetic exactly: train ends at
    int(n·train_frac), val at train_end + int(n·val_frac), and the test
    slice is everything after val — the bytes train.py never loads.
    """
    paths, _mults, splits = load_data_mix(args.data_mix)
    path = args.data.resolve()
    if path not in [p.resolve() for p in paths]:
        raise SystemExit(f"{args.data} is not included by {args.data_mix}")
    by_resolved = {p.resolve(): s for p, s in splits.items()}
    if path not in by_resolved:
        raise SystemExit(
            f"{args.data} has no splits entry in {args.data_mix} — "
            "it trains on 100% of its bytes, so there is nothing held out"
        )
    train_frac, val_frac = by_resolved[path]

    raw = path.read_bytes()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    n = data.shape[0]
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    if args.split == "val":
        sl, lo, hi = data[train_end:val_end], train_end, val_end
    else:
        sl, lo, hi = data[val_end:], val_end, n
    if sl.shape[0] < 2:
        raise SystemExit(f"{args.split} slice of {path.name} is empty")
    if args.max_bytes > 0:
        sl = sl[: args.max_bytes]
        hi = lo + sl.shape[0]
    desc = f"{path.name}[{lo:,}:{hi:,}] ({args.split}, {sl.shape[0]:,} bytes)"
    return sl, desc


@torch.no_grad()
def eval_slice(model, data: torch.Tensor, seq_len: int, batch_size: int,
               device: torch.device) -> tuple[float, int]:
    """Total cross-entropy (nats) over a contiguous byte slice.

    The slice is cut into back-to-back windows: window k holds inputs
    data[k·L : k·L+L] and targets shifted one byte right, so every byte
    except data[0] is predicted exactly once. Returns (nll_sum, n_pred).
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
        if (i // batch_size) % 50 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            tps = n_pred / elapsed
            eta = (n - 1 - n_pred) / tps
            print(
                f"    {n_pred:>10,}/{n - 1:,} bytes  "
                f"bpb so far {nll_sum / n_pred / LN2:.3f}  "
                f"({tps:,.0f} B/s, ETA {eta / 60:.0f}m)"
            )
    return nll_sum, n_pred


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.set_float32_matmul_precision("high")

    ckpt_paths = args.checkpoints or sorted(
        (PROJECT_ROOT / "checkpoints").glob("*/best.pt")
    )
    if not ckpt_paths:
        raise SystemExit(
            "no checkpoints given and no checkpoints/*/best.pt found — "
            "pass paths explicitly or pull runs from the bucket"
        )

    data, desc = resolve_slice(args)
    print(f"slice:  {desc}")
    print(f"device: {device}")
    print()

    results = []
    for path in ckpt_paths:
        if not path.exists():
            print(f"skipping {path} (not found)")
            continue
        run = path.resolve().parent.name
        print(f"{run} ({path.name})")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt.pop("optimizer", None)  # free the largest piece before building
        cfg = ckpt["config"]
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        seq_len = args.seq_len or cfg.max_seq_len

        t0 = time.perf_counter()
        nll_sum, n_pred = eval_slice(model, data, seq_len, args.batch_size, device)
        elapsed = time.perf_counter() - t0
        bpb = nll_sum / n_pred / LN2
        print(f"    bpb={bpb:.4f}  loss={nll_sum / n_pred:.4f} nats  "
              f"({n_pred:,} bytes in {elapsed:.0f}s)")
        results.append({
            "run": run,
            "checkpoint": str(path),
            "preset": ckpt.get("preset"),
            "n_params": model.num_parameters(),
            "step": ckpt.get("step"),
            "tokens_seen": ckpt.get("tokens_seen"),
            "split": args.split,
            "data": args.data.name,
            "seq_len": seq_len,
            "bytes": n_pred,
            "loss_nats": nll_sum / n_pred,
            "bpb": bpb,
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        del model, ckpt
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit("no checkpoints evaluated")

    if args.json is not None:
        with open(args.json, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nappended {len(results)} result(s) to {args.json}")

    print(f"\n{args.split} results ({desc}):")
    header = f"{'run':<50} {'params':>7} {'step':>7} {'tokens':>7} {'bpb':>7}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["bpb"]):
        params = f"{r['n_params'] / 1e6:.0f}M"
        tokens = f"{r['tokens_seen'] / 1e6:.0f}M" if r["tokens_seen"] else "?"
        step = r["step"] if r["step"] is not None else "?"
        print(f"{r['run']:<50} {params:>7} {step:>7} {tokens:>7} {r['bpb']:>7.4f}")


if __name__ == "__main__":
    main()
