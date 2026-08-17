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

The split geometry comes from the --data-mix config (same file
pretrain.py trains from), so the slice boundaries are byte-for-byte the
ones training used: pretraining loads train+val and never touches the
remainder — that remainder is the test slice this script reads. For
enwik8 with the canonical 90/5/5 split, the test slice is the final
5 MB, which no training or model-selection decision has ever seen.

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

from transformer import build_model
from transformer.data import load_data_mix
from transformer.eval import slice_nll
from transformer.tokenizer import load_tokenizer

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
        help="Checkpoints saved by any stage script. Default: every "
        "checkpoints/<stage>/*/best.pt",
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
        "--holdout",
        action="store_true",
        help="Treat --data as a fully-held-out file (excluded from the "
        "mix, never trained — e.g. emma.txt / great_expectations.txt): "
        "evaluate 100%% of its bytes as the test slice, skipping the "
        "splits lookup",
    )
    p.add_argument(
        "--no-bucket-push",
        action="store_true",
        help="Don't upload evals.jsonl to the run's bucket mirror after "
        "writing (push is on by default when GCS is configured — the "
        "training-stage sync never covers post-hoc evals)",
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

    Mirrors transformer.data.load_data's arithmetic exactly: train ends
    at int(n·train_frac), val at train_end + int(n·val_frac), and the
    test slice is everything after val — bytes pretraining never loads.

    --holdout instead evaluates the whole file: for corpora excluded
    from the mix entirely (the fully-held-out books), every byte is
    test.
    """
    if args.holdout:
        raw = args.data.read_bytes()
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        if args.max_bytes > 0:
            data = data[: args.max_bytes]
        return data, (f"{args.data.name}[0:{data.shape[0]:,}] "
                      f"(holdout — full file, {data.shape[0]:,} bytes)")
    paths, _mults, splits, _groups = load_data_mix(args.data_mix)
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


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    torch.set_float32_matmul_precision("high")

    ckpt_paths = args.checkpoints or sorted(
        list((PROJECT_ROOT / "checkpoints").glob("*/*/best.pt"))
        + list((PROJECT_ROOT / "checkpoints").glob("*/best.pt"))  # pre-stage layout
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

        # Evaluate in the model's own token space; bpb always normalizes
        # by BYTES of the slice, so numbers are comparable across
        # tokenizers. (Slice boundaries are byte-defined either way.)
        tok = load_tokenizer(getattr(cfg, "tokenizer", "bytes"))
        n_bytes = data.shape[0]
        if tok.name == "bytes":
            eval_data = data
        else:
            eval_data = torch.tensor(tok.encode(data.numpy().tobytes()),
                                     dtype=torch.long)

        t0 = time.perf_counter()
        nll_sum, n_pred = slice_nll(model, eval_data, seq_len, args.batch_size, device)
        elapsed = time.perf_counter() - t0
        denom = n_pred if tok.name == "bytes" else n_bytes
        bpb = nll_sum / denom / LN2
        print(f"    bpb={bpb:.4f}  loss={nll_sum / n_pred:.4f} nats/token  "
              f"({n_pred:,} tokens over {n_bytes:,} bytes in {elapsed:.0f}s)")
        results.append({
            "run": run,
            "checkpoint": str(path),
            "preset": ckpt.get("preset"),
            "tokenizer": tok.name,
            "n_params": model.num_parameters(),
            "step": ckpt.get("step"),
            "tokens_seen": ckpt.get("tokens_seen"),
            "split": "holdout" if args.holdout else args.split,
            "data": args.data.name,
            "seq_len": seq_len,
            "bytes": n_bytes,
            "tokens": n_pred,
            "loss_nats_per_token": nll_sum / n_pred,
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

    # Results live with the run: append to <run>/evals.jsonl — and push
    # it to the bucket mirror ourselves. BucketSync's last kick fires at
    # the final training save, BEFORE offline evals append, so without
    # this the file never syncs (gen-3 incident: retrieved manually
    # through an L4-stockout window).
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_utils import push_run_file
    for r in results:
        evals_path = Path(r["checkpoint"]).parent / "evals.jsonl"
        with open(evals_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if not args.no_bucket_push:
            if push_run_file(evals_path):
                print(f"  pushed {evals_path.parent.name}/evals.jsonl to bucket")

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
