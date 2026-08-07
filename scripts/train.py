"""Train a TransformerLM on a byte-level text corpus.

Examples:

    First run (auto-generated run name in checkpoints/):
        .venv/bin/python scripts/train.py --steps 500

    Train the Kimi K3 miniature (KDA runs a sequential scan — prefer a
    modest --seq-len):
        .venv/bin/python scripts/train.py --preset kimi3 --seq-len 256 --steps 2000

    Custom run name:
        .venv/bin/python scripts/train.py --run-name my-experiment --steps 5000

    Resume training (uses the same directory):
        .venv/bin/python scripts/train.py \\
            --resume checkpoints/calm-river-20260503-141522/latest.pt --steps 10000

    Skip in-training samples for cleaner logs:
        .venv/bin/python scripts/train.py --steps 1000 --no-sample

Each run writes to its own subdirectory under checkpoints/ containing:

    step_NNNN.pt    rolling checkpoints (oldest pruned to --keep-last)
    latest.pt       symlink to most recent
    best.pt         full checkpoint, replaced whenever val_loss hits a new low
    interrupted.pt  saved on Ctrl-C if training is interrupted
    run.json        manifest: args, config, datasets, start time
    metrics.jsonl   append-only event log (steps / evals / saves / samples)
    train.log       full stdout, mirrored from console

The same `transformer.TransformerLM` is used for training and inference.
Training runs in fp32 for stability; trained weights can be cast to bf16
at inference time. Run from the project root.
"""

import argparse
import json
import random
import signal
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

# Make the `transformer` package importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from transformer import MODELS, generate

# ---------- Auto-naming ----------

_ADJECTIVES = [
    "calm",
    "wild",
    "swift",
    "bright",
    "silver",
    "golden",
    "stormy",
    "gentle",
    "crisp",
    "frozen",
    "scarlet",
    "dusky",
    "frosty",
    "sunny",
    "quiet",
    "fierce",
    "eager",
    "jolly",
    "hidden",
    "crystal",
    "crimson",
    "ember",
    "azure",
    "twilight",
    "rugged",
]
_NOUNS = [
    "river",
    "mountain",
    "valley",
    "harbor",
    "glacier",
    "cypress",
    "sparrow",
    "otter",
    "wolf",
    "falcon",
    "brook",
    "meadow",
    "canyon",
    "lighthouse",
    "garden",
    "forest",
    "owl",
    "comet",
    "nova",
    "dawn",
    "willow",
    "raven",
    "tide",
    "summit",
    "dell",
]


def generate_run_name() -> str:
    """Return e.g. 'calm-river-20260503-141522'.

    Uses a separate time-seeded RNG so the chosen name is independent of
    --seed (otherwise two default-seed runs started in the same second
    would collide).
    """
    rng = random.Random()  # seeded from os.urandom
    adj = rng.choice(_ADJECTIVES)
    noun = rng.choice(_NOUNS)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{adj}-{noun}-{ts}"


# ---------- Tee writer (stdout -> console + file) ----------


class Tee:
    """File-like object that writes to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------- ETA formatting ----------


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) / 60)
    return f"{h}h{m:02d}m"


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("model")
    g.add_argument(
        "--preset",
        type=str,
        default="base",
        choices=sorted(MODELS),
        help="Architecture to train: base (GQA + MoE), vanilla (MHA + dense), "
        "deepseek (MLA + DeepSeekMoE), kimi3 (KDA/MLA hybrid + AttnRes + LatentMoE)",
    )

    g = p.add_argument_group("training")
    g.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Total training steps (target — counts steps from any resumed checkpoint)",
    )
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--seq-len", type=int, default=512, help="Sequence length per batch")
    g.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")

    g = p.add_argument_group("logging & evaluation")
    g.add_argument(
        "--log-every", type=int, default=25, help="Log train loss every N steps"
    )
    g.add_argument(
        "--eval-every", type=int, default=100, help="Run val eval every N steps"
    )
    g.add_argument(
        "--eval-batches",
        type=int,
        default=16,
        help="Number of val batches averaged per eval",
    )
    g.add_argument(
        "--sample-every", type=int, default=100, help="Generate a sample every N steps"
    )
    g.add_argument("--sample-tokens", type=int, default=200)
    g.add_argument("--sample-prompt", type=str, default="ROMEO:\n")
    g.add_argument("--no-sample", action="store_true")

    g = p.add_argument_group("checkpointing")
    g.add_argument("--save-every", type=int, default=100)
    g.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run output directory. Default: checkpoints/{run-name}/",
    )
    g.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Auto-generated if omitted (e.g. 'calm-river-20260503-141522')",
    )
    g.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from (uses its parent dir as --out). "
        "Must have been trained with the same --preset",
    )
    g.add_argument(
        "--keep-last",
        type=int,
        default=5,
        help="Keep only N most recent checkpoints (best.pt's target is protected)",
    )

    g = p.add_argument_group("data")
    g.add_argument(
        "--data",
        action="append",
        type=Path,
        default=None,
        help="Path to a training corpus. Repeat for mixture training.",
    )
    g.add_argument(
        "--data-weights",
        type=str,
        default=None,
        help="Comma-separated sampling weights. Default: weight by byte size.",
    )
    g.add_argument("--val-frac", type=float, default=0.05)

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="mps")

    return p.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Pick the output directory based on --out / --resume / --run-name."""
    if args.out is not None:
        return args.out
    if args.resume is not None:
        return args.resume.parent.resolve()
    name = args.run_name or generate_run_name()
    return (PROJECT_ROOT / "checkpoints" / name).resolve()


# ---------- Data ----------


def load_data(paths, device, val_frac):
    train_list, val_list, byte_counts = [], [], []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/download_data.py first."
            )
        raw = path.read_bytes()
        data = torch.tensor(list(raw), dtype=torch.long, device=device)
        n_val = max(int(len(data) * val_frac), 1)
        train_list.append(data[:-n_val])
        val_list.append(data[-n_val:])
        byte_counts.append(len(raw))
    return train_list, val_list, byte_counts


def get_batch(datasets, weights, batch_size, seq_len):
    if len(datasets) == 1:
        d = datasets[0]
        starts = torch.randint(
            0, d.shape[0] - seq_len - 1, (batch_size,), device=d.device
        )
        inputs = torch.stack([d[s : s + seq_len] for s in starts])
        targets = torch.stack([d[s + 1 : s + 1 + seq_len] for s in starts])
        return inputs, targets

    chosen = torch.multinomial(weights, batch_size, replacement=True).tolist()
    inputs, targets = [], []
    for d_idx in chosen:
        d = datasets[d_idx]
        s = int(torch.randint(0, d.shape[0] - seq_len - 1, (1,)).item())
        inputs.append(d[s : s + seq_len])
        targets.append(d[s + 1 : s + 1 + seq_len])
    return torch.stack(inputs), torch.stack(targets)


def parse_weights(weights_str, byte_counts):
    if weights_str is None:
        return torch.tensor([float(b) for b in byte_counts])
    weights = [float(w) for w in weights_str.split(",")]
    if len(weights) != len(byte_counts):
        raise SystemExit(
            f"--data-weights has {len(weights)} entries; --data has {len(byte_counts)}."
        )
    return torch.tensor(weights)


# ---------- Eval & sampling ----------


@torch.no_grad()
def eval_loss(model, val_datasets, weights, batch_size, seq_len, n_batches):
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
def sample_text(model, device, prompt, n_tokens):
    was_training = model.training
    model.eval()
    try:
        ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
        out_ids = generate(model, ids, max_new_tokens=n_tokens)
        out_bytes = bytes(b if 0 <= b < 256 else 0x3F for b in out_ids)
        full = prompt + out_bytes
        return full.decode("utf-8", errors="replace")
    finally:
        if was_training:
            model.train()


# ---------- Checkpointing ----------


def save_checkpoint(path, model, optimizer, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def update_symlink(out_dir, name, target):
    """(re)point `out_dir/name` at `target` (relative filename)."""
    link = out_dir / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.name)


def prune_old_checkpoints(out_dir, keep_last):
    """Delete all but the most recent `keep_last` step_*.pt files.

    `best.pt` is now a regular file (not a symlink to a step_*.pt), so
    pruning step files doesn't risk deleting it.
    """
    if keep_last <= 0:
        return []
    ckpts = sorted(
        out_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if len(ckpts) <= keep_last:
        return []
    to_delete = ckpts[:-keep_last]
    for p in to_delete:
        p.unlink()
    return to_delete


def recover_best_from_metrics(metrics_path):
    """Scan an existing metrics.jsonl for the best val seen so far.

    Returns (best_val, best_step) or (inf, None) if no eval events found.
    Used on resume to continue tracking best from where the previous run
    left off, instead of resetting to inf.
    """
    best_val = float("inf")
    best_step = None
    if not metrics_path.exists():
        return best_val, best_step
    with open(metrics_path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "eval":
                v = ev.get("val_loss")
                if v is not None and v < best_val:
                    best_val = v
                    best_step = ev.get("step")
    return best_val, best_step


# ---------- Metrics + run manifest ----------


def write_run_manifest(path, args, cfg, byte_counts, weights):
    """One-shot snapshot of the run setup. Written at start; never updated."""
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": path.parent.name,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "config": asdict(cfg) if is_dataclass(cfg) else {},
        "config_dtype": str(cfg.dtype),
        "datasets": [
            {"path": str(p), "bytes": int(b), "weight": float(w)}
            for p, b, w in zip(args.data, byte_counts, weights.tolist())
        ],
    }
    # cfg.dtype is a torch.dtype; not JSON-serializable. Strip from inner dict.
    manifest["config"].pop("dtype", None)
    path.write_text(json.dumps(manifest, indent=2, default=str))


def open_metrics_log(path):
    """Open the metrics JSONL file in append mode, line-buffered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", buffering=1)


def emit(metrics_f, **fields):
    """Append one JSONL event. Always includes a timestamp."""
    fields.setdefault("time", datetime.now().isoformat(timespec="seconds"))
    metrics_f.write(json.dumps(fields) + "\n")


# ---------- Main ----------


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out_dir = resolve_run_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out = out_dir

    # Mirror stdout to a per-run train.log. Closed in the finally block.
    log_file = open(out_dir / "train.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = open_metrics_log(metrics_path)

    # Build the model from its preset. Always train in fp32 for stability.
    config_cls, model_cls = MODELS[args.preset]
    cfg = config_cls(dtype=torch.float32, max_seq_len=args.seq_len)
    model = model_cls(cfg).to(device)
    print(
        f"run dir: {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}"
    )
    print(
        f"model:   {args.preset} ({model_cls.__name__}), {model.num_parameters():,} params, "
        f"dtype={cfg.dtype}, device={device}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_step = 0
    if args.resume is not None:
        print(f"resuming from {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed at step {start_step}")

    if start_step >= args.steps:
        print(f"nothing to do — start_step ({start_step}) >= --steps ({args.steps}).")
        return

    if not args.data:
        args.data = [PROJECT_ROOT / "data" / "tinyshakespeare.txt"]

    train_data, val_data, byte_counts = load_data(args.data, device, args.val_frac)
    weights = parse_weights(args.data_weights, byte_counts)
    norm_weights = weights / weights.sum()

    print(f"datasets ({len(args.data)}):")
    for path, n_bytes, w in zip(args.data, byte_counts, norm_weights.tolist()):
        rel = (
            path.relative_to(PROJECT_ROOT)
            if path.is_absolute() and path.is_relative_to(PROJECT_ROOT)
            else path
        )
        print(f"  {str(rel):<42}  {n_bytes:>10,} bytes  ({w * 100:>5.1f}% sampling)")

    # Run manifest (only on first start; preserves original on resume).
    manifest_path = out_dir / "run.json"
    if not manifest_path.exists():
        write_run_manifest(manifest_path, args, cfg, byte_counts, norm_weights)

    print(
        f"training from step {start_step + 1} to {args.steps} "
        f"(batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr})"
    )
    print()

    emit(
        metrics_f,
        event="start",
        step=start_step,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        resumed=args.resume is not None,
    )

    # Set up SIGINT (Ctrl-C) handler. We don't exit immediately — we set a
    # flag and let the next iteration of the loop save cleanly and then exit.
    interrupted = {"flag": False}

    def sigint_handler(_signum, _frame):
        interrupted["flag"] = True
        print("\n[SIGINT] will save and exit at next step boundary...")

    signal.signal(signal.SIGINT, sigint_handler)

    # On resume, recover best_val/best_step from the existing metrics log so
    # we keep tracking the historical best instead of starting from inf.
    if args.resume is not None:
        best_val, best_step = recover_best_from_metrics(metrics_path)
        if best_step is not None:
            print(f"recovered best so far: val_loss={best_val:.4f} at step {best_step}")
    else:
        best_val = float("inf")
        best_step = None

    model.train()
    t_start = time.perf_counter()
    n_done = 0
    last_step_done = start_step

    try:
        for step in range(start_step + 1, args.steps + 1):
            if interrupted["flag"]:
                break

            inputs, targets = get_batch(
                train_data, weights, args.batch_size, args.seq_len
            )
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            n_done += 1
            last_step_done = step

            if step == start_step + 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - t_start
                tps = n_done * args.batch_size * args.seq_len / elapsed
                steps_left = args.steps - step
                eta = (elapsed / n_done) * steps_left
                print(
                    f"step {step:>5}/{args.steps}  train_loss={loss.item():.4f}  "
                    f"({tps:>6.0f} tok/s)  ETA {fmt_eta(eta)}"
                )
                emit(
                    metrics_f,
                    event="step",
                    step=step,
                    train_loss=float(loss.item()),
                    tok_per_sec=tps,
                    eta_s=eta,
                )

            if step % args.eval_every == 0:
                val = eval_loss(
                    model,
                    val_data,
                    weights,
                    args.batch_size,
                    args.seq_len,
                    args.eval_batches,
                )
                is_best = val < best_val
                if is_best:
                    best_val = val
                    best_step = step
                    # Save best.pt as a regular file (not a symlink) immediately
                    # at the eval that produced the new best. This avoids the
                    # eval-vs-save alignment bug.
                    save_checkpoint(out_dir / "best.pt", model, optimizer, step, cfg)
                print(f"        val_loss={val:.4f}" + ("  ← best" if is_best else ""))
                emit(metrics_f, event="eval", step=step, val_loss=val, is_best=is_best)

            if step % args.sample_every == 0 and not args.no_sample:
                text = sample_text(
                    model, device, args.sample_prompt.encode(), args.sample_tokens
                ).rstrip()
                print("---")
                print(text)
                print("---")
                emit(
                    metrics_f,
                    event="sample",
                    step=step,
                    prompt=args.sample_prompt,
                    text=text,
                )

            if step % args.save_every == 0:
                ckpt = out_dir / f"step_{step}.pt"
                save_checkpoint(ckpt, model, optimizer, step, cfg)
                update_symlink(out_dir, "latest.pt", ckpt)
                pruned = prune_old_checkpoints(out_dir, args.keep_last)
                msg = f"        saved {ckpt.name}  (latest -> {ckpt.name})"
                if pruned:
                    msg += f"  pruned {len(pruned)}"
                print(msg)
                emit(
                    metrics_f,
                    event="save",
                    step=step,
                    path=ckpt.name,
                    pruned=len(pruned),
                )
    finally:
        # Always: save an interrupted/final checkpoint so we never lose work.
        elapsed = time.perf_counter() - t_start
        if interrupted["flag"]:
            ckpt = out_dir / "interrupted.pt"
            save_checkpoint(ckpt, model, optimizer, last_step_done, cfg)
            update_symlink(out_dir, "latest.pt", ckpt)
            print(f"\ninterrupted at step {last_step_done} after {elapsed:.1f}s")
            print(f"  saved {ckpt.name}; resume with --resume {ckpt}")
            emit(metrics_f, event="interrupted", step=last_step_done, elapsed_s=elapsed)
        elif n_done > 0:
            print(
                f"\ndone — {n_done} steps in {elapsed:.1f}s "
                f"({n_done / elapsed:.2f} steps/s)"
            )
            # Avoid a duplicate save when the loop's final iteration already
            # saved this step (i.e. last_step_done % save_every == 0).
            final = out_dir / f"step_{last_step_done}.pt"
            if not final.exists():
                save_checkpoint(final, model, optimizer, last_step_done, cfg)
                update_symlink(out_dir, "latest.pt", final)
                prune_old_checkpoints(out_dir, args.keep_last)
            print(f"final checkpoint: {final.name}")
            if best_step is not None:
                print(f"best val_loss={best_val:.4f} at step {best_step} (best.pt)")
            emit(
                metrics_f,
                event="end",
                step=last_step_done,
                elapsed_s=elapsed,
                best_val=best_val if best_step else None,
                best_step=best_step,
            )
        metrics_f.close()
        # Restore stdout/stderr and close the log file.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
