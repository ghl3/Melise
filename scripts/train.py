"""Train a TransformerLM on a byte-level text corpus.

Examples:

    First run (defaults):
        .venv/bin/python scripts/train.py --steps 500

    Longer run with bigger batch:
        .venv/bin/python scripts/train.py --steps 5000 --batch-size 32

    Resume training from a checkpoint:
        .venv/bin/python scripts/train.py --resume checkpoints/latest.pt --steps 10000

    Run training without sampling interruptions:
        .venv/bin/python scripts/train.py --steps 1000 --no-sample

The same `transformer.TransformerLM` is used for training and inference.
Training runs in fp32 for stability; the trained weights can be cast to
bf16 at inference time if desired.

Run from the project root.
"""

import argparse
import sys
import time
from pathlib import Path

# Make the `transformer` package importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from transformer import Config, TransformerLM, generate


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training schedule
    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=500,
                   help="Total training steps (target — counts steps from any resumed checkpoint)")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--seq-len", type=int, default=128, help="Sequence length per batch")
    g.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")

    # Logging / eval / sampling
    g = p.add_argument_group("logging & evaluation")
    g.add_argument("--log-every", type=int, default=25, help="Log train loss every N steps")
    g.add_argument("--eval-every", type=int, default=100, help="Run val eval every N steps")
    g.add_argument("--eval-batches", type=int, default=16,
                   help="Number of val batches averaged per eval")
    g.add_argument("--sample-every", type=int, default=100,
                   help="Generate a sample every N steps")
    g.add_argument("--sample-tokens", type=int, default=200,
                   help="Tokens to generate per sample")
    g.add_argument("--sample-prompt", type=str, default="ROMEO:\n")
    g.add_argument("--no-sample", action="store_true", help="Skip sampling during training")

    # Checkpointing
    g = p.add_argument_group("checkpointing")
    g.add_argument("--save-every", type=int, default=100,
                   help="Save a checkpoint every N steps")
    g.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints",
                   help="Checkpoint output directory")
    g.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint to resume from")
    g.add_argument("--keep-last", type=int, default=3,
                   help="Keep only the N most recent checkpoints (0 = keep all)")

    # Data
    g = p.add_argument_group("data")
    g.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "tinyshakespeare.txt")
    g.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of dataset held out for evaluation")

    # Misc
    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="mps")

    return p.parse_args()


# ---------- Data ----------

def load_data(path: Path, device: torch.device, val_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Read raw bytes; split into train and val (val = last `val_frac` of the file)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py first.")
    raw = path.read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long, device=device)
    n_val = max(int(len(data) * val_frac), 1)
    return data[:-n_val], data[-n_val:]


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crops with shifted targets."""
    starts = torch.randint(0, data.shape[0] - seq_len - 1, (batch_size,), device=data.device)
    inputs = torch.stack([data[s : s + seq_len] for s in starts])
    targets = torch.stack([data[s + 1 : s + 1 + seq_len] for s in starts])
    return inputs, targets


# ---------- Eval & sampling ----------

@torch.no_grad()
def eval_loss(model: TransformerLM, val_data: torch.Tensor, batch_size: int,
              seq_len: int, n_batches: int) -> float:
    """Average cross-entropy over `n_batches` random val crops."""
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        inputs, targets = get_batch(val_data, batch_size, seq_len)
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, model.cfg.vocab_size), targets.view(-1))
        total += loss.item()
    if was_training:
        model.train()
    return total / n_batches


@torch.no_grad()
def sample_text(model: TransformerLM, device: torch.device, prompt: bytes,
                n_tokens: int) -> str:
    was_training = model.training
    model.eval()
    ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
    out_ids = generate(model, ids, max_new_tokens=n_tokens)
    out_bytes = bytes(b if 0 <= b < 256 else 0x3f for b in out_ids)
    full = prompt + out_bytes
    if was_training:
        model.train()
    return full.decode("utf-8", errors="replace")


# ---------- Checkpointing ----------

def save_checkpoint(path: Path, model: TransformerLM, optimizer: torch.optim.Optimizer,
                    step: int, cfg: Config) -> None:
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


def load_checkpoint(path: Path, model: TransformerLM, optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def update_latest_symlink(out_dir: Path, target: Path) -> None:
    """Refresh `out_dir/latest.pt` to point at `target` (filename only — relative)."""
    latest = out_dir / "latest.pt"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(target.name)


def prune_old_checkpoints(out_dir: Path, keep_last: int) -> list[Path]:
    """Delete all but the most recent `keep_last` step_*.pt files.

    Returns the list of paths deleted. `keep_last <= 0` is a no-op.
    Doesn't touch `latest.pt` (which is a symlink, not a step_*.pt file).
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


# ---------- Main ----------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Build the model. Always train in fp32 for stability; cfg.dtype only
    # matters at inference time.
    cfg = Config(dtype=torch.float32, max_seq_len=args.seq_len)
    model = TransformerLM(cfg).to(device)
    print(f"model: {model.num_parameters():,} params, dtype={cfg.dtype}, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Resume?
    start_step = 0
    if args.resume is not None:
        print(f"resuming from {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed at step {start_step}")

    if start_step >= args.steps:
        print(f"nothing to do — start_step ({start_step}) >= --steps ({args.steps}). "
              f"Pass a higher --steps to continue training.")
        return

    train_data, val_data = load_data(args.data, device, args.val_frac)
    print(f"train: {train_data.numel():,} bytes  val: {val_data.numel():,} bytes")
    print(f"training from step {start_step + 1} to {args.steps} "
          f"(batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr})")
    print(f"output dir: {args.out}")
    print()

    args.out.mkdir(parents=True, exist_ok=True)

    model.train()
    t_start = time.perf_counter()
    n_done = 0

    for step in range(start_step + 1, args.steps + 1):
        inputs, targets = get_batch(train_data, args.batch_size, args.seq_len)

        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n_done += 1

        if step == start_step + 1 or step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            tps = n_done * args.batch_size * args.seq_len / elapsed
            print(f"step {step:>5}/{args.steps}  train_loss={loss.item():.4f}  ({tps:>6.0f} tok/s)")

        if step % args.eval_every == 0:
            val = eval_loss(model, val_data, args.batch_size, args.seq_len, args.eval_batches)
            print(f"        val_loss={val:.4f}")

        if step % args.sample_every == 0 and not args.no_sample:
            print("---")
            print(sample_text(model, device, args.sample_prompt.encode(),
                              args.sample_tokens).rstrip())
            print("---")

        if step % args.save_every == 0:
            ckpt = args.out / f"step_{step}.pt"
            save_checkpoint(ckpt, model, optimizer, step, cfg)
            update_latest_symlink(args.out, ckpt)
            pruned = prune_old_checkpoints(args.out, args.keep_last)
            msg = f"        saved {ckpt.name}  (latest.pt -> {ckpt.name})"
            if pruned:
                msg += f"  pruned {len(pruned)}"
            print(msg)

    elapsed = time.perf_counter() - t_start
    print(f"\ndone — {n_done} steps in {elapsed:.1f}s ({n_done / elapsed:.2f} steps/s)")

    # Always save a final checkpoint at the exact ending step.
    final = args.out / f"step_{args.steps}.pt"
    save_checkpoint(final, model, optimizer, args.steps, cfg)
    update_latest_symlink(args.out, final)
    prune_old_checkpoints(args.out, args.keep_last)
    print(f"final checkpoint: {final}")


if __name__ == "__main__":
    main()
