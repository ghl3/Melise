"""Supervised fine-tuning on byte-level chat data (stage 2 of 3).

Examples:

    Fine-tune the Chinchilla small run on the chat mix at seq 1024:
        .venv/bin/python scripts/sft.py \\
            --init checkpoints/pretrain/kimi3-small-17M-scarlet-harbor-*/best.pt \\
            --steps 3000

    Resume an SFT run:
        .venv/bin/python scripts/sft.py --resume checkpoints/sft/<run>/latest.pt --steps 5000

Trains on conversations in the byte chat template (transformer.chat)
from data/chat_*.txt (build those with scripts/prep_chat_data.py and
scripts/gen_task_sft.py). Differences from pre-training (pretrain.py):

  - Examples are whole conversations (one per row, padded/truncated to
    --seq-len), not random windows of a byte stream.
  - The loss is masked to assistant spans: the model learns to produce
    responses and the end-of-turn byte, not to imitate user text.
  - The model is rebuilt at --seq-len (default 1024 > the 256 used in
    pre-training). Weights are sequence-length independent; kimi3 is
    NoPE/KDA so length extension is clean. RoPE models (base/vanilla)
    get freshly sized tables — expect some extrapolation cost.

Run dirs live at checkpoints/sft/sft-<name> and mirror to
gs://<bucket>/runs/sft/<name>; TensorBoard at --logdir checkpoints
groups runs by stage. Checkpoints use the same payload as pretrain.py,
so sample.py, eval_checkpoint.py, and grpo.py --init all load them
unchanged.

Reported bpb is bits per *assistant* byte (masked positions only), so it
is not comparable to pre-training bpb on enwik8.
"""

import argparse
import dataclasses
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from transformer import build_model, generate
from transformer.chat import completion_text, make_prompt
from transformer.data import conversation_batch, load_conversations
from transformer.eval import masked_conversation_loss

from run_utils import (
    BucketSync,
    MoEMonitor,
    Tee,
    derive_run_name,
    emit,
    fmt_eta,
    lr_at,
    open_metrics_log,
    prune_old_checkpoints,
    save_checkpoint,
    update_symlink,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

LN2 = math.log(2.0)

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about the sea.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supervised fine-tuning on byte-level chat data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model")
    g.add_argument("--init", type=Path, default=None,
                   help="Base checkpoint to fine-tune (a pretrain.py best.pt)")
    g.add_argument("--resume", type=Path, default=None,
                   help="SFT checkpoint to resume (continues its run dir)")
    g.add_argument("--seq-len", type=int, default=1024,
                   help="Max conversation length in bytes; model is rebuilt to this")

    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=3000)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-4,
                   help="Peak LR (lower than pre-training — we're fine-tuning)")
    g.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=("cosine", "constant"))
    g.add_argument("--warmup-frac", type=float, default=0.03)
    g.add_argument("--min-lr-frac", type=float, default=0.1)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0)

    g = p.add_argument_group("data")
    g.add_argument("--data", action="append", type=Path, default=None,
                   help="Chat corpus files. Default: data/chat_*.txt")
    g.add_argument("--val-frac", type=float, default=0.01,
                   help="Fraction of conversations held out for validation")

    g = p.add_argument_group("logging & checkpointing")
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--eval-batches", type=int, default=32)
    g.add_argument("--sample-every", type=int, default=200)
    g.add_argument("--no-sample", action="store_true")
    g.add_argument("--save-every", type=int, default=200)
    g.add_argument("--keep-last", type=int, default=3)
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--out", type=Path, default=None)
    g.add_argument("--no-tensorboard", action="store_true")
    g.add_argument("--no-bucket-sync", action="store_true")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="mps")
    return p.parse_args()


@torch.no_grad()
def sample_chat(model, device, user_text: str, max_new: int = 128) -> str:
    was_training = model.training
    model.eval()
    try:
        prompt = make_prompt(user_text)
        ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
        out = generate(model, ids, max_new_tokens=max_new)
        return completion_text(bytes(b & 0xFF for b in out))
    finally:
        if was_training:
            model.train()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    if (args.init is None) == (args.resume is None):
        raise SystemExit("exactly one of --init / --resume is required")

    src = args.resume or args.init
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    preset = ckpt.get("preset", "?")
    if cfg.max_seq_len != args.seq_len:
        cfg = dataclasses.replace(cfg, max_seq_len=args.seq_len)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    if args.out is not None:
        out_dir = args.out
    elif args.resume is not None:
        out_dir = args.resume.parent.resolve()
    else:
        name = args.run_name or derive_run_name(
            "sft", args.init, preset, model.num_parameters())
        out_dir = (PROJECT_ROOT / "checkpoints" / "sft" / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "train.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    metrics_f = open_metrics_log(out_dir / "metrics.jsonl")

    print(f"run dir: {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}")
    print(f"model:   {preset} ({type(model).__name__}), "
          f"{model.num_parameters():,} params, seq_len={cfg.max_seq_len}, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    start_step, tokens_seen = 0, 0
    best_val, best_step = float("inf"), None
    if args.resume is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        if ckpt.get("best_val") is not None:
            best_val, best_step = ckpt["best_val"], ckpt.get("best_step")
        print(f"resumed at step {start_step}")
    else:
        print(f"initialized from {args.init}")
    del ckpt

    if start_step >= args.steps:
        print(f"nothing to do — start_step ({start_step}) >= --steps ({args.steps}).")
        return

    data_paths = args.data or sorted((PROJECT_ROOT / "data").glob("chat_*.txt"))
    train_convs, val_convs = load_conversations(data_paths, args.seed, args.val_frac)
    n_bytes = sum(len(c) for c in train_convs)
    print(f"data: {len(train_convs):,} train / {len(val_convs):,} val conversations "
          f"({n_bytes / 1e6:.1f} MB train) from {[p.name for p in data_paths]}")
    truncated = sum(1 for c in train_convs if len(c) > args.seq_len + 1)
    print(f"      {truncated:,} conversations exceed seq_len and will be truncated "
          f"({truncated / len(train_convs):.1%})")

    manifest_path = out_dir / "run.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": out_dir.name,
            "kind": "sft",
            "preset": preset,
            "n_params": model.num_parameters(),
            "init": str(args.init),
            "args": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
            "datasets": [{"path": str(p), "bytes": p.stat().st_size}
                         for p in data_paths],
        }, indent=2, default=str))

    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    monitor = MoEMonitor(model)
    sync = BucketSync(out_dir, enabled=not args.no_bucket_sync, stage="sft")
    print(f"bucket sync: {sync.dest or 'off'}")
    print(f"training from step {start_step + 1} to {args.steps} "
          f"(batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr})\n")

    emit(metrics_f, event="start", step=start_step, kind="sft", preset=preset,
         n_params=model.num_parameters(), total_steps=args.steps,
         batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
         resumed=args.resume is not None)

    interrupted = {"flag": False}
    signal.signal(signal.SIGINT,
                  lambda *_: (interrupted.__setitem__("flag", True),
                              print("\n[SIGINT] will save and exit at next step boundary...")))

    def full_save(path, step):
        save_checkpoint(path, model, optimizer, step, cfg, preset=preset,
                        best_val=best_val, best_step=best_step,
                        tokens_seen=tokens_seen, device=device)

    model.train()
    t_start = time.perf_counter()
    n_done, last_step_done = 0, start_step

    try:
        for step in range(start_step + 1, args.steps + 1):
            if interrupted["flag"]:
                break
            lr = lr_at(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            is_log_step = step == start_step + 1 or step % args.log_every == 0
            monitor.enabled = is_log_step

            idx = torch.randint(0, len(train_convs), (args.batch_size,)).tolist()
            inputs, targets, mask = conversation_batch(
                train_convs, idx, args.seq_len, device)
            logits = model(inputs)
            loss = F.cross_entropy(logits[mask], targets[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            monitor.enabled = False
            n_done += 1
            last_step_done = step
            tokens_seen += args.batch_size * args.seq_len

            if is_log_step:
                elapsed = time.perf_counter() - t_start
                tps = n_done * args.batch_size * args.seq_len / elapsed
                eta = (elapsed / n_done) * (args.steps - step)
                loss_val = float(loss.item())
                print(f"step {step:>5}/{args.steps}  sft_loss={loss_val:.4f}  "
                      f"bpb={loss_val / LN2:.3f}  grad_norm={float(grad_norm.item()):.2f}  "
                      f"lr={lr:.2e}  ({tps:>6.0f} tok/s)  ETA {fmt_eta(eta)}")
                emit(metrics_f, event="step", step=step, train_loss=loss_val,
                     bpb=loss_val / LN2, lr=lr, grad_norm=float(grad_norm.item()),
                     tok_per_sec=tps, tokens_seen=tokens_seen,
                     moe_layers=monitor.detail(), **monitor.summary())
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/bpb", loss_val / LN2, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm.item()), step)

            if step % args.eval_every == 0:
                val = masked_conversation_loss(model, val_convs, args.batch_size,
                                               args.seq_len, args.eval_batches, device)
                is_best = val < best_val
                if is_best:
                    best_val, best_step = val, step
                    full_save(out_dir / "best.pt", step)
                print(f"        val_loss={val:.4f}  val_bpb={val / LN2:.3f}"
                      + ("  ← best" if is_best else ""))
                emit(metrics_f, event="eval", step=step, val_loss=val,
                     val_bpb=val / LN2, tokens_seen=tokens_seen, is_best=is_best)
                if writer is not None:
                    writer.add_scalar("val/loss", val, step)
                    writer.add_scalar("val/bpb", val / LN2, step)
                    writer.flush()
                sync.kick()

            if step % args.sample_every == 0 and not args.no_sample:
                for prompt in SAMPLE_PROMPTS:
                    reply = sample_chat(model, device, prompt)
                    print(f"--- {prompt!r}\n{reply}\n---")
                    emit(metrics_f, event="sample", step=step, prompt=prompt, text=reply)
                    if writer is not None:
                        writer.add_text("samples", f"**{prompt}**\n\n{reply}", step)

            if step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step}.pt"
                full_save(ckpt_path, step)
                update_symlink(out_dir, "latest.pt", ckpt_path)
                pruned = prune_old_checkpoints(out_dir, args.keep_last)
                print(f"        saved {ckpt_path.name}"
                      + (f"  pruned {len(pruned)}" if pruned else ""))
                emit(metrics_f, event="save", step=step, path=ckpt_path.name)
                sync.kick()
    finally:
        elapsed = time.perf_counter() - t_start
        if interrupted["flag"]:
            ckpt_path = out_dir / "interrupted.pt"
            full_save(ckpt_path, last_step_done)
            update_symlink(out_dir, "latest.pt", ckpt_path)
            print(f"\ninterrupted at step {last_step_done}; resume with --resume {ckpt_path}")
            emit(metrics_f, event="interrupted", step=last_step_done, elapsed_s=elapsed)
        elif n_done > 0:
            final = out_dir / f"step_{last_step_done}.pt"
            if not final.exists():
                full_save(final, last_step_done)
                update_symlink(out_dir, "latest.pt", final)
            print(f"\ndone — {n_done} steps in {elapsed:.1f}s; final: {final.name}")
            if best_step is not None:
                print(f"best val_loss={best_val:.4f} at step {best_step} (best.pt)")
            emit(metrics_f, event="end", step=last_step_done, elapsed_s=elapsed,
                 tokens_seen=tokens_seen, best_val=best_val, best_step=best_step)
        if writer is not None:
            writer.close()
        metrics_f.close()
        sync.finalize()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
