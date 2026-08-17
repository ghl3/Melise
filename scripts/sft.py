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
from transformer.chat import DEFAULT_PREAMBLE, make_prompt_ids, split_conversations
from transformer.data import (conversation_batch, get_batch,
                              load_conversations, load_data, load_data_mix)
from transformer.eval import masked_conversation_loss
from transformer.tokenizer import load_tokenizer

from run_utils import (
    BucketSync,
    MoEMonitor,
    Tee,
    checkpoint_identity,
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
    g.add_argument("--seq-len", type=int, default=2048,
                   help="Max conversation length in bytes; model is rebuilt to this "
                   "(clean for NoPE models regardless of pretrain seq-len)")

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
    g.add_argument("--replay-frac", type=float, default=0.0,
                   help="Fraction of steps spent on raw pretrain-mix LM "
                   "batches instead of chat (e.g. 0.03 → every ~33rd "
                   "step) — anchors the base distribution against SFT "
                   "forgetting (gen-3: chat-only SFT cost +0.74 bpb on "
                   "the enwik8 test slice and destroyed novel-name "
                   "induction). 0 disables")
    g.add_argument("--replay-mix", type=Path, default=None,
                   help="Mixture config for --replay-frac (normally the "
                   "generation's pretrain DATA_MIX; train slices only). "
                   "Corpora stay in CPU memory — no GPU cost")

    g = p.add_argument_group("logging & checkpointing")
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--eval-batches", type=int, default=32)
    g.add_argument("--probe-every", type=int, default=0,
                   help="Run the behavior probe suite (transformer.probes, "
                   "chat forms + raw verbatim) every N steps — probe/* "
                   "scalars into TB and metrics.jsonl. 0 disables")
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
def sample_chat(model, device, user_text: str, tok, max_new: int = 128) -> str:
    was_training = model.training
    model.eval()
    try:
        ids = torch.tensor([make_prompt_ids(user_text, tok)],
                           device=device, dtype=torch.long)
        out = generate(model, ids, max_new_tokens=max_new)
        if tok.end_turn_id in out:
            out = out[: out.index(tok.end_turn_id)]
        return tok.decode(out).strip()
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
    tok = load_tokenizer(getattr(cfg, "tokenizer", "bytes"))
    # Per-token-id byte lengths: bpb below = bits per assistant BYTE,
    # comparable across tokenizers (all-ones for byte models).
    byte_lens = torch.tensor(tok.byte_lengths(), dtype=torch.float32,
                             device=device)

    if args.out is not None:
        out_dir = args.out
    elif args.resume is not None:
        out_dir = args.resume.parent.resolve()
    else:
        name = args.run_name or derive_run_name("sft", ckpt, args.init)
        out_dir = (PROJECT_ROOT / "checkpoints" / "sft" / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lineage: identity comes from the loaded checkpoint's metadata (or,
    # for pre-metadata checkpoints, its dir name); this run appends
    # itself to the ancestor chain.
    identity = checkpoint_identity(ckpt, src)
    if args.resume is not None:
        lineage = ckpt.get("lineage") or [out_dir.name]
    else:
        lineage = (ckpt.get("lineage") or [src.resolve().parent.name]) + [out_dir.name]

    log_file = open(out_dir / "train.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    metrics_f = open_metrics_log(out_dir / "metrics.jsonl")

    print(f"run dir: {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}")
    print(f"model:   {preset} ({type(model).__name__}), "
          f"{model.num_parameters():,} params, seq_len={cfg.max_seq_len}, device={device}")
    print(f"lineage: {' -> '.join(lineage)}")

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
    train_convs, val_convs = load_conversations(
        data_paths, args.seed, args.val_frac, seq_len=args.seq_len, tok=tok)
    n_bytes = sum(len(c) for c in train_convs)
    print(f"data: {len(train_convs):,} train / {len(val_convs):,} val examples "
          f"({n_bytes / 1e6:.1f} MB train; long conversations split at turn "
          f"boundaries to fit seq_len) from {[p.name for p in data_paths]}")

    # Cold-start diagnostic: a fixed slice of the synthetic task corpus,
    # evaluated separately so task-following progress is visible apart
    # from general chat loss. (Overlaps the train pool — a diagnostic,
    # not a held-out benchmark.)
    tasks_path = PROJECT_ROOT / "data" / "chat_tasks.txt"
    val_tasks_convs = (split_conversations(tasks_path.read_bytes())[:128]
                       if tasks_path.exists() else [])

    # Per-source diagnostics (same caveat as tasks_bpb: fixed slices
    # overlapping the train pool). One bpb curve per corpus answers
    # "which register is being learned" — the SFT analogue of
    # pretrain's val_domain metrics. Evaluated every 4th eval.
    source_val = {
        p.stem.removeprefix("chat_"): split_conversations(p.read_bytes())[:32]
        for p in data_paths
    }

    # Pretrain replay: raw LM batches from the pretrain mix, interleaved
    # at --replay-frac. Corpora load onto CPU (the token cache from the
    # pretrain stage is reused — same file/range/tokenizer keys); each
    # replay batch is a tiny host→device copy, so the resident GPU
    # footprint is zero.
    replay_data, replay_weights, replay_interval = None, None, 0
    if args.replay_frac > 0:
        if args.replay_mix is None:
            raise SystemExit("--replay-frac requires --replay-mix")
        r_paths, r_mults, r_splits, _ = load_data_mix(args.replay_mix)
        replay_data, _rv, _rvb, _rvp, r_bytes = load_data(
            r_paths, torch.device("cpu"), r_splits, tok=tok)
        replay_weights = torch.tensor(
            [b * m for b, m in zip(r_bytes, r_mults)])
        replay_interval = max(int(round(1.0 / args.replay_frac)), 2)
        print(f"replay: {len(r_paths)} corpora from {args.replay_mix}, "
              f"1 raw LM batch every {replay_interval} steps "
              f"({1 / replay_interval:.1%} of steps)")

    manifest_path = out_dir / "run.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": out_dir.name,
            "identity": identity,
            "lineage": lineage,
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

    probe_runner, probe_round = None, 0
    if args.probe_every > 0:
        from transformer.probes import ProbeRunner
        probe_runner = ProbeRunner(model, tok, device, chat=True,
                                   seed=args.seed, preamble=DEFAULT_PREAMBLE,
                                   facts_per_family=8)
        print(f"probes: chat forms every {args.probe_every} steps")
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
                        tokens_seen=tokens_seen, device=device,
                        run_name=out_dir.name, identity=identity, stage="sft",
                        lineage=lineage)

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

            is_replay = (replay_data is not None
                         and step % replay_interval == 0)
            if is_replay:
                # Raw LM batch from the pretrain mix — full-token CE, no
                # assistant mask. Same optimizer step; different loss
                # stream (train/replay_bpb).
                inputs, targets = get_batch(
                    replay_data, replay_weights, args.batch_size, args.seq_len)
                inputs, targets = inputs.to(device), targets.to(device)
                mask = None
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size), targets.view(-1))
            else:
                idx = torch.randint(0, len(train_convs), (args.batch_size,)).tolist()
                inputs, targets, mask = conversation_batch(
                    train_convs, idx, args.seq_len, device, tok=tok)
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

            if is_replay:
                # Replay steps are rare (~1 in 33) — log every one, so
                # train/replay_bpb is a dense curve of base-LM health.
                loss_val = float(loss.item())
                replay_bpb = (loss_val * targets.numel() / LN2
                              / max(float(byte_lens[targets].sum()), 1.0))
                print(f"step {step:>5}/{args.steps}  replay_loss={loss_val:.4f}  "
                      f"replay_bpb={replay_bpb:.3f}  lr={lr:.2e}")
                emit(metrics_f, event="replay", step=step,
                     replay_loss=loss_val, replay_bpb=replay_bpb, lr=lr,
                     tokens_seen=tokens_seen)
                if writer is not None:
                    writer.add_scalar("train/replay_bpb", replay_bpb, step)
            elif is_log_step:
                elapsed = time.perf_counter() - t_start
                tps = n_done * args.batch_size * args.seq_len / elapsed
                eta = (elapsed / n_done) * (args.steps - step)
                loss_val = float(loss.item())
                n_masked = int(mask.sum().item())
                train_bpb = (loss_val * n_masked / LN2
                             / max(float(byte_lens[targets[mask]].sum()), 1.0))
                print(f"step {step:>5}/{args.steps}  sft_loss={loss_val:.4f}  "
                      f"bpb={train_bpb:.3f}  grad_norm={float(grad_norm.item()):.2f}  "
                      f"lr={lr:.2e}  ({tps:>6.0f} tok/s)  ETA {fmt_eta(eta)}")
                emit(metrics_f, event="step", step=step, train_loss=loss_val,
                     bpb=train_bpb, lr=lr, grad_norm=float(grad_norm.item()),
                     tok_per_sec=tps, tokens_seen=tokens_seen,
                     moe_layers=monitor.detail(), **monitor.summary())
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/bpb", train_bpb, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm.item()), step)

            if step % args.eval_every == 0:
                val, val_bpb = masked_conversation_loss(
                    model, val_convs, args.batch_size, args.seq_len,
                    args.eval_batches, device, tok=tok, byte_lens=byte_lens)
                tasks_bpb = None
                if val_tasks_convs:
                    _, tasks_bpb = masked_conversation_loss(
                        model, val_tasks_convs, args.batch_size, args.seq_len,
                        4, device, tok=tok, byte_lens=byte_lens)
                is_best = val < best_val
                if is_best:
                    best_val, best_step = val, step
                    full_save(out_dir / "best.pt", step)
                source_bpb = {}
                if step % (args.eval_every * 4) == 0:
                    for name, cs in source_val.items():
                        if not cs:
                            continue
                        n_b = -(-len(cs) // args.batch_size)  # ceil
                        _, s_bpb = masked_conversation_loss(
                            model, cs, args.batch_size, args.seq_len, n_b,
                            device, tok=tok, byte_lens=byte_lens)
                        source_bpb[name] = round(s_bpb, 4)
                tasks_str = (f"  tasks_bpb={tasks_bpb:.3f}"
                             if tasks_bpb is not None else "")
                print(f"        val_loss={val:.4f}  val_bpb={val_bpb:.3f}"
                      f"{tasks_str}" + ("  ← best" if is_best else ""))
                if source_bpb:
                    print("        src: " + "  ".join(
                        f"{k}={v:.3f}" for k, v in sorted(source_bpb.items())))
                emit(metrics_f, event="eval", step=step, val_loss=val,
                     val_bpb=val_bpb, val_tasks_bpb=tasks_bpb,
                     source_bpb=source_bpb or None,
                     tokens_seen=tokens_seen, is_best=is_best)
                if writer is not None:
                    writer.add_scalar("val/loss", val, step)
                    writer.add_scalar("val/bpb", val_bpb, step)
                    if tasks_bpb is not None:
                        writer.add_scalar("val/tasks_bpb", tasks_bpb, step)
                    for name, v in source_bpb.items():
                        writer.add_scalar(f"val_source/{name}", v, step)
                    writer.flush()
                sync.kick()

            if step % args.sample_every == 0 and not args.no_sample:
                for prompt in SAMPLE_PROMPTS:
                    reply = sample_chat(model, device, prompt, tok)
                    print(f"--- {prompt!r}\n{reply}\n---")
                    emit(metrics_f, event="sample", step=step, prompt=prompt, text=reply)
                    if writer is not None:
                        writer.add_text("samples", f"**{prompt}**\n\n{reply}", step)

            if probe_runner is not None and step % args.probe_every == 0:
                t_probe = time.perf_counter()
                scalars = probe_runner.run()
                dumps = probe_runner.probe_dumps(rotation=probe_round, k=2)
                probe_round += 1
                dt = time.perf_counter() - t_probe
                brief = "  ".join(
                    f"{k.removeprefix('probe/')}={v:.2f}"
                    for k, v in sorted(scalars.items())
                    if k in ("probe/identity/novel", "probe/identity/pool",
                             "probe/verbatim/heldout"))
                print(f"        probes: {len(scalars)} scalars in {dt:.0f}s  {brief}")
                emit(metrics_f, event="probe", step=step, scalars=scalars,
                     elapsed_s=dt)
                for d in dumps:
                    emit(metrics_f, event="probe_dump", step=step, **d)
                if writer is not None:
                    for k, v in scalars.items():
                        writer.add_scalar(k, v, step)
                    for d in dumps:
                        writer.add_text(
                            f"probe_dump/{d['id']}",
                            f"**{d['prompt']}**\n\n```\n{d['text']}\n```", step)

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
