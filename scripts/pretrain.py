"""Pre-train a model preset on byte-level text corpora (stage 1 of 3).

Examples:

    First run (auto-generated run name in checkpoints/pretrain/):
        .venv/bin/python scripts/pretrain.py --steps 500

    Train the Kimi K3 miniature (KDA runs a sequential scan on non-CUDA —
    prefer a modest --seq-len):
        .venv/bin/python scripts/pretrain.py --preset kimi3 --seq-len 256 --steps 2000

    The full mixture with reserved splits:
        .venv/bin/python scripts/pretrain.py --data-mix configs/mix-downweight-wiki.json

    Resume training (uses the same directory; model, optimizer, RNG
    streams, and best-val tracking all continue where they left off):
        .venv/bin/python scripts/pretrain.py \\
            --resume checkpoints/pretrain/<run>/latest.pt --steps 10000

    Watch runs in TensorBoard (stages group as pretrain/ sft/ rlvr/):
        .venv/bin/tensorboard --logdir checkpoints

Stages: pretrain (this script) → sft.py → grpo.py. Run dirs live at
checkpoints/pretrain/<run-name> and mirror to
gs://<bucket>/runs/pretrain/<run-name>; each contains:

    step_NNNN.pt    rolling checkpoints (oldest pruned to --keep-last)
    latest.pt       symlink to most recent
    best.pt         full checkpoint, replaced whenever val_loss hits a new low
    interrupted.pt  saved on Ctrl-C if training is interrupted
    run.json        manifest: args, config, datasets, start time
    metrics.jsonl   append-only event log (steps / evals / saves / samples)
    train.log       full stdout, mirrored from console
    tb/             TensorBoard event files

What gets tracked (metrics.jsonl and TensorBoard):

    train/loss, train/bpb   cross-entropy in nats and bits-per-byte
    train/lr                the scheduled learning rate
    train/grad_norm         pre-clip global gradient norm
    train/tok_per_sec, train/tokens_seen, train/mem_gb
    val/loss, val/bpb       deterministic fixed-window eval (pinned
                            windows per domain — see transformer.eval)
    val/best                best trailing-3 mean of val_bpb (drives best.pt)
    val/acc, val/top5, val/ent      aggregate accuracy/top-5/entropy
    val_domain/*            per-domain bpb; val_domain_{acc,top5,ent}/*
    val_group/*             byte-weighted per-group bpb (mix "groups")
    probe/*                 behavior probes (--probe-every; transformer.probes)
    params/global_norm      global L2 norm of all weights (on evals)
    moe/L*_max_load         per-MoE-layer max expert load fraction
    moe/L*_entropy          per-MoE-layer normalized routing entropy
    moe/L*_bias_span        balancing-bias spread (DeepSeek/Kimi routers)

Checkpoints embed the model config, optimizer state, RNG streams (CPU +
device), best-val tracking, and cumulative token count, so a resumed run
is bit-for-bit the run that would have happened without the interruption.
Checkpoint writes are atomic (tmp file + rename), so Ctrl-C can never
leave a truncated checkpoint behind.

Training runs in fp32 for stability; trained weights can be cast to bf16
at inference time. Run from the project root.
"""

import argparse
import json
import math
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

from transformer import MODELS, build_model, generate
from transformer.data import get_batch, load_data, load_data_mix, parse_weights
from transformer.eval import fixed_window_eval, pin_val_windows
from transformer.tokenizer import load_tokenizer

from run_utils import (
    BEST_WINDOW,
    BucketSync,
    MoEMonitor,
    Tee,
    device_mem_gb,
    emit,
    fmt_eta,
    generate_run_name,
    global_param_norm,
    lr_at,
    open_metrics_log,
    prune_old_checkpoints,
    recover_best_from_metrics,
    restore_rng_state,
    save_checkpoint,
    strip_run_identity,
    update_symlink,
    windowed_best_val,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed — script still works
    SummaryWriter = None

LN2 = math.log(2.0)


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
        "deepseek (MLA + DeepSeekMoE), kimi3 (KDA/MLA hybrid + AttnRes + LatentMoE). "
        "Ignored when resuming — the checkpoint's config wins",
    )
    g.add_argument(
        "--tokenizer",
        type=str,
        default="bytes",
        help="'bytes' (vocab 256) or a trained artifact name like 'bpe4k' "
        "(configs/tokenizer-bpe4k.json). Sets vocab_size and is embedded "
        "in the config, so every downstream stage inherits it. Ignored "
        "when resuming",
    )

    g = p.add_argument_group("training")
    g.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Total training steps (target — counts steps from any resumed checkpoint)",
    )
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--seq-len", type=int, default=2048,
                   help="Sequence length per batch. 2048 is the next-gen "
                   "default (2026-08-08): KDA layers are O(L), so on CUDA "
                   "long windows are cheap; only the MLA quarter pays O(L²)")
    g.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    g.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("cosine", "constant", "wsd"),
        help="cosine: linear warmup then cosine decay to --min-lr-frac * lr "
        "(schedule is a pure function of step, so it resumes exactly; "
        "note it is shaped by --steps, so extending --steps on resume "
        "reshapes the tail). wsd: warmup, hold at peak, linear decay "
        "over the final --decay-frac of --steps — extending --steps "
        "while in the hold just extends the hold (gen-4 default; see "
        "run_utils.lr_at)",
    )
    g.add_argument(
        "--warmup-frac",
        type=float,
        default=0.01,
        help="Fraction of --steps spent in linear warmup (cosine schedule)",
    )
    g.add_argument(
        "--min-lr-frac",
        type=float,
        default=0.1,
        help="Final LR as a fraction of --lr (cosine schedule)",
    )
    g.add_argument(
        "--decay-frac",
        type=float,
        default=0.15,
        help="wsd only: fraction of --steps spent in the final linear decay",
    )
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")

    g = p.add_argument_group("logging & evaluation")
    g.add_argument(
        "--log-every", type=int, default=25, help="Log train metrics every N steps"
    )
    g.add_argument(
        "--eval-every", type=int, default=100, help="Run val eval every N steps"
    )
    g.add_argument(
        "--eval-windows",
        type=int,
        default=16,
        help="Pinned windows scored per val domain at every eval "
        "(deterministic — the same windows every time; see "
        "transformer.eval.pin_val_windows)",
    )
    g.add_argument(
        "--probe-every",
        type=int,
        default=0,
        help="Run the behavior probe suite (transformer.probes, raw "
        "continuation forms) every N steps: probe/* scalars + rotating "
        "generation dumps into TB and metrics.jsonl. 0 disables. "
        "Budget ~1 min of GPU per round — pick N ≈ 4× --eval-every",
    )
    g.add_argument(
        "--sample-every", type=int, default=100, help="Generate a sample every N steps"
    )
    g.add_argument("--sample-tokens", type=int, default=200)
    g.add_argument("--sample-prompt", type=str, default="ROMEO:\n")
    g.add_argument("--no-sample", action="store_true")
    g.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard event files (tb/ subdir)",
    )
    g.add_argument(
        "--no-bucket-sync",
        action="store_true",
        help="Don't mirror the run dir (checkpoints, tb, metrics, logs) to "
        "gs://<bucket>/runs/pretrain/<run-name>. Sync is on by default whenever "
        "configs/gcs.json exists and gcloud is on PATH.",
    )

    g = p.add_argument_group("checkpointing")
    g.add_argument("--save-every", type=int, default=100)
    g.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Run output directory. Default: checkpoints/pretrain/{run-name}/",
    )
    g.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Auto-generated if omitted (e.g. 'kimi3-small-17M-calm-river-20260503-141522')",
    )
    g.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from (uses its parent dir as --out). "
        "The model is rebuilt from the checkpoint's own config",
    )
    g.add_argument(
        "--keep-last",
        type=int,
        default=5,
        help="Keep only N most recent checkpoints (best.pt's target is protected)",
    )
    g.add_argument(
        "--keep-every",
        type=int,
        default=0,
        help="Exempt milestone checkpoints (step %% N == 0) from pruning — "
        "sparse keepers for post-hoc analysis (gen-4 recipe: 25000). "
        "0 disables",
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
    g.add_argument(
        "--data-mix",
        type=Path,
        default=None,
        help="JSON mixture config: {\"include\": <glob or list of globs>, "
        "\"exclude\": <glob(s)>, \"multipliers\": {<path or filename>: <mult>}, "
        "\"splits\": {...}}. Sampling weight is byte_size × multiplier. "
        "Mutually exclusive with --data/--data-weights.",
    )
    g.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Legacy --data runs only: hold out the last fraction of every "
        "file for validation. --data-mix runs use the config's per-file "
        "\"splits\" instead (files without one train on 100%% of their bytes).",
    )

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="mps")

    return p.parse_args()


def resolve_run_dir(args: argparse.Namespace, preset: str, n_params: int) -> Path:
    """Pick the output directory based on --out / --resume / --run-name."""
    if args.out is not None:
        return args.out
    if args.resume is not None:
        return args.resume.parent.resolve()
    name = args.run_name or generate_run_name(preset, n_params)
    return (PROJECT_ROOT / "checkpoints" / "pretrain" / name).resolve()


@torch.no_grad()
def sample_text(model, device, prompt: str, n_tokens, tok):
    was_training = model.training
    model.eval()
    try:
        ids = torch.tensor([tok.encode(prompt)], device=device, dtype=torch.long)
        out_ids = generate(model, ids, max_new_tokens=n_tokens)
        return prompt + tok.decode(out_ids)
    finally:
        if was_training:
            model.train()


def write_run_manifest(path, args, cfg, preset, n_params, byte_counts, weights):
    """One-shot snapshot of the run setup. Written at start; never updated."""
    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": path.parent.name,
        "identity": strip_run_identity(path.parent.name),
        "kind": "pretrain",
        "preset": preset,
        "n_params": n_params,
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # TF32 matmuls on Ampere+ — big speedup for fp32 training, negligible
    # precision cost. No-op on non-CUDA devices.
    torch.set_float32_matmul_precision("high")

    # Build the model BEFORE resolving the run dir — auto-generated run
    # names lead with the preset and parameter count. On a fresh run the
    # preset decides the architecture; on resume the checkpoint's embedded
    # config is authoritative, so a wrong --preset/--seq-len flag can't
    # silently build a mismatched model. Notes are printed once the
    # train.log tee is open so they're captured in the log.
    notes = []
    ckpt = None
    if args.resume is not None:
        notes.append(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        preset = ckpt.get("preset", args.preset)
        if preset != args.preset and "preset" in ckpt:
            notes.append(f"  note: checkpoint was trained with --preset {preset}; using that")
        if cfg.max_seq_len != args.seq_len:
            notes.append(
                f"  note: checkpoint config has max_seq_len={cfg.max_seq_len}; "
                f"overriding --seq-len {args.seq_len}"
            )
            args.seq_len = cfg.max_seq_len
        model = build_model(cfg).to(device)
    else:
        preset = args.preset
        config_cls, model_cls = MODELS[preset]
        _tok = load_tokenizer(args.tokenizer)
        cfg = config_cls(dtype=torch.float32, max_seq_len=args.seq_len,
                         vocab_size=_tok.vocab_size, tokenizer=_tok.name)
        model = model_cls(cfg).to(device)
    tok = load_tokenizer(getattr(cfg, "tokenizer", "bytes"))
    # Per-token-id byte lengths: every logged bpb below divides bits by
    # RAW BYTES, so curves are comparable across tokenizers and to the
    # enwik8 literature. (For byte models this table is all ones and
    # numbers are identical to the old per-token logging.)
    byte_lens = torch.tensor(tok.byte_lengths(), dtype=torch.float32,
                             device=device)

    out_dir = resolve_run_dir(args, preset, model.num_parameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out = out_dir

    # Lineage identity: from the checkpoint on resume (authoritative),
    # else derived from this new run's own name.
    identity = (ckpt.get("identity") if ckpt is not None else None) \
        or strip_run_identity(out_dir.name)
    lineage = (ckpt.get("lineage") if ckpt is not None else None) or [out_dir.name]

    # Mirror stdout to a per-run train.log. Closed in the finally block.
    log_file = open(out_dir / "train.log", "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = open_metrics_log(metrics_path)

    for note in notes:
        print(note)
    print(
        f"run dir: {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}"
    )
    print(
        f"model:   {preset} ({type(model).__name__}), {model.num_parameters():,} params, "
        f"dtype={cfg.dtype}, tokenizer={tok.name} (vocab {tok.vocab_size}), "
        f"device={device}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Restore training state from the checkpoint: weights, optimizer
    # moments, step counter, best-val tracking, token count, RNG streams.
    start_step = 0
    tokens_seen = 0
    # best_val is the min TRAILING-BEST_WINDOW mean of val_bpb (see
    # run_utils.BEST_WINDOW for why single-eval best was retired).
    best_val = float("inf")
    best_step = None
    val_bpb_hist: list[float] = []
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", start_step * args.batch_size * args.seq_len))
        if "val_bpb_hist" in ckpt:  # windowed-bpb era checkpoint
            val_bpb_hist = list(ckpt["val_bpb_hist"])
            if ckpt.get("best_val") is not None:
                best_val, best_step = ckpt["best_val"], ckpt.get("best_step")
        else:  # older checkpoint (best_val was single-eval val_loss) —
            # rebuild under the current windowed-val_bpb semantics
            best_val, best_step = recover_best_from_metrics(metrics_path)
        if restore_rng_state(ckpt.get("rng"), device):
            print(f"resumed at step {start_step} (RNG streams restored)")
        else:
            print(f"resumed at step {start_step}")
        if best_step is not None:
            print(f"best so far: {best_val:.4f} at step {best_step} "
                  f"(trailing-{BEST_WINDOW} val_bpb)")
        del ckpt  # free the 100s-of-MB state dict copy

    if start_step >= args.steps:
        print(f"nothing to do — start_step ({start_step}) >= --steps ({args.steps}).")
        return

    mix_mults = None
    splits = None
    mix_groups: dict = {}
    if args.data_mix is not None:
        if args.data or args.data_weights:
            raise SystemExit("--data-mix is mutually exclusive with --data/--data-weights")
        args.data, mix_mults, splits, mix_groups = load_data_mix(args.data_mix)
        print(f"data mix: {args.data_mix} ({len(args.data)} files, "
              f"{len(set(mix_groups.values()))} groups)")
    if not args.data:
        args.data = [PROJECT_ROOT / "data" / "tinyshakespeare.txt"]
    if splits is None:
        # Legacy --data path: hold out the last val_frac of every file.
        splits = {p: (1.0 - args.val_frac, args.val_frac) for p in args.data}

    train_data, val_data, val_bytes, val_paths, byte_counts = load_data(
        args.data, device, splits, tok=tok)
    if mix_mults is not None:
        weights = torch.tensor([b * m for b, m in zip(byte_counts, mix_mults)])
    else:
        weights = parse_weights(args.data_weights, byte_counts)
    norm_weights = weights / weights.sum()
    # Pinned eval windows: every eval scores the SAME seeded windows per
    # val slice — deterministic curves, equal per-domain coverage, and a
    # best.pt criterion no lucky draw can hold (gen-3 lessons 2+3).
    val_windows = pin_val_windows(val_data, val_paths, args.seq_len,
                                  args.eval_windows, seed=args.seed)
    have_val = any(val_windows)
    if not have_val:
        print("note: no file defines a val split — skipping evals and best.pt tracking")
    # stem -> group name, for val_group/* aggregates.
    val_groups = {Path(p).stem: mix_groups[p] for p in val_paths
                  if p in mix_groups}

    print(f"datasets ({len(args.data)}):")
    for path, n_bytes, w in zip(args.data, byte_counts, norm_weights.tolist()):
        rel = (
            path.relative_to(PROJECT_ROOT)
            if path.is_absolute() and path.is_relative_to(PROJECT_ROOT)
            else path
        )
        train_frac, val_frac = splits.get(path, (1.0, 0.0))
        note = ""
        if (train_frac, val_frac) != (1.0, 0.0):
            test_frac = 1.0 - train_frac - val_frac
            note = f"  [train {train_frac:.0%} / val {val_frac:.0%}" + (
                f" / test {test_frac:.0%} reserved]" if test_frac > 1e-9 else "]"
            )
        print(f"  {str(rel):<42}  {n_bytes:>10,} bytes  ({w * 100:>5.1f}% sampling){note}")

    # Run manifest (only on first start; preserves original on resume).
    manifest_path = out_dir / "run.json"
    if not manifest_path.exists():
        write_run_manifest(
            manifest_path, args, cfg, preset, model.num_parameters(),
            byte_counts, norm_weights,
        )

    # TensorBoard writer. Resumed runs append to the same tb/ dir; steps
    # continue from where they left off, so curves join up seamlessly.
    writer = None
    if not args.no_tensorboard:
        if SummaryWriter is None:
            print("tensorboard not installed — skipping (pip install tensorboard)")
        else:
            writer = SummaryWriter(log_dir=str(out_dir / "tb"))
            if start_step == 0:
                config_md = json.dumps(
                    {k: str(v) for k, v in asdict(cfg).items()}, indent=2
                )
                writer.add_text("run/config", f"```\n{config_md}\n```", 0)
                writer.add_text("run/preset", preset, 0)

    monitor = MoEMonitor(model)
    if monitor.names:
        print(f"tracking {len(monitor.names)} MoE layers for load balance")

    sync = BucketSync(out_dir, enabled=not args.no_bucket_sync, stage="pretrain")
    print(f"bucket sync: {sync.dest or 'off'}")

    print(
        f"training from step {start_step + 1} to {args.steps} "
        f"(batch={args.batch_size}, seq_len={args.seq_len}, lr={args.lr}, "
        f"schedule={args.lr_schedule})"
    )
    print()

    emit(
        metrics_f,
        event="start",
        step=start_step,
        kind="pretrain",
        preset=preset,
        n_params=model.num_parameters(),
        total_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        resumed=args.resume is not None,
    )

    # Set up SIGINT (Ctrl-C) handler. We don't exit immediately — we set a
    # flag and let the next iteration of the loop save cleanly and then exit.
    interrupted = {"flag": False}

    def sigint_handler(_signum, _frame):
        interrupted["flag"] = True
        print("\n[SIGINT] will save and exit at next step boundary...")

    signal.signal(signal.SIGINT, sigint_handler)

    def full_save(path, step):
        save_checkpoint(
            path, model, optimizer, step, cfg,
            preset=preset, best_val=best_val, best_step=best_step,
            tokens_seen=tokens_seen, device=device,
            run_name=out_dir.name, identity=identity, stage="pretrain",
            lineage=lineage,
            # Trailing val_bpb window so windowed-best tracking resumes
            # exactly (see run_utils.BEST_WINDOW).
            extra={"val_bpb_hist": val_bpb_hist[-BEST_WINDOW:]},
        )

    probe_runner, probe_round = None, 0
    if args.probe_every > 0:
        from transformer.probes import ProbeRunner
        probe_runner = ProbeRunner(model, tok, device, chat=False,
                                   seed=args.seed, facts_per_family=8)
        print(f"probes: raw-continuation forms every {args.probe_every} steps")

    model.train()
    t_start = time.perf_counter()
    n_done = 0
    last_step_done = start_step

    try:
        for step in range(start_step + 1, args.steps + 1):
            if interrupted["flag"]:
                break

            lr = lr_at(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            is_log_step = step == start_step + 1 or step % args.log_every == 0
            monitor.enabled = is_log_step

            inputs, targets = get_batch(
                train_data, weights, args.batch_size, args.seq_len
            )
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

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
                steps_left = args.steps - step
                eta = (elapsed / n_done) * steps_left
                loss_val = float(loss.item())
                gnorm = float(grad_norm.item())
                mem = device_mem_gb(device)
                train_bpb = (loss_val * targets.numel() / LN2
                             / float(byte_lens[targets].sum()))
                with torch.no_grad():  # confidence proxy: mean prediction entropy (bits)
                    logp = F.log_softmax(logits.detach().float(), dim=-1)
                    ent = float((-logp.exp() * logp).sum(-1).mean()) / LN2
                print(
                    f"step {step:>5}/{args.steps}  train_loss={loss_val:.4f}  "
                    f"bpb={train_bpb:.3f}  ent={ent:.2f}  grad_norm={gnorm:.2f}  "
                    f"lr={lr:.2e}  ({tps:>6.0f} tok/s)  ETA {fmt_eta(eta)}"
                )
                emit(
                    metrics_f,
                    event="step",
                    step=step,
                    train_loss=loss_val,
                    bpb=train_bpb,
                    entropy_bits=ent,
                    lr=lr,
                    grad_norm=gnorm,
                    tok_per_sec=tps,
                    tokens_seen=tokens_seen,
                    eta_s=eta,
                    mem_gb=mem,
                    moe_layers=monitor.detail(),
                    **monitor.summary(),
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_val, step)
                    writer.add_scalar("train/bpb", train_bpb, step)
                    writer.add_scalar("train/entropy_bits", ent, step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", gnorm, step)
                    writer.add_scalar("train/tok_per_sec", tps, step)
                    writer.add_scalar("train/tokens_seen", tokens_seen, step)
                    if mem is not None:
                        writer.add_scalar("train/mem_gb", mem, step)
                    for name, load in monitor.loads.items():
                        writer.add_scalar(f"moe/{name}_max_load", float(load.max()), step)
                        writer.add_scalar(f"moe/{name}_entropy", monitor.entropy(load), step)
                    for name, span in monitor.bias_spans.items():
                        writer.add_scalar(f"moe/{name}_bias_span", span, step)

            if step % args.eval_every == 0 and have_val:
                # Deterministic fixed-window eval: bpb + accuracy/top-5/
                # entropy per domain. The triple decomposes bpb moves
                # live: bpb↑+acc-flat+ent↓ = confidence misallocation
                # (gen-3's war_and_peace shape); bpb↑+acc↓ = rank loss.
                agg, domains = fixed_window_eval(
                    model, val_data, val_paths, val_windows, args.seq_len,
                    args.batch_size, byte_lens=byte_lens,
                    agg_weights=[float(b) for b in val_bytes])
                val, val_bpb = agg["loss"], agg["bpb"]
                val_bpb_hist.append(val_bpb)
                windowed = windowed_best_val(val_bpb_hist)
                is_best = windowed < best_val
                if is_best:
                    best_val = windowed
                    best_step = step
                    # Save best.pt immediately at the eval that produced the
                    # new best — no eval-vs-save alignment gap.
                    full_save(out_dir / "best.pt", step)
                pnorm = global_param_norm(model)
                domain_bpb = {k: round(v["bpb"], 4) for k, v in domains.items()}
                domain_acc = {k: round(v["acc"], 4) for k, v in domains.items()}
                domain_top5 = {k: round(v["top5"], 4) for k, v in domains.items()}
                domain_ent = {k: round(v["ent"], 4) for k, v in domains.items()}
                # Byte-weighted group aggregates (val_group/*): one honest
                # curve per corpus group instead of N noisy per-file ones.
                gw: dict[str, float] = {}
                gsum: dict[str, float] = {}
                for p, b in zip(val_paths, val_bytes):
                    stem = Path(p).stem
                    g = val_groups.get(stem)
                    if g is not None and stem in domains:
                        gw[g] = gw.get(g, 0.0) + b
                        gsum[g] = gsum.get(g, 0.0) + b * domains[stem]["bpb"]
                group_bpb = {g: round(gsum[g] / gw[g], 4) for g in gw}
                print(
                    f"        val_loss={val:.4f}  val_bpb={val_bpb:.3f}  "
                    f"acc={agg['acc']:.3f}  ent={agg['ent']:.2f}"
                    + ("  ← best" if is_best else "")
                    + ("  [" + "  ".join(f"{k}={v:.3f}"
                                         for k, v in domain_bpb.items()) + "]"
                       if len(domain_bpb) > 1 else "")
                )
                if group_bpb:
                    print("        grp: " + "  ".join(
                        f"{k}={v:.3f}" for k, v in sorted(group_bpb.items())))
                emit(
                    metrics_f,
                    event="eval",
                    step=step,
                    val_loss=val,
                    val_bpb=val_bpb,
                    val_acc=agg["acc"],
                    val_top5=agg["top5"],
                    val_ent=agg["ent"],
                    domain_bpb=domain_bpb,
                    domain_acc=domain_acc,
                    domain_top5=domain_top5,
                    domain_ent=domain_ent,
                    group_bpb=group_bpb or None,
                    tokens_seen=tokens_seen,
                    param_norm=pnorm,
                    is_best=is_best,
                )
                if writer is not None:
                    writer.add_scalar("val/loss", val, step)
                    writer.add_scalar("val/bpb", val_bpb, step)
                    writer.add_scalar("val/best", best_val, step)
                    writer.add_scalar("val/acc", agg["acc"], step)
                    writer.add_scalar("val/top5", agg["top5"], step)
                    writer.add_scalar("val/ent", agg["ent"], step)
                    for name, v in domain_bpb.items():
                        writer.add_scalar(f"val_domain/{name}", v, step)
                    for name, v in domain_acc.items():
                        writer.add_scalar(f"val_domain_acc/{name}", v, step)
                    for name, v in domain_top5.items():
                        writer.add_scalar(f"val_domain_top5/{name}", v, step)
                    for name, v in domain_ent.items():
                        writer.add_scalar(f"val_domain_ent/{name}", v, step)
                    for name, v in group_bpb.items():
                        writer.add_scalar(f"val_group/{name}", v, step)
                    writer.add_scalar("params/global_norm", pnorm, step)
                    writer.flush()
                sync.kick()

            if step % args.sample_every == 0 and not args.no_sample:
                text = sample_text(
                    model, device, args.sample_prompt, args.sample_tokens, tok
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
                if writer is not None:
                    writer.add_text("samples", f"```\n{text}\n```", step)

            if probe_runner is not None and step % args.probe_every == 0:
                t_probe = time.perf_counter()
                scalars = probe_runner.run()
                dumps = probe_runner.probe_dumps(rotation=probe_round, k=2)
                probe_round += 1
                dt = time.perf_counter() - t_probe
                brief = "  ".join(
                    f"{k.removeprefix('probe/')}={v:.2f}"
                    for k, v in sorted(scalars.items())
                    if k in ("probe/identity/novel",
                             "probe/facts/instances/heldout",
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
                pruned = prune_old_checkpoints(out_dir, args.keep_last,
                                               args.keep_every)
                msg = f"        saved {ckpt_path.name}  (latest -> {ckpt_path.name})"
                if pruned:
                    msg += f"  pruned {len(pruned)}"
                print(msg)
                emit(
                    metrics_f,
                    event="save",
                    step=step,
                    path=ckpt_path.name,
                    pruned=len(pruned),
                )
                sync.kick()
    finally:
        # Always: save an interrupted/final checkpoint so we never lose work.
        elapsed = time.perf_counter() - t_start
        if interrupted["flag"]:
            ckpt_path = out_dir / "interrupted.pt"
            full_save(ckpt_path, last_step_done)
            update_symlink(out_dir, "latest.pt", ckpt_path)
            print(f"\ninterrupted at step {last_step_done} after {elapsed:.1f}s")
            print(f"  saved {ckpt_path.name}; resume with --resume {ckpt_path}")
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
                full_save(final, last_step_done)
                update_symlink(out_dir, "latest.pt", final)
                prune_old_checkpoints(out_dir, args.keep_last, args.keep_every)
            print(f"final checkpoint: {final.name}")
            if best_step is not None:
                print(f"best val_bpb={best_val:.4f} at step {best_step} "
                      f"(best.pt, trailing-{BEST_WINDOW} mean)")
            emit(
                metrics_f,
                event="end",
                step=last_step_done,
                elapsed_s=elapsed,
                tokens_seen=tokens_seen,
                best_val=best_val if best_step else None,
                best_step=best_step,
            )
        if writer is not None:
            writer.close()
        metrics_f.close()
        # One last blocking sync so the final checkpoint and full logs land
        # in the bucket before we exit.
        sync.finalize()
        # Restore stdout/stderr and close the log file.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
