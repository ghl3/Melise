"""GRPO on verifiable rewards (stage 3 of 3, after sft.py).

Examples:

    Train an SFT checkpoint against all reward tasks:
        .venv/bin/python scripts/grpo.py \\
            --init checkpoints/sft/sft-kimi3-small-17M-.../best.pt --steps 200

    A subset of tasks, bigger groups:
        .venv/bin/python scripts/grpo.py --init ... --tasks copy,parity --group-size 16

The algorithm and environment live in transformer.rl (rollout.py,
grpo.py, tasks.py — see their docstrings for the math); this script owns
the run: hyperparameters, the rollout→score→update loop, logging,
checkpoints, bucket sync.

Per step: sample --prompts-per-step tasks; decode --group-size
completions per prompt (one KV-cache batch each); score with the task's
deterministic reward, minus --no-stop-penalty when a completion never
emits the end-of-turn byte; z-score rewards within each group
(transformer.rl.group_advantages); update with the clipped surrogate +
k3 KL to the frozen --init reference (transformer.rl.grpo_loss).

Rollouts sample at --temperature from raw logits (log-probs always come
from untempered logits; temperature ≠ 1 is exploration, mildly
off-policy). Evals decode greedily on a fixed seeded task set and drive
best.pt. NOTE: unlike pretrain.py/sft.py, best_val stored in checkpoints
is the eval mean *reward* — higher is better.

Run dirs live at checkpoints/rlvr/rlvr-<name> and mirror to
gs://<bucket>/runs/rlvr/<name>; TensorBoard at --logdir checkpoints
groups runs by stage. Checkpoints keep the pretrain.py payload, so
sample.py works on them. Resume with --resume (the reference is rebuilt
from the init path recorded in run.json).
"""

import argparse
import json
import random
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from transformer import build_model
from transformer.chat import make_prompt_ids
from transformer.tokenizer import load_tokenizer
from transformer.rl import (
    TASKS,
    eval_rewards,
    gather_completion_logprobs,
    group_advantages,
    grpo_loss,
    pad_rollouts,
    rollout_batch,
    sample_tasks,
)

from run_utils import (
    BucketSync,
    Tee,
    checkpoint_identity,
    derive_run_name,
    emit,
    fmt_eta,
    open_metrics_log,
    prune_old_checkpoints,
    save_checkpoint,
    update_symlink,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO on verifiable rewards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model")
    g.add_argument("--init", type=Path, default=None,
                   help="Policy start + frozen KL reference (an sft.py best.pt)")
    g.add_argument("--resume", type=Path, default=None,
                   help="GRPO checkpoint to resume (continues its run dir)")

    g = p.add_argument_group("rollouts")
    g.add_argument("--tasks", type=str, default=",".join(TASKS),
                   help="Comma-separated task kinds (see transformer/rl/tasks.py)")
    g.add_argument("--prompts-per-step", type=int, default=16)
    g.add_argument("--group-size", type=int, default=8,
                   help="Completions sampled per prompt (the G in GRPO)")
    g.add_argument("--max-new", type=int, default=64,
                   help="Max completion bytes per rollout")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--no-stop-penalty", type=float, default=0.2,
                   help="Reward subtracted when a completion never emits END_TURN")

    g = p.add_argument_group("optimization")
    g.add_argument("--steps", type=int, default=200)
    g.add_argument("--lr", type=float, default=2e-5)
    g.add_argument("--kl-coef", type=float, default=0.05,
                   help="β on the KL-to-reference penalty")
    g.add_argument("--clip-eps", type=float, default=0.2,
                   help="PPO ratio clip ε")
    g.add_argument("--inner-epochs", type=int, default=1,
                   help="Optimizer passes over each step's rollout batch")
    g.add_argument("--update-microbatch", type=int, default=32,
                   help="Sequences per backward pass (gradient accumulation)")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--grad-clip", type=float, default=1.0)

    g = p.add_argument_group("logging & checkpointing")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--eval-every", type=int, default=20)
    g.add_argument("--eval-prompts", type=int, default=20)
    g.add_argument("--save-every", type=int, default=25)
    g.add_argument("--keep-last", type=int, default=3)
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--out", type=Path, default=None)
    g.add_argument("--no-tensorboard", action="store_true")
    g.add_argument("--no-bucket-sync", action="store_true")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", type=str, default="mps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    if (args.init is None) == (args.resume is None):
        raise SystemExit("exactly one of --init / --resume is required")
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in task_names:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t!r} (have: {', '.join(TASKS)})")

    # Resolve run dir + init path (resume rebuilds the reference from the
    # init recorded in run.json).
    if args.resume is not None:
        out_dir = args.resume.parent.resolve()
        init_path = Path(json.loads((out_dir / "run.json").read_text())["init"])
        src = args.resume
    else:
        init_path = args.init
        src = args.init

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    preset = ckpt.get("preset", "?")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = load_tokenizer(getattr(cfg, "tokenizer", "bytes"))

    ref_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
    reference = build_model(ref_ckpt["config"]).to(device)
    reference.load_state_dict(ref_ckpt["model"])
    reference.eval()
    for p_ in reference.parameters():
        p_.requires_grad_(False)
    del ref_ckpt

    if args.out is not None:
        out_dir = args.out
    elif args.resume is None:
        name = args.run_name or derive_run_name("rlvr", ckpt, init_path)
        out_dir = (PROJECT_ROOT / "checkpoints" / "rlvr" / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lineage: identity from the loaded checkpoint's metadata (legacy
    # fallback: its dir name); this run appends itself to the chain.
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
    print(f"model:   {preset} ({type(model).__name__}), {model.num_parameters():,} params, "
          f"seq_len={cfg.max_seq_len}, device={device}")
    print(f"tasks:   {', '.join(task_names)}  "
          f"(P={args.prompts_per_step} × G={args.group_size} rollouts/step, "
          f"max_new={args.max_new})")
    print(f"lineage: {' -> '.join(lineage)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))

    start_step, tokens_seen = 0, 0
    best_reward, best_step = -float("inf"), None
    if args.resume is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        if ckpt.get("best_val") is not None:
            best_reward, best_step = ckpt["best_val"], ckpt.get("best_step")
        print(f"resumed at step {start_step}")
    else:
        print(f"initialized from {init_path}")
    del ckpt

    if start_step >= args.steps:
        print(f"nothing to do — start_step ({start_step}) >= --steps ({args.steps}).")
        return

    manifest_path = out_dir / "run.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "run_name": out_dir.name,
            "identity": identity,
            "lineage": lineage,
            "kind": "rlvr-grpo",
            "preset": preset,
            "n_params": model.num_parameters(),
            "init": str(init_path),
            "tasks": task_names,
            "args": {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items()},
        }, indent=2, default=str))

    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    sync = BucketSync(out_dir, enabled=not args.no_bucket_sync, stage="rlvr")
    print(f"bucket sync: {sync.dest or 'off'}\n")

    emit(metrics_f, event="start", step=start_step, kind="rlvr-grpo", preset=preset,
         n_params=model.num_parameters(), total_steps=args.steps,
         tasks=task_names, group_size=args.group_size,
         prompts_per_step=args.prompts_per_step, lr=args.lr,
         kl_coef=args.kl_coef, resumed=args.resume is not None)

    interrupted = {"flag": False}
    signal.signal(signal.SIGINT,
                  lambda *_: (interrupted.__setitem__("flag", True),
                              print("\n[SIGINT] will save and exit at next step boundary...")))

    def full_save(path, step):
        save_checkpoint(path, model, optimizer, step, cfg, preset=preset,
                        best_val=best_reward if best_step else None,
                        best_step=best_step, tokens_seen=tokens_seen, device=device,
                        run_name=out_dir.name, identity=identity, stage="rlvr",
                        lineage=lineage)

    model.train()
    t_start = time.perf_counter()
    n_done, last_step_done = 0, start_step

    try:
        for step in range(start_step + 1, args.steps + 1):
            if interrupted["flag"]:
                break

            # ----- Rollout phase (transformer.rl.rollout) -----
            task_rng = random.Random(f"{args.seed}-{step}")
            tasks = sample_tasks(task_names, args.prompts_per_step, task_rng)
            # Rollouts batch across prompts of EQUAL length (KDA state
            # has no clean left-padding, so rectangular prefill only) —
            # task templates collide on length constantly, so most steps
            # decode several whole groups per forward pass. seqs is
            # indexed (pi·G + gi) to match the advantage flatten order.
            G = args.group_size
            prompts = [make_prompt_ids(t.prompt, tok) for t in tasks]
            seqs = [None] * (len(tasks) * G)
            rewards_pg = torch.zeros(len(tasks), G)
            kind_scores = defaultdict(list)
            by_len = defaultdict(list)
            for pi, p in enumerate(prompts):
                by_len[len(p)].append(pi)
            groups_per_batch = max(64 // G, 1)
            for length, pis in by_len.items():
                for lo in range(0, len(pis), groups_per_batch):
                    chunk = pis[lo : lo + groups_per_batch]
                    rows = [prompts[pi] for pi in chunk for _ in range(G)]
                    comps, lps, lengths = rollout_batch(
                        model, rows, args.max_new, args.temperature,
                        device, stop_id=tok.end_turn_id)
                    for j, pi in enumerate(chunk):
                        task = tasks[pi]
                        for gi in range(G):
                            comp = comps[j * G + gi]
                            r = task.score(tok.decode(comp).strip())
                            if comp[-1] != tok.end_turn_id:
                                r -= args.no_stop_penalty
                            rewards_pg[pi, gi] = r
                            kind_scores[task.kind].append(r)
                            seqs[pi * G + gi] = (
                                prompts[pi], comp,
                                lps[j * G + gi, : len(comp)].cpu())
                    tokens_seen += len(rows) * (length + max(lengths))

            # Dead groups produce zero advantage — the direct gauge of
            # the cold-start problem (tasks the policy never varies on).
            dead_frac = float((rewards_pg.std(dim=1) < 1e-6).float().mean())

            adv = group_advantages(rewards_pg).to(device)
            batch = pad_rollouts(seqs).to(device)
            total_toks = batch.total_tokens
            cls = [len(c) for _, c, _ in seqs]
            N = len(seqs)

            # ----- Update phase (transformer.rl.grpo) -----
            stats = defaultdict(float)
            for _ in range(args.inner_epochs):
                optimizer.zero_grad(set_to_none=True)
                for lo in range(0, N, args.update_microbatch):
                    hi = min(lo + args.update_microbatch, N)
                    new_lp = gather_completion_logprobs(
                        model, batch.ids[lo:hi], batch.pos[lo:hi], batch.tok[lo:hi])
                    with torch.no_grad():
                        ref_lp = gather_completion_logprobs(
                            reference, batch.ids[lo:hi], batch.pos[lo:hi],
                            batch.tok[lo:hi])
                    loss, s = grpo_loss(
                        new_lp, batch.old_lp[lo:hi], ref_lp, adv[lo:hi],
                        batch.mask[lo:hi], clip_eps=args.clip_eps,
                        kl_coef=args.kl_coef, total_tokens=total_toks)
                    loss.backward()
                    stats["kl"] += s["kl"]
                    stats["clipped"] += s["clipped"]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
                optimizer.step()
            n_done += 1
            last_step_done = step

            # ----- Logging -----
            if step == start_step + 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / n_done) * (args.steps - step)
                mean_r = float(rewards_pg.mean())
                stopped = sum(1 for _, c, _ in seqs if c[-1] == tok.end_turn_id) / N
                mean_len = sum(cls) / N
                kl_per_tok = stats["kl"] / (total_toks * args.inner_epochs)
                # Mean -log p of sampled tokens ≈ policy entropy (collapse
                # early-warning; rising confidence drives it toward 0).
                entropy = float((-batch.old_lp * batch.mask).sum() / total_toks)
                kinds = {k: sum(v) / len(v) for k, v in sorted(kind_scores.items())}
                kinds_str = "  ".join(f"{k}={v:.2f}" for k, v in kinds.items())
                print(f"step {step:>4}/{args.steps}  reward={mean_r:.3f}  "
                      f"[{kinds_str}]  len={mean_len:.0f}  stop={stopped:.0%}  "
                      f"dead={dead_frac:.0%}  ent={entropy:.2f}  "
                      f"kl={kl_per_tok:.4f}  grad={float(grad_norm.item()):.2f}  "
                      f"ETA {fmt_eta(eta)}")
                ex_task, ex = tasks[0], seqs[0]
                print(f"      e.g. {ex_task.prompt!r} -> "
                      f"{tok.decode(ex[1]).strip()!r}  r={float(rewards_pg[0, 0]):.2f}")
                emit(metrics_f, event="step", step=step, reward=mean_r,
                     reward_by_kind=kinds, mean_len=mean_len, stop_frac=stopped,
                     dead_frac=dead_frac, entropy=entropy,
                     kl=kl_per_tok, grad_norm=float(grad_norm.item()),
                     clip_frac=stats["clipped"] / (total_toks * args.inner_epochs),
                     tokens_seen=tokens_seen)
                if writer is not None:
                    writer.add_scalar("reward/mean", mean_r, step)
                    for k, v in kinds.items():
                        writer.add_scalar(f"reward/{k}", v, step)
                    writer.add_scalar("rollout/mean_len", mean_len, step)
                    writer.add_scalar("rollout/stop_frac", stopped, step)
                    writer.add_scalar("rollout/dead_frac", dead_frac, step)
                    writer.add_scalar("rollout/entropy", entropy, step)
                    writer.add_scalar("train/kl", kl_per_tok, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm.item()), step)

            if step % args.eval_every == 0:
                mean_r, kinds = eval_rewards(model, task_names, args.eval_prompts,
                                             args.seed, args.max_new, device,
                                             tok=tok)
                is_best = mean_r > best_reward
                if is_best:
                    best_reward, best_step = mean_r, step
                    full_save(out_dir / "best.pt", step)
                kinds_str = "  ".join(f"{k}={v:.2f}" for k, v in kinds.items())
                print(f"      eval reward={mean_r:.3f}  [{kinds_str}]"
                      + ("  ← best" if is_best else ""))
                emit(metrics_f, event="eval", step=step, reward=mean_r,
                     reward_by_kind=kinds, is_best=is_best)
                if writer is not None:
                    writer.add_scalar("eval/reward", mean_r, step)
                    writer.flush()
                sync.kick()

            if step % args.save_every == 0:
                ckpt_path = out_dir / f"step_{step}.pt"
                full_save(ckpt_path, step)
                update_symlink(out_dir, "latest.pt", ckpt_path)
                pruned = prune_old_checkpoints(out_dir, args.keep_last)
                print(f"      saved {ckpt_path.name}"
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
                print(f"best eval reward={best_reward:.3f} at step {best_step} (best.pt)")
            emit(metrics_f, event="end", step=last_step_done, elapsed_s=elapsed,
                 tokens_seen=tokens_seen, best_reward=best_reward, best_step=best_step)
        if writer is not None:
            writer.close()
        metrics_f.close()
        sync.finalize()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()


if __name__ == "__main__":
    main()
