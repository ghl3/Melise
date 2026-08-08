"""GRPO on verifiable rewards (stage 2 of post-training, after sft.py).

Examples:

    Train an SFT checkpoint against all reward tasks:
        .venv/bin/python scripts/grpo.py \\
            --init checkpoints/sft-kimi3-small-17M-.../best.pt --steps 200

    A subset of tasks, bigger groups:
        .venv/bin/python scripts/grpo.py --init ... --tasks copy,parity --group-size 16

Group Relative Policy Optimization (DeepSeekMath, arXiv:2402.03300; the
DeepSeek-R1 algorithm). Per step, for each of --prompts-per-step task
prompts:

  1. Sample a group of --group-size completions from the current policy
     (batched KV-cache decode; the whole group decodes in one batch).
  2. Score each completion with the task's deterministic reward
     (scripts/rewards.py — no reward model anywhere).
  3. Advantage = the completion's reward z-scored within its group.
     The group baseline replaces PPO's learned value network — that's
     the whole trick, and why this fits in small memory.
  4. Update with the PPO clipped surrogate on per-token log-prob ratios,
     plus a KL penalty to the frozen --init reference (the k3 estimator,
     exp(ref−cur) − (ref−cur) − 1), which keeps the policy from
     collapsing into a reward-hacking degenerate.

Rollouts sample at --temperature from raw logits (log-probs are always
computed from untempered logits; temperatures ≠ 1 are exploration, and
mildly off-policy). Completions that never emit the end-of-turn byte
within --max-new get --no-stop-penalty subtracted — stopping is part of
the task.

Evals run greedy decoding on a fixed task set (seeded independently of
training), report mean reward overall and per task kind, and drive
best.pt. NOTE: unlike train.py/sft.py, best_val stored in checkpoints is
the eval mean *reward* — higher is better.

Checkpoints keep the train.py payload, so sample.py works on them; run
dirs follow the same layout (run.json, metrics.jsonl, tb/, bucket sync)
with names prefixed "grpo-". Resume with --resume (the frozen reference
is rebuilt from the init path recorded in run.json).
"""

import argparse
import json
import math
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
import torch.nn.functional as F

from transformer import build_model

from chat_format import END_TURN, completion_text, make_prompt
from rewards import TASKS, sample_tasks
from train import (
    BucketSync,
    Tee,
    emit,
    fmt_eta,
    generate_run_name,
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
                   help="Comma-separated task kinds (see scripts/rewards.py)")
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


# ---------- Rollout ----------


@torch.no_grad()
def rollout_group(model, prompt: bytes, n: int, max_new: int, temperature: float,
                  device, greedy: bool = False):
    """Decode `n` completions of one prompt in a single batch.

    Returns (completions, old_lp, lengths):
      completions  list of n token-id lists, each cut at its END_TURN
                   (inclusive) or max_new
      old_lp       (n, T) log-probs of the sampled tokens under the
                   policy that sampled them (raw logits, untempered)
      lengths      (n,) completion lengths
    """
    max_new = min(max_new, model.cfg.max_seq_len - len(prompt))
    ids = torch.tensor([list(prompt)] * n, device=device, dtype=torch.long)
    cache = model.new_cache(n, device)
    logits = model(ids, kv_cache=cache)[:, -1]  # (n, V)

    toks, lps = [], []
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(max_new):
        lp_all = F.log_softmax(logits, dim=-1)
        if greedy:
            tok = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1)
        toks.append(tok.squeeze(1))
        lps.append(lp_all.gather(1, tok).squeeze(1))
        done |= tok.squeeze(1) == END_TURN
        if bool(done.all()):
            break
        logits = model(tok, kv_cache=cache)[:, -1]

    toks = torch.stack(toks, dim=1)  # (n, T)
    lps = torch.stack(lps, dim=1)
    completions, lengths = [], []
    for row in toks.tolist():
        cut = row.index(END_TURN) + 1 if END_TURN in row else len(row)
        completions.append(row[:cut])
        lengths.append(cut)
    return completions, lps, lengths


def gather_completion_logprobs(model, ids, pos, tok):
    """Log-probs the model assigns to completion tokens.

    ids (N, L) padded full sequences; pos (N, T) index of the logit that
    predicts each completion token; tok (N, T) the completion tokens.
    """
    logits = model(ids)  # (N, L, V)
    at = logits.gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
    return F.log_softmax(at, dim=-1).gather(2, tok.unsqueeze(-1)).squeeze(-1)


# ---------- Eval ----------


def eval_rewards(model, task_names, n_prompts, seed, max_new, device):
    """Greedy decode a fixed task set; mean reward overall and per kind."""
    model.eval()
    try:
        tasks = sample_tasks(task_names, n_prompts, random.Random(f"{seed}-eval"))
        by_kind = defaultdict(list)
        for task in tasks:
            comps, _, _ = rollout_group(
                model, make_prompt(task.prompt), 1, max_new, 1.0, device, greedy=True
            )
            text = completion_text(bytes(comps[0]))
            by_kind[task.kind].append(task.score(text))
        all_scores = [s for v in by_kind.values() for s in v]
        return (sum(all_scores) / len(all_scores),
                {k: sum(v) / len(v) for k, v in sorted(by_kind.items())})
    finally:
        model.train()


# ---------- Main ----------


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
        name = args.run_name or f"grpo-{generate_run_name(preset, model.num_parameters())}"
        out_dir = (PROJECT_ROOT / "checkpoints" / name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "kind": "grpo",
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
    sync = BucketSync(out_dir, enabled=not args.no_bucket_sync)
    print(f"bucket sync: {sync.dest or 'off'}\n")

    emit(metrics_f, event="start", step=start_step, kind="grpo", preset=preset,
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
                        best_step=best_step, tokens_seen=tokens_seen, device=device)

    model.train()
    t_start = time.perf_counter()
    n_done, last_step_done = 0, start_step

    try:
        for step in range(start_step + 1, args.steps + 1):
            if interrupted["flag"]:
                break

            # ----- Rollout phase -----
            task_rng = random.Random(f"{args.seed}-{step}")
            tasks = sample_tasks(task_names, args.prompts_per_step, task_rng)
            seqs = []      # (prompt_bytes, completion_tokens, old_lp_row, reward)
            rewards_pg = torch.zeros(len(tasks), args.group_size)
            kind_scores = defaultdict(list)
            for pi, task in enumerate(tasks):
                prompt = make_prompt(task.prompt)
                comps, lps, lengths = rollout_group(
                    model, prompt, args.group_size, args.max_new,
                    args.temperature, device)
                for gi, comp in enumerate(comps):
                    text = completion_text(bytes(comp))
                    r = task.score(text)
                    if comp[-1] != END_TURN:
                        r -= args.no_stop_penalty
                    rewards_pg[pi, gi] = r
                    kind_scores[task.kind].append(r)
                    seqs.append((prompt, comp, lps[gi, : len(comp)].cpu()))
                tokens_seen += args.group_size * (len(prompt) + max(lengths))

            # Group-relative advantages (the G in GRPO).
            adv = (rewards_pg - rewards_pg.mean(dim=1, keepdim=True)) / (
                rewards_pg.std(dim=1, keepdim=True) + 1e-4
            )
            adv = adv.reshape(-1).to(device)

            # ----- Tensorize padded batch -----
            N = len(seqs)
            pls = [len(p) for p, _, _ in seqs]
            cls = [len(c) for _, c, _ in seqs]
            Lmax, Tmax = max(p + c for p, c in zip(pls, cls)), max(cls)
            ids = torch.zeros(N, Lmax, dtype=torch.long)
            tok = torch.zeros(N, Tmax, dtype=torch.long)
            pos = torch.zeros(N, Tmax, dtype=torch.long)
            mask = torch.zeros(N, Tmax, dtype=torch.bool)
            old_lp = torch.zeros(N, Tmax)
            for i, (p, c, lp) in enumerate(seqs):
                pl, cl = pls[i], cls[i]
                ids[i, :pl] = torch.tensor(list(p))
                ids[i, pl : pl + cl] = torch.tensor(c)
                tok[i, :cl] = torch.tensor(c)
                pos[i, :cl] = torch.arange(pl - 1, pl - 1 + cl)
                mask[i, :cl] = True
                old_lp[i, :cl] = lp
            ids, tok, pos = ids.to(device), tok.to(device), pos.to(device)
            mask, old_lp = mask.to(device), old_lp.to(device)
            total_toks = int(mask.sum().item())

            # ----- Update phase -----
            stats = defaultdict(float)
            for _ in range(args.inner_epochs):
                optimizer.zero_grad(set_to_none=True)
                for lo in range(0, N, args.update_microbatch):
                    hi = min(lo + args.update_microbatch, N)
                    m = mask[lo:hi]
                    new_lp = gather_completion_logprobs(
                        model, ids[lo:hi], pos[lo:hi], tok[lo:hi])
                    with torch.no_grad():
                        ref_lp = gather_completion_logprobs(
                            reference, ids[lo:hi], pos[lo:hi], tok[lo:hi])
                    ratio = torch.exp(new_lp - old_lp[lo:hi])
                    a = adv[lo:hi].unsqueeze(1)
                    surr = torch.minimum(
                        ratio * a,
                        ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * a,
                    )
                    d = ref_lp - new_lp
                    kl = torch.exp(d) - d - 1  # k3 estimator, always ≥ 0
                    loss = -((surr - args.kl_coef * kl) * m).sum() / total_toks
                    loss.backward()
                    with torch.no_grad():
                        stats["kl"] += float((kl * m).sum().item())
                        stats["clip"] += float(
                            ((ratio - 1).abs() > args.clip_eps)[m].sum().item())
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
                stopped = sum(1 for _, c, _ in seqs if c[-1] == END_TURN) / N
                mean_len = sum(cls) / N
                kl_per_tok = stats["kl"] / (total_toks * args.inner_epochs)
                kinds = {k: sum(v) / len(v) for k, v in sorted(kind_scores.items())}
                kinds_str = "  ".join(f"{k}={v:.2f}" for k, v in kinds.items())
                print(f"step {step:>4}/{args.steps}  reward={mean_r:.3f}  "
                      f"[{kinds_str}]  len={mean_len:.0f}  stop={stopped:.0%}  "
                      f"kl={kl_per_tok:.4f}  grad={float(grad_norm.item()):.2f}  "
                      f"ETA {fmt_eta(eta)}")
                ex_task, ex = tasks[0], seqs[0]
                print(f"      e.g. {ex_task.prompt!r} -> "
                      f"{completion_text(bytes(ex[1]))!r}  r={float(rewards_pg[0, 0]):.2f}")
                emit(metrics_f, event="step", step=step, reward=mean_r,
                     reward_by_kind=kinds, mean_len=mean_len, stop_frac=stopped,
                     kl=kl_per_tok, grad_norm=float(grad_norm.item()),
                     clip_frac=stats["clip"] / (total_toks * args.inner_epochs),
                     tokens_seen=tokens_seen)
                if writer is not None:
                    writer.add_scalar("reward/mean", mean_r, step)
                    for k, v in kinds.items():
                        writer.add_scalar(f"reward/{k}", v, step)
                    writer.add_scalar("rollout/mean_len", mean_len, step)
                    writer.add_scalar("rollout/stop_frac", stopped, step)
                    writer.add_scalar("train/kl", kl_per_tok, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm.item()), step)

            if step % args.eval_every == 0:
                mean_r, kinds = eval_rewards(model, task_names, args.eval_prompts,
                                             args.seed, args.max_new, device)
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
