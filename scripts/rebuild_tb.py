"""Regenerate a run's tb/ event files from its metrics.jsonl.

Examples:

    .venv/bin/python scripts/rebuild_tb.py checkpoints/sft/<run>
    .venv/bin/python scripts/rebuild_tb.py checkpoints/*/*        # all runs

metrics.jsonl is the authoritative metric record; TensorBoard event
files are a derived view. This script deletes tb/ and replays every
step/eval/sample event through a fresh SummaryWriter, producing the same
scalars the live run would have written. Use it after any incident that
truncates or corrupts tb/ (see also recover_metrics.py when
metrics.jsonl itself is the casualty), or to backfill schema changes
into old runs' TensorBoard views.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch.utils.tensorboard import SummaryWriter

from run_utils import BEST_WINDOW, windowed_best_val

# eval-event dict fields → TB tag prefixes (per-key scalars).
_EVAL_DICTS = (("domain_bpb", "val_domain"), ("domain_acc", "val_domain_acc"),
               ("domain_top5", "val_domain_top5"),
               ("domain_ent", "val_domain_ent"), ("group_bpb", "val_group"),
               ("source_bpb", "val_source"))


def rebuild(run_dir: Path) -> int:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        print(f"{run_dir.name}: no metrics.jsonl, skipping")
        return 0
    tb_dir = run_dir / "tb"
    if tb_dir.exists():
        shutil.rmtree(tb_dir)
    writer = SummaryWriter(log_dir=str(tb_dir))
    n, best_loss = 0, float("inf")
    bpb_hist: list[float] = []
    best_windowed = float("inf")
    for line in metrics.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # NUL-torn lines from preemption kills — skip
        step, kind = ev.get("step"), ev.get("event")
        if step is None:
            continue
        if kind == "step":
            for key, tag in (("train_loss", "train/loss"), ("bpb", "train/bpb"),
                             ("entropy_bits", "train/entropy_bits"),
                             ("lr", "train/lr"), ("grad_norm", "train/grad_norm"),
                             ("tok_per_sec", "train/tok_per_sec"),
                             ("tokens_seen", "train/tokens_seen"),
                             ("mem_gb", "train/mem_gb"),
                             ("reward", "reward/mean"), ("mean_len", "rollout/mean_len"),
                             ("stop_frac", "rollout/stop_frac"),
                             ("dead_frac", "rollout/dead_frac"),
                             ("entropy", "rollout/entropy"),
                             ("kl", "train/kl"),
                             ("clip_frac", "train/clip_frac")):
                if ev.get(key) is not None:
                    writer.add_scalar(tag, ev[key], step)
            for k, v in (ev.get("reward_by_kind") or {}).items():
                writer.add_scalar(f"reward/{k}", v, step)
            for name, d in (ev.get("moe_layers") or {}).items():
                for stat, val in d.items():
                    writer.add_scalar(f"moe/{name}_{stat}", val, step)
        elif kind == "eval":
            if ev.get("val_loss") is not None:
                best_loss = min(best_loss, ev["val_loss"])
                writer.add_scalar("val/loss", ev["val_loss"], step)
            if ev.get("val_bpb") is not None:
                writer.add_scalar("val/bpb", ev["val_bpb"], step)
                # val/best mirrors the live run's windowed-bpb tracking;
                # pre-window runs (no val_bpb) fall back to min val_loss.
                bpb_hist.append(ev["val_bpb"])
                best_windowed = min(best_windowed,
                                    windowed_best_val(bpb_hist, BEST_WINDOW))
                writer.add_scalar("val/best", best_windowed, step)
            elif ev.get("val_loss") is not None:
                writer.add_scalar("val/best", best_loss, step)
            for key, tag in (("val_acc", "val/acc"), ("val_top5", "val/top5"),
                             ("val_ent", "val/ent"),
                             ("val_tasks_bpb", "val/tasks_bpb")):
                if ev.get(key) is not None:
                    writer.add_scalar(tag, ev[key], step)
            for key, prefix in _EVAL_DICTS:
                for name, v in (ev.get(key) or {}).items():
                    writer.add_scalar(f"{prefix}/{name}", v, step)
            if ev.get("param_norm") is not None:
                writer.add_scalar("params/global_norm", ev["param_norm"], step)
            if ev.get("reward") is not None:
                writer.add_scalar("eval/reward", ev["reward"], step)
        elif kind == "replay":
            if ev.get("replay_bpb") is not None:
                writer.add_scalar("train/replay_bpb", ev["replay_bpb"], step)
        elif kind == "probe":
            for tag, v in (ev.get("scalars") or {}).items():
                writer.add_scalar(tag, v, step)
        elif kind == "probe_dump" and ev.get("text"):
            writer.add_text(f"probe_dump/{ev.get('id', '?')}",
                            f"**{ev.get('prompt', '')}**\n\n```\n{ev['text']}\n```",
                            step)
        elif kind == "sample" and ev.get("text"):
            writer.add_text("samples", f"**{ev.get('prompt', '')}**\n\n{ev['text']}", step)
        else:
            continue
        n += 1
    writer.close()
    print(f"{run_dir.name}: rebuilt tb/ from {n} events")
    return n


def main() -> None:
    p = argparse.ArgumentParser(
        description="Regenerate tb/ event files from metrics.jsonl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("runs", type=Path, nargs="+", help="Run directories")
    args = p.parse_args()
    for run_dir in args.runs:
        if run_dir.is_dir():
            rebuild(run_dir)


if __name__ == "__main__":
    main()
