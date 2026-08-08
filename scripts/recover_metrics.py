"""Reconstruct a run's metrics.jsonl from its stdout training log.

Examples:

    .venv/bin/python scripts/recover_metrics.py ~/sft_run.log \\
        checkpoints/sft/<run>
    .venv/bin/python scripts/recover_metrics.py <run>/train.log <run>  # if intact

Disaster-recovery path: every stage script tees full stdout to a log
(scripts pipe to ~/<stage>_run.log; the run dir's train.log holds the
same lines), and those printed lines carry the whole metric history.
This parses pretrain/sft step+val lines and grpo step+eval lines back
into metrics.jsonl events matching run_utils.emit's schema, then
rebuild_tb.py can regenerate the TensorBoard view:

    recover_metrics.py <log> <run-dir>   # writes <run-dir>/metrics.jsonl
    rebuild_tb.py <run-dir>              # writes <run-dir>/tb/

Existing metrics.jsonl is backed up to metrics.jsonl.bak first.
tokens_seen is reconstructed as step × batch × seq_len read from the
run.json manifest (exact for non-resumed runs; a resumed run's offset is
the resume step's count). Sample texts and wall-clock ETA/tok-per-sec
are recovered where present in the log.
"""

import argparse
import json
import re
import sys
from pathlib import Path

STEP_LM = re.compile(
    r"step\s+(\d+)/\d+\s+(?:train_loss|sft_loss)=([\d.]+)\s+bpb=([\d.]+)\s+"
    r"grad_norm=([\d.]+)\s+lr=([\d.eE+-]+)\s+\(\s*(\d+) tok/s\)"
)
VAL_LM = re.compile(r"val_loss=([\d.]+)\s+val_bpb=([\d.]+)(\s+← best)?")
STEP_RL = re.compile(
    r"step\s+(\d+)/\d+\s+reward=(-?[\d.]+)\s+\[(.*?)\]\s+len=([\d.]+)\s+"
    r"stop=(\d+)%\s+kl=([\d.]+)\s+grad=([\d.]+)"
)
EVAL_RL = re.compile(r"eval reward=(-?[\d.]+)\s+\[(.*?)\](\s+← best)?")
KIND = re.compile(r"(\w+)=(-?[\d.]+)")


def parse_kinds(blob: str) -> dict:
    return {k: float(v) for k, v in KIND.findall(blob)}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconstruct metrics.jsonl from a stdout training log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("log", type=Path, help="Stdout log file to parse")
    p.add_argument("run_dir", type=Path, help="Run directory to write into")
    args = p.parse_args()

    tokens_per_step = 0
    manifest = args.run_dir / "run.json"
    if manifest.exists():
        margs = json.loads(manifest.read_text()).get("args", {})
        tokens_per_step = int(margs.get("batch_size", 0)) * int(margs.get("seq_len", 0))

    events = []
    last_step = 0
    for line in args.log.read_text(errors="replace").splitlines():
        if m := STEP_LM.search(line):
            last_step = int(m.group(1))
            events.append({
                "event": "step", "step": last_step,
                "train_loss": float(m.group(2)), "bpb": float(m.group(3)),
                "grad_norm": float(m.group(4)), "lr": float(m.group(5)),
                "tok_per_sec": float(m.group(6)),
                "tokens_seen": last_step * tokens_per_step or None,
                "recovered": True,
            })
        elif m := STEP_RL.search(line):
            last_step = int(m.group(1))
            events.append({
                "event": "step", "step": last_step,
                "reward": float(m.group(2)), "reward_by_kind": parse_kinds(m.group(3)),
                "mean_len": float(m.group(4)), "stop_frac": int(m.group(5)) / 100,
                "kl": float(m.group(6)), "grad_norm": float(m.group(7)),
                "recovered": True,
            })
        elif m := EVAL_RL.search(line):
            events.append({
                "event": "eval", "step": last_step,
                "reward": float(m.group(1)), "reward_by_kind": parse_kinds(m.group(2)),
                "is_best": bool(m.group(3)), "recovered": True,
            })
        elif m := VAL_LM.search(line):
            events.append({
                "event": "eval", "step": last_step,
                "val_loss": float(m.group(1)), "val_bpb": float(m.group(2)),
                "tokens_seen": last_step * tokens_per_step or None,
                "is_best": bool(m.group(3)), "recovered": True,
            })

    if not events:
        raise SystemExit(f"no metric lines recognized in {args.log}")
    out = args.run_dir / "metrics.jsonl"
    if out.exists():
        out.replace(out.with_suffix(".jsonl.bak"))
    with open(out, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    steps = sum(1 for e in events if e["event"] == "step")
    evals = sum(1 for e in events if e["event"] == "eval")
    print(f"{out}: {steps} step events, {evals} eval events "
          f"(previous file backed up to metrics.jsonl.bak)")


if __name__ == "__main__":
    main()
