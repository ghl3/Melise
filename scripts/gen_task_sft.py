"""Generate the cold-start SFT corpus from the RL reward tasks.

Example:

    .venv/bin/python scripts/gen_task_sft.py            # 25k examples
    .venv/bin/python scripts/gen_task_sft.py --n 50000 --seed 1

Writes data/chat_tasks.txt: single-turn conversations in the byte chat
template whose prompts come from the scripts/rewards.py generators and
whose responses are each task's canonical full-credit answer. Because
sft.py picks up every data/chat_*.txt, this joins the SFT mix
automatically.

Why this exists: GRPO only gets a gradient when a rollout group contains
*different* rewards. A policy that has never seen a task succeed scores
0 across the whole group and learns nothing (the cold-start problem).
A modest slice of worked examples in SFT gives every task a nonzero
success rate, which is all the RL stage needs to take over.

Sizing note: sft.py samples batches uniformly per conversation, so the
task share of SFT batches is set by example COUNT vs the chat corpora's
conversation count (~190k), not by bytes. The default 25k ≈ 12%.

Every example is self-checked: the task's own scorer must give its
canonical answer full credit, so the SFT target and the RL reward can
never disagree.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chat_format import encode_conversation
from rewards import TASKS

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the cold-start SFT corpus from reward tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=25000, help="Total examples")
    p.add_argument("--tasks", type=str, default=",".join(TASKS),
                   help="Comma-separated task kinds to include")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "chat_tasks.txt")
    args = p.parse_args()

    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rng = random.Random(args.seed)
    convs, kinds = [], Counter()
    for _ in range(args.n):
        task = TASKS[rng.choice(names)](rng)
        got = task.score(task.answer)
        assert got >= 0.99, (
            f"canonical answer scores {got:.2f} for {task.kind}: "
            f"{task.prompt!r} -> {task.answer!r}"
        )
        convs.append(encode_conversation(
            [("user", task.prompt), ("assistant", task.answer)]))
        kinds[task.kind] += 1

    blob = b"".join(convs)
    args.out.write_bytes(blob)
    print(f"wrote {args.out} — {len(convs):,} examples, {len(blob) / 1e6:.1f} MB")
    print("  per kind:", dict(sorted(kinds.items())))
    for conv in convs[:5]:
        print(f"  e.g. {conv!r}")


if __name__ == "__main__":
    main()
