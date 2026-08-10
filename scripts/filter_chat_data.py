"""Filter a chat corpus down to casual-chat material.

    .venv/bin/python scripts/filter_chat_data.py --dry-run   # stats only
    .venv/bin/python scripts/filter_chat_data.py             # rewrite in place

Built for chat_smoltalk.txt after a sampling audit (2026-08-11, 500
random conversations): ~9% is tool-calling (apigen — '<tools>' schemas
and '<tool_call>' JSON), ~15% is text-processing pipelines with fixed
preambles (summarize / rewrite / extract) whose long input payloads
aren't conversation, and the >6 KB tail (~p88+) fragments against the
2048-token context so the loss trains on truncated-context noise.
None of that serves the casual-chat use case the site targets. Short
code Q&A is kept on purpose — it's legitimate conversational variety.

Drops a conversation if ANY of:
- any turn contains a --drop-contains marker (tool-call formats)
- the first user turn starts with a --drop-prefix string (pipeline
  subsets with fixed preambles)
- the whole conversation exceeds --max-bytes

Rewrites the file (or --out) and prints per-rule counts. The result
still needs `add_dataset.py --seed-from-local` to reach the bucket.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load chat.py directly — the package __init__ imports torch, which a
# pure text filter neither needs nor should require.
_spec = importlib.util.spec_from_file_location(
    "chat", PROJECT_ROOT / "transformer" / "chat.py")
_chat = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _chat
_spec.loader.exec_module(_chat)
parse_turns, split_conversations = _chat.parse_turns, _chat.split_conversations

DROP_CONTAINS = ["<tools>", "<tool_call>"]
DROP_PREFIXES = [
    "You are an expert in composing functions.",
    "Provide a concise, objective summary of the input text",
    "You're an AI assistant for text re-writing.",
    "You are an AI rewriting assistant.",
    "Extract and present the main key point of the input",
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Filter a chat corpus down to casual-chat material.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="inp", type=Path,
                   default=PROJECT_ROOT / "data" / "chat_smoltalk.txt")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: rewrite --in)")
    p.add_argument("--max-bytes", type=int, default=6144,
                   help="Drop conversations larger than this")
    p.add_argument("--drop-contains", action="append", default=None,
                   help=f"Marker substrings (default: {DROP_CONTAINS})")
    p.add_argument("--drop-prefix", action="append", default=None,
                   help=f"First-turn prefixes (default: {len(DROP_PREFIXES)} "
                        "known pipeline preambles)")
    p.add_argument("--dry-run", action="store_true", help="Stats only")
    args = p.parse_args()

    contains = [s.encode() for s in (args.drop_contains or DROP_CONTAINS)]
    prefixes = [s.encode() for s in (args.drop_prefix or DROP_PREFIXES)]

    blob = args.inp.read_bytes()
    convs = split_conversations(blob)
    kept, dropped = [], {"tool-markers": 0, "pipeline-preamble": 0,
                         "oversized": 0}
    for conv in convs:
        if any(m in conv for m in contains):
            dropped["tool-markers"] += 1
            continue
        turns = parse_turns(conv)
        first = turns[0][1] if turns else b""
        if any(first.startswith(pre) for pre in prefixes):
            dropped["pipeline-preamble"] += 1
            continue
        if len(conv) > args.max_bytes:
            dropped["oversized"] += 1
            continue
        kept.append(conv)

    n, k = len(convs), len(kept)
    print(f"{args.inp}: {n:,} conversations, {len(blob) / 1e6:.1f} MB")
    for rule, count in dropped.items():
        print(f"  drop {rule:<18} {count:>7,}  ({count / n:.1%})")
    out_blob = b"".join(kept)
    print(f"  keep {'':<18} {k:>7,}  ({k / n:.1%})  -> {len(out_blob) / 1e6:.1f} MB")

    if args.dry_run:
        return
    out = args.out or args.inp
    out.write_bytes(out_blob)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
