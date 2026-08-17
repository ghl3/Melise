"""Run the shared probe suite (transformer/probes.py) against one
checkpoint, offline.

    .venv/bin/python scripts/probe_checkpoint.py <ckpt> [--quick]
    .venv/bin/python scripts/probe_checkpoint.py <ckpt> --families facts,identity

Stage form is inferred from the checkpoint path (a pretrain checkpoint
gets raw continuations; sft/rlvr get the chat template) — override
with --chat/--raw. Prints scalars and dump generations; --out appends
a probes.jsonl record next to the checkpoint.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from transformer.models import build_model
from transformer.probes import ProbeRunner
from transformer.tokenizer import load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--device", default="mps")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--families", default="facts,identity,date,verbatim")
    p.add_argument("--quick", action="store_true",
                   help="Small counts per family (smoke test)")
    p.add_argument("--no-dumps", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", action="store_true",
                   help="Append results to probes.jsonl next to the checkpoint")
    args = p.parse_args()

    chat = "pretrain" not in str(args.checkpoint)
    if args.chat:
        chat = True
    if args.raw:
        chat = False

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_tokenizer(getattr(cfg, "tokenizer", "bytes"))
    print(f"{args.checkpoint} — {'chat' if chat else 'raw'} forms, "
          f"tokenizer={tok.name}")

    kwargs = dict(chat=chat, seed=args.seed)
    if args.quick:
        kwargs.update(facts_per_family=3, names_per_stratum=3,
                      verbatim_per_file=1)
    runner = ProbeRunner(model, tok, args.device, **kwargs)

    with torch.no_grad():
        scalars = runner.run(tuple(args.families.split(",")))
        dumps = [] if args.no_dumps else runner.probe_dumps()

    for k in sorted(scalars):
        print(f"  {k:40s} {scalars[k]:.3f}")
    for d in dumps:
        print(f"\n--- dump:{d['id']} ({d['prompt'][:50]!r})\n{d['text']}")

    if args.out:
        rec = {"checkpoint": str(args.checkpoint), "chat": chat,
               "seed": args.seed, "scalars": scalars, "dumps": dumps,
               "time": datetime.now().isoformat(timespec="seconds")}
        out = args.checkpoint.parent / "probes.jsonl"
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"\nappended -> {out}")


if __name__ == "__main__":
    main()
