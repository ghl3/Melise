"""Sample examples from the training datasets (see docs/DATASETS.md).

    .venv/bin/python scripts/sample_data.py pretrain              # random dataset, weight-proportional
    .venv/bin/python scripts/sample_data.py pretrain --name enwik8 -n 3
    .venv/bin/python scripts/sample_data.py sft --name chat-tasks
    .venv/bin/python scripts/sample_data.py rl -n 5 --no-meta     # random task family

Stage semantics mirror training:
- pretrain  datasets weighted by bytes × mix multiplier
            (configs/mix-downweight-wiki.json); excerpts come from the
            train slice only, snapped to line boundaries.
- sft       datasets weighted by conversation count (sft.py samples
            uniformly per conversation); examples are whole
            conversations, printed per turn.
- rl        task families, uniform (sample_tasks behavior); examples
            are freshly generated prompts with canonical answers.

Canonical dataset names are the bucket-manifest names (chat-tasks,
war-and-peace, …) for pretrain/sft and family names (copy, arith,
parity, count, words, recall) for rl. The manifest is fetched once and
cached in data/.manifest-cache.json so the tool works offline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import mmap
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _load_module(rel_path: str):
    """Import a repo module by file path, bypassing transformer/__init__
    (which imports the model stack and torch — this tool needs neither,
    and should run under plain python3)."""
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look modules up by name
    spec.loader.exec_module(mod)
    return mod


_chat = _load_module("transformer/chat.py")
parse_turns, split_conversations = _chat.parse_turns, _chat.split_conversations
TASKS = _load_module("transformer/rl/tasks.py").TASKS

MIX = PROJECT_ROOT / "configs" / "mix-downweight-wiki.json"
CACHE = PROJECT_ROOT / "data" / ".manifest-cache.json"


def manifest_names() -> dict[str, dict]:
    """filename -> {name, source_url, kind}; cached for offline use."""
    try:
        from gcs_util import read_manifest
        entries = read_manifest()["datasets"]
        CACHE.write_text(json.dumps(entries))
    except Exception:
        entries = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return {e["filename"]: {"name": name, **e} for name, e in entries.items()}


# ---------- stage catalogs: name -> {file, weight, meta...} ----------

def pretrain_catalog(by_file: dict) -> dict[str, dict]:
    mix = json.loads(MIX.read_text())
    exclude = set(PROJECT_ROOT.glob(mix.get("exclude", "\0")))
    mults = {PROJECT_ROOT / k: v for k, v in mix.get("multipliers", {}).items()}
    splits = {PROJECT_ROOT / k: v for k, v in mix.get("splits", {}).items()}
    out = {}
    for f in sorted(PROJECT_ROOT.glob(mix["include"])):
        if f in exclude or not f.is_file():
            continue
        info = by_file.get(f.name, {})
        name = info.get("name", f.stem.replace("_", "-"))
        mult = mults.get(f, 1.0)
        train_frac = splits.get(f, {}).get("train", 1.0)
        size = f.stat().st_size
        out[name] = {"file": f, "bytes": size, "mult": mult,
                     "train_frac": train_frac,
                     "weight": size * mult * train_frac,
                     "source": info.get("source_url", "?")}
    return out


def sft_catalog(by_file: dict) -> dict[str, dict]:
    out = {}
    for f in sorted(PROJECT_ROOT.glob("data/chat_*.txt")):
        info = by_file.get(f.name, {})
        name = info.get("name", f.stem.replace("_", "-"))
        with open(f, "rb") as fh, mmap.mmap(fh.fileno(), 0,
                                            access=mmap.ACCESS_READ) as mm:
            convs = 0
            pos = mm.find(b"\x04")
            while pos != -1:
                convs += 1
                pos = mm.find(b"\x04", pos + 1)
        out[name] = {"file": f, "bytes": f.stat().st_size, "convs": convs,
                     "weight": convs, "source": info.get("source_url", "?")}
    return out


def rl_catalog() -> dict[str, dict]:
    return {name: {"weight": 1.0, "gen": gen,
                   "doc": (gen.__doc__ or "").strip().split("\n")[0]}
            for name, gen in TASKS.items()}


# ---------- example extraction ----------

def pretrain_example(entry: dict, rng: random.Random, chars: int) -> str:
    limit = int(entry["bytes"] * entry["train_frac"])
    window = chars * 4  # utf-8 headroom
    start = rng.randrange(max(limit - window, 1))
    with open(entry["file"], "rb") as fh:
        fh.seek(start)
        blob = fh.read(min(window, limit - start))
    # Snap to whole lines when possible, then trim to the char budget.
    head = blob.find(b"\n")
    if 0 <= head < len(blob) - 1:
        blob = blob[head + 1:]
    text = blob.decode("utf-8", errors="ignore")[:chars]
    return text[: text.rfind("\n")] if "\n" in text else text


def sft_example(entry: dict, rng: random.Random) -> str:
    with open(entry["file"], "rb") as fh, mmap.mmap(fh.fileno(), 0,
                                                    access=mmap.ACCESS_READ) as mm:
        # Uniform over conversations: walk separators to the k-th one.
        k = rng.randrange(entry["convs"])
        start = 0
        for _ in range(k):
            start = mm.find(b"\x04", start) + 1
        end = mm.find(b"\x04", start)
        conv = mm[start:end + 1]
    lines = [f"{role:>9}: {content.decode('utf-8', errors='replace')}"
             for role, content in parse_turns(split_conversations(conv)[0])]
    return "\n".join(lines)


def rl_example(entry: dict, rng: random.Random) -> str:
    task = entry["gen"](rng)
    return f"   prompt: {task.prompt}\ncanonical: {task.answer}"


# ---------- CLI ----------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample examples from the training datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("stage", choices=("pretrain", "sft", "rl"))
    p.add_argument("--name", default=None,
                   help="Canonical dataset/family name; random (weight-"
                        "proportional) if omitted")
    p.add_argument("-n", "--n", type=int, default=1, help="Examples to print")
    p.add_argument("--no-meta", action="store_true",
                   help="Skip the dataset metadata header")
    p.add_argument("--chars", type=int, default=600,
                   help="Excerpt length for pretrain examples")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for reproducible sampling")
    args = p.parse_args()

    rng = random.Random(args.seed)
    if args.stage == "rl":
        catalog = rl_catalog()
    else:
        by_file = manifest_names()
        catalog = (pretrain_catalog if args.stage == "pretrain"
                   else sft_catalog)(by_file)
    if not catalog:
        raise SystemExit(f"no {args.stage} datasets found — is data/ populated?")

    if args.name is not None:
        if args.name not in catalog:
            raise SystemExit(f"unknown {args.stage} dataset {args.name!r}. "
                             f"Available: {', '.join(sorted(catalog))}")
        name = args.name
    else:
        names = sorted(catalog)
        weights = [catalog[k]["weight"] for k in names]
        name = rng.choices(names, weights=weights)[0]

    entry = catalog[name]
    total = sum(e["weight"] for e in catalog.values())
    if not args.no_meta:
        print(f"dataset: {name}  (stage: {args.stage}, "
              f"sampling share {entry['weight'] / total:.1%})")
        if args.stage == "pretrain":
            print(f"   file: {entry['file'].relative_to(PROJECT_ROOT)}  "
                  f"{entry['bytes']:,} B  mix multiplier ×{entry['mult']:g}"
                  + (f"  train slice {entry['train_frac']:.0%}"
                     if entry["train_frac"] < 1 else ""))
            print(f" source: {entry['source']}")
        elif args.stage == "sft":
            print(f"   file: {entry['file'].relative_to(PROJECT_ROOT)}  "
                  f"{entry['bytes']:,} B  {entry['convs']:,} conversations")
            print(f" source: {entry['source']}")
        else:
            print(f" family: {entry['doc']}")
        print()

    make = {"pretrain": lambda: pretrain_example(entry, rng, args.chars),
            "sft": lambda: sft_example(entry, rng),
            "rl": lambda: rl_example(entry, rng)}[args.stage]
    for i in range(args.n):
        if args.n > 1:
            print(f"--- example {i + 1}/{args.n} ---")
        print(make())
        if i < args.n - 1:
            print()


if __name__ == "__main__":
    main()
