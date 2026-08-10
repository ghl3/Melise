"""Download open chat datasets and convert them to byte-level corpora.

Examples:

    Everything (Dolly-15k + OASST1 + 2 SmolTalk shards):
        .venv/bin/python scripts/prep_chat_data.py

    More SmolTalk:
        .venv/bin/python scripts/prep_chat_data.py --smoltalk-shards 4

Produces (skipping any that already exist; --force rebuilds):

    data/chat_dolly.txt      databricks-dolly-15k (CC-BY-SA-3.0) —
                             15k human-written instruction/response pairs
    data/chat_oasst1.txt     OpenAssistant OASST1 (Apache-2.0) — human
                             multi-turn chat; English trees, best-ranked
                             path through each tree
    data/chat_smoltalk.txt   SmolTalk (Apache-2.0) — HuggingFace's
                             synthetic chat mix built for small models

Each file is conversations in the byte chat template (see
chat_format.py): control-byte role markers, END_CONV between
conversations. Files are plain bytes and flow through the existing data
pipeline (scripts/add_dataset.py to canonicalize into the bucket).

Raw downloads are cached in a temp dir; only the converted .txt lands in
data/. No HuggingFace libraries needed — files come over plain HTTPS
(SmolTalk parquet shards are read with pyarrow).
"""

import argparse
import gzip
import json
import sys
import tempfile
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformer.chat import END_CONV, encode_conversation, split_conversations

from hf_util import HF, fetch, hf_list

DOLLY_URL = f"{HF}/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
OASST1_REPO = "OpenAssistant/oasst1"
OASST1_FILE = "2023-04-12_oasst_ready.trees.jsonl.gz"
SMOLTALK_REPO = "HuggingFaceTB/smoltalk"
SMOLTALK_DIR = "data/all"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download open chat datasets and convert to byte corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smoltalk-shards", type=int, default=2,
                   help="Number of SmolTalk parquet shards to pull")
    p.add_argument("--force", action="store_true",
                   help="Rebuild outputs that already exist")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where raw downloads land (default: a temp dir)")
    return p.parse_args()


def write_corpus(path: Path, conversations: list[bytes]) -> None:
    blob = b"".join(conversations)
    path.write_bytes(blob)
    n_bytes = len(blob)
    print(f"  wrote {path.name}: {len(conversations):,} conversations, "
          f"{n_bytes / 1e6:.1f} MB")
    # Round-trip sanity: the corpus must split back into the same count.
    assert len(split_conversations(blob)) == len(conversations)


# ---------- Dolly ----------


def build_dolly(out: Path, cache: Path) -> None:
    raw = fetch(DOLLY_URL, cache / "dolly.jsonl")
    convs = []
    for line in raw.read_text().splitlines():
        rec = json.loads(line)
        user = rec["instruction"].strip()
        if rec.get("context", "").strip():
            user += "\n\n" + rec["context"].strip()
        resp = rec["response"].strip()
        if user and resp:
            convs.append(encode_conversation([("user", user), ("assistant", resp)]))
    write_corpus(out, convs)


# ---------- OASST1 ----------


def best_path(node: dict) -> list[tuple[str, str]]:
    """Walk a message tree following the best-ranked reply at each level."""
    turns = []
    while node is not None:
        role = "user" if node.get("role") == "prompter" else "assistant"
        text = (node.get("text") or "").strip()
        if not text or node.get("deleted"):
            break
        turns.append((role, text))
        replies = [r for r in node.get("replies", []) if not r.get("deleted")]
        if not replies:
            break
        node = sorted(replies, key=lambda r: (r.get("rank") is None, r.get("rank") or 0))[0]
    # A conversation must end on an assistant turn to be a training example.
    while turns and turns[-1][0] != "assistant":
        turns.pop()
    return turns


def all_paths(root):
    """Every root-to-leaf path through a message tree — all ranked
    replies, not just the best one (added 2026-08-11 to grow the
    real-human-conversation share of SFT ~4x). Shared prefixes appear in
    multiple paths; that mild duplication is accepted."""
    stack = [(root, [])]
    while stack:
        node, turns = stack.pop()
        role = "user" if node.get("role") == "prompter" else "assistant"
        text = (node.get("text") or "").strip()
        replies = []
        if text and not node.get("deleted"):
            turns = turns + [(role, text)]
            replies = [r for r in node.get("replies", []) if not r.get("deleted")]
        if replies:
            stack.extend((r, turns) for r in replies)
        else:
            while turns and turns[-1][0] != "assistant":
                turns = turns[:-1]
            if len(turns) >= 2:
                yield turns


def build_oasst1(out: Path, cache: Path) -> None:
    try:
        raw = fetch(f"{HF}/datasets/{OASST1_REPO}/resolve/main/{OASST1_FILE}",
                    cache / OASST1_FILE)
    except urllib.error.HTTPError:
        # Filename drifted — find the ready-trees export in the repo listing.
        files = [f["path"] for f in hf_list(OASST1_REPO, "")
                 if f["path"].endswith("trees.jsonl.gz")]
        if not files:
            raise SystemExit("no *trees.jsonl.gz found in the oasst1 repo")
        raw = fetch(f"{HF}/datasets/{OASST1_REPO}/resolve/main/{files[0]}",
                    cache / Path(files[0]).name)
    convs = []
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            root = tree.get("prompt") or tree
            if not str(root.get("lang", "")).startswith("en"):
                continue
            for turns in all_paths(root):
                conv = encode_conversation(turns)
                if len(conv) <= 6144:  # match the casual-chat size cap
                    convs.append(conv)
    write_corpus(out, convs)


# ---------- SmolTalk ----------


def build_smoltalk(out: Path, cache: Path, n_shards: int) -> None:
    import pyarrow.parquet as pq

    shards = sorted(f["path"] for f in hf_list(SMOLTALK_REPO, SMOLTALK_DIR)
                    if f["path"].endswith(".parquet")
                    and Path(f["path"]).name.startswith("train-"))
    if not shards:
        raise SystemExit(f"no parquet shards under {SMOLTALK_REPO}/{SMOLTALK_DIR}")
    convs = []
    for shard in shards[:n_shards]:
        raw = fetch(f"{HF}/datasets/{SMOLTALK_REPO}/resolve/main/{shard}",
                    cache / Path(shard).name)
        table = pq.read_table(raw, columns=["messages"])
        for messages in table.column("messages").to_pylist():
            turns = []
            system = ""
            for m in messages:
                role, text = m.get("role"), (m.get("content") or "").strip()
                if not text:
                    continue
                if role == "system":
                    # No system slot in the template — fold into the next
                    # user turn so the instruction still reaches the model.
                    system = text
                elif role == "user":
                    turns.append(("user", f"{system}\n\n{text}" if system else text))
                    system = ""
                elif role == "assistant":
                    turns.append(("assistant", text))
            while turns and turns[-1][0] != "assistant":
                turns.pop()
            if len(turns) >= 2:
                convs.append(encode_conversation(turns))
    write_corpus(out, convs)


def main() -> None:
    args = parse_args()
    cache = args.cache_dir or Path(tempfile.gettempdir()) / "chat-data-cache"
    data = PROJECT_ROOT / "data"
    targets = [
        (data / "chat_dolly.txt", lambda o: build_dolly(o, cache)),
        (data / "chat_oasst1.txt", lambda o: build_oasst1(o, cache)),
        (data / "chat_smoltalk.txt",
         lambda o: build_smoltalk(o, cache, args.smoltalk_shards)),
    ]
    for out, build in targets:
        if out.exists() and not args.force:
            print(f"{out.name}: exists, skipping (--force to rebuild)")
            continue
        print(f"{out.name}:")
        build(out)
    total = sum(t.stat().st_size for t, _ in targets if t.exists())
    print(f"\ntotal chat corpus: {total / 1e6:.1f} MB")
    print("canonicalize into the bucket with scripts/add_dataset.py when happy")


if __name__ == "__main__":
    main()
