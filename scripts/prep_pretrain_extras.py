"""Download code and math corpora for pretraining.

Example:

    .venv/bin/python scripts/prep_pretrain_extras.py            # both, ~100 MB each
    .venv/bin/python scripts/prep_pretrain_extras.py --max-mb 50

Produces (skipping any that already exist; --force rebuilds):

    data/code_python.txt   Python source from codeparrot/codeparrot-clean
                           (valid split: cleaned, deduplicated GitHub
                           Python; ungated, one 142 MB shard)
    data/math_openweb.txt  mathematical web text from
                           open-web-math/open-web-math (LaTeX-heavy CC
                           pages: proofs, worked problems, notation)

(bigcode/the-stack-smol was the first choice for code but is gated
behind BigCode terms — 401 without an HF token.)

Why: the corpus mix is literature + wikipedia — digits are 0.03–2.2% of
bytes and worked computation is essentially absent, which starves the
downstream arithmetic/counting reward tasks (see transformer/rl/tasks.py).
Code adds structure (indentation, brackets, identifiers); OpenWebMath
adds numerals, operators, and step-by-step derivations.

Documents are concatenated with blank-line separators — plain byte
corpora like everything else in data/. C0 control bytes are stripped
(except tab/newline) so the chat template's markers stay unambiguous
everywhere. The mixture config weights these via multipliers
(configs/mix-downweight-wiki.json); they only affect the NEXT pretrain
run — existing checkpoints are untouched.

Canonicalize into the bucket with scripts/add_dataset.py afterwards.
"""

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformer.chat import sanitize

from hf_util import fetch, hf_list, hf_resolve

CODE_REPO = "codeparrot/codeparrot-clean-valid"  # JSONL.gz, 'content' field
OWM_REPO = "open-web-math/open-web-math"
OWM_DIR = "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download code (Python) and math (OpenWebMath) corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-mb", type=int, default=100,
                   help="Cap each output corpus at roughly this many MB")
    p.add_argument("--force", action="store_true",
                   help="Rebuild outputs that already exist")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Where raw downloads land (default: a temp dir)")
    return p.parse_args()


def build_codeparrot(out: Path, cache: Path, max_bytes: int) -> None:
    """codeparrot-clean-valid is gzipped JSONL with a 'content' field."""
    import gzip
    import json

    shard = next(f["path"] for f in hf_list(CODE_REPO, "")
                 if f["path"].endswith(".json.gz"))
    raw = fetch(hf_resolve(CODE_REPO, shard), cache / Path(shard).name)
    docs, total = [], 0
    with gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            text = sanitize(json.loads(line).get("content") or "").strip()
            if not text:
                continue
            docs.append(text)
            total += len(text) + 2
            if total >= max_bytes:
                break
    out.write_bytes("\n\n".join(docs).encode("utf-8"))
    print(f"  wrote {out.name}: {len(docs):,} files, "
          f"{out.stat().st_size / 1e6:.1f} MB")


def build_from_parquet(repo: str, subdir: str, out: Path, cache: Path,
                       max_bytes: int, text_columns=("content", "text")) -> None:
    """Concatenate one parquet column across shards until max_bytes."""
    import pyarrow.parquet as pq

    shards = sorted(f["path"] for f in hf_list(repo, subdir)
                    if f["path"].endswith(".parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {repo}/{subdir}")
    docs, total = [], 0
    for shard in shards:
        raw = fetch(hf_resolve(repo, shard), cache / f"{repo.split('/')[0]}-{Path(shard).name}")
        schema_names = pq.read_schema(raw).names
        col = next(c for c in text_columns if c in schema_names)
        for text in pq.read_table(raw, columns=[col]).column(col).to_pylist():
            text = sanitize(text or "").strip()
            if not text:
                continue
            docs.append(text)
            total += len(text) + 2
            if total >= max_bytes:
                break
        if total >= max_bytes:
            break
    out.write_bytes("\n\n".join(docs).encode("utf-8"))
    print(f"  wrote {out.name}: {len(docs):,} documents, "
          f"{out.stat().st_size / 1e6:.1f} MB (from {shard})")


def main() -> None:
    args = parse_args()
    cache = args.cache_dir or Path(tempfile.gettempdir()) / "pretrain-extras-cache"
    data = PROJECT_ROOT / "data"
    targets = [
        (data / "code_python.txt",
         lambda o: build_codeparrot(o, cache, args.max_mb * 1_000_000)),
        (data / "math_openweb.txt",
         lambda o: build_from_parquet(OWM_REPO, OWM_DIR, o, cache,
                                      args.max_mb * 1_000_000)),
    ]
    for out, build in targets:
        if out.exists() and not args.force:
            print(f"{out.name}: exists, skipping (--force to rebuild)")
            continue
        print(f"{out.name}:")
        build(out)
    print("\ncanonicalize into the bucket with scripts/add_dataset.py when happy")


if __name__ == "__main__":
    main()
