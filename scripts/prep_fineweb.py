"""Fetch a FineWeb-Edu text sample into data/fineweb_edu.txt.

    .venv/bin/python scripts/prep_fineweb.py            # ~2 GB (gen-4)
    .venv/bin/python scripts/prep_fineweb.py --mb 100

Reads the 10BT-sample parquet shards through HTTP range requests (a
seekable file-like over urllib) and pulls row groups until the byte
target is met. Column projection keeps the transferred bytes close to
the text actually kept.

Row groups are drawn in SEEDED-RANDOM order across ALL shards
(gen-4 fix): the shards are dump-CLUSTERED — each ~1000-doc row group
holds one CommonCrawl snapshot, a few dumps interleaved in rotation —
so gen-3's sequential prefix of shard 0 contained exactly 3 of ~95
snapshots (2013-20, 2017-26, 2020-05; nothing after Jan 2020). The
shuffle makes every dump reachable at any --mb; the dump histogram is
printed at the end as verification.

Modern, quality-filtered web prose (ODC-By): the register between
Gutenberg books and encyclopedic wikitext, and the biggest single
share of the gen-4 mix (configs/mix-gen4-chat.json).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hf_util import SSL_CTX, hf_list
from transformer.chat import sanitize

REPO = "HuggingFaceFW/fineweb-edu"
SUBDIR = "sample/10BT"
HF = "https://huggingface.co"


class HttpRangeFile:
    """Minimal seekable read-only file over HTTP range requests —
    exactly what pyarrow needs to read remote parquet piecemeal."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, context=SSL_CTX) as r:
            self.length = int(r.headers["Content-Length"])
        self.transferred = 0

    def size(self) -> int:
        return self.length

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:  # pyarrow probes the ATTRIBUTE, not a call
        return False

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.length + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.length - self.pos
        if n == 0:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, context=SSL_CTX) as r:
            blob = r.read()
        self.pos += len(blob)
        self.transferred += len(blob)
        return blob

    def close(self) -> None:
        pass


def main() -> None:
    import random
    from collections import Counter

    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(
        description="Fetch a FineWeb-Edu sample via HTTP range reads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mb", type=int, default=2000, help="Target size in MB")
    p.add_argument("--seed", type=int, default=0,
                   help="Row-group shuffle seed (same seed + same --mb "
                        "= same corpus)")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "fineweb_edu.txt")
    args = p.parse_args()

    shards = sorted(f["path"] for f in hf_list(REPO, SUBDIR)
                    if f["path"].endswith(".parquet"))
    print(f"{len(shards)} shards; indexing row groups...")
    files, index = [], []  # index: (shard_idx, row_group)
    for si, shard in enumerate(shards):
        src = HttpRangeFile(f"{HF}/datasets/{REPO}/resolve/main/{shard}")
        pf = pq.ParquetFile(src)
        files.append((src, pf))
        index.extend((si, g) for g in range(pf.num_row_groups))
        print(f"  shard {si}: {pf.num_row_groups} row groups, "
              f"{src.length / 1e9:.2f} GB")
    random.Random(args.seed).shuffle(index)
    print(f"{len(index)} row groups total, shuffled with seed {args.seed}")

    target = args.mb * 1_000_000
    written, n_read = 0, 0
    dumps: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for si, g in index:
            src, pf = files[si]
            table = pf.read_row_group(g, columns=["text", "dump"])
            dumps.update(table.column("dump").to_pylist())
            for text in table.column("text").to_pylist():
                text = sanitize(text).strip()
                if text:
                    out.write(text + "\n\n")
                    written += len(text) + 2
            n_read += 1
            if n_read % 20 == 0 or written >= target:
                total_tx = sum(f.transferred for f, _ in files)
                print(f"  {n_read} row groups: {written / 1e6:.1f} MB "
                      f"written, {total_tx / 1e6:.0f} MB transferred",
                      flush=True)
            if written >= target:
                break
    total_tx = sum(f.transferred for f, _ in files)
    print(f"wrote {args.out} — {written / 1e6:.1f} MB from {n_read} row "
          f"groups across {len({si for si, _ in index[:n_read]})} shards "
          f"({total_tx / 1e6:.0f} MB transferred)")
    print(f"dump coverage: {len(dumps)} distinct dumps")
    for dump, n in dumps.most_common():
        print(f"  {dump}: {n} docs")


if __name__ == "__main__":
    main()
