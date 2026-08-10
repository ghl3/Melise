"""Fetch a FineWeb-Edu text sample into data/fineweb_edu.txt.

    .venv/bin/python scripts/prep_fineweb.py            # ~100 MB
    .venv/bin/python scripts/prep_fineweb.py --mb 50

The 10BT-sample shards are ~2.15 GB parquet each — far more than
needed — so instead of downloading one, this reads the shard through
HTTP range requests (a seekable file-like over urllib) and pulls row
groups until the byte target is met. Column projection keeps the
transferred bytes close to the text actually kept.

Modern, quality-filtered web prose (ODC-By) for the gen-3 pretrain mix
(configs/mix-gen3-chat.json): the register between Gutenberg books and
encyclopedic wikitext that the gen-2 mix lacked entirely.
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
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(
        description="Fetch a FineWeb-Edu sample via HTTP range reads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mb", type=int, default=100, help="Target size in MB")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "fineweb_edu.txt")
    args = p.parse_args()

    shard = sorted(f["path"] for f in hf_list(REPO, SUBDIR)
                   if f["path"].endswith(".parquet"))[0]
    url = f"{HF}/datasets/{REPO}/resolve/main/{shard}"
    src = HttpRangeFile(url)
    pf = pq.ParquetFile(src)
    target = args.mb * 1_000_000
    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=["text"])
            for text in table.column("text").to_pylist():
                text = sanitize(text).strip()
                if text:
                    out.write(text + "\n\n")
                    written += len(text) + 2
            print(f"  row group {g}: {written / 1e6:.1f} MB written, "
                  f"{src.transferred / 1e6:.0f} MB transferred", flush=True)
            if written >= target:
                break
    print(f"wrote {args.out} — {written / 1e6:.1f} MB "
          f"({src.transferred / 1e6:.0f} MB transferred of a "
          f"{src.length / 1e9:.2f} GB shard)")


if __name__ == "__main__":
    main()
