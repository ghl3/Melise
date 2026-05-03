"""Download text corpora for training.

Available datasets:
    shakespeare        TinyShakespeare (Karpathy's small dump)        ~1 MB
    alice              Alice in Wonderland                           ~150 KB
    frankenstein       Mary Shelley, Frankenstein                    ~440 KB
    pride-prejudice    Jane Austen, Pride and Prejudice              ~700 KB
    sherlock           Adventures of Sherlock Holmes                 ~600 KB
    moby-dick          Herman Melville, Moby Dick                    ~1.2 MB
    bible-kjv          King James Bible                              ~4.5 MB
    shakespeare-all    Complete Works of Shakespeare (Gutenberg)     ~5.5 MB
    wikitext2          WikiText-2 (Salesforce, train split, raw)     ~11 MB
    webster            Webster's Unabridged Dictionary (1913)        ~29 MB

Examples:
    .venv/bin/python scripts/download_data.py
    .venv/bin/python scripts/download_data.py --dataset alice
    .venv/bin/python scripts/download_data.py --dataset alice --dataset frankenstein
    .venv/bin/python scripts/download_data.py --all
    .venv/bin/python scripts/download_data.py --list

Project Gutenberg files come with ~10–30 KB of legal/copyright preamble and
trailer. We strip these automatically by looking for the standard markers,
so the downloaded `.txt` is just the body of the book.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# (url, output filename, kind) where kind ∈ {"raw", "gutenberg"}
DATASETS: dict[str, tuple[str, str, str]] = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "tinyshakespeare.txt",
        "raw",
    ),
    "alice": (
        "https://www.gutenberg.org/files/11/11-0.txt",
        "alice.txt",
        "gutenberg",
    ),
    "frankenstein": (
        "https://www.gutenberg.org/files/84/84-0.txt",
        "frankenstein.txt",
        "gutenberg",
    ),
    "pride-prejudice": (
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "pride_prejudice.txt",
        "gutenberg",
    ),
    "sherlock": (
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        "sherlock.txt",
        "gutenberg",
    ),
    "moby-dick": (
        "https://www.gutenberg.org/files/2701/2701-0.txt",
        "moby_dick.txt",
        "gutenberg",
    ),
    "bible-kjv": (
        "https://www.gutenberg.org/files/10/10-0.txt",
        "bible_kjv.txt",
        "gutenberg",
    ),
    "shakespeare-all": (
        "https://www.gutenberg.org/files/100/100-0.txt",
        "shakespeare_complete.txt",
        "gutenberg",
    ),
    "wikitext2": (
        # Salesforce's original S3 bucket is gone; this mirror in pytorch/examples
        # (the tokenized — not raw — version, with rare words replaced by `<unk>`).
        # For byte-level training the difference is minor.
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "wikitext2.txt",
        "raw",
    ),
    "webster": (
        "https://www.gutenberg.org/cache/epub/29765/pg29765.txt",
        "webster_dictionary.txt",
        "gutenberg",
    ),
}


GUTENBERG_START_MARKERS = [
    "*** START OF THIS PROJECT GUTENBERG",
    "*** START OF THE PROJECT GUTENBERG",
]
GUTENBERG_END_MARKERS = [
    "*** END OF THIS PROJECT GUTENBERG",
    "*** END OF THE PROJECT GUTENBERG",
]


def strip_gutenberg(text: str) -> str:
    """Trim PG header (legal/copyright) and footer (license) if present."""
    for sm in GUTENBERG_START_MARKERS:
        idx = text.find(sm)
        if idx != -1:
            nl = text.find("\n", idx)
            if nl != -1:
                text = text[nl + 1 :]
            break
    for em in GUTENBERG_END_MARKERS:
        idx = text.find(em)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip() + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download text training datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name (repeat for multiple). Default: shakespeare.",
    )
    p.add_argument("--all", action="store_true", help="Download every known dataset.")
    p.add_argument("--list", action="store_true", help="List available datasets and exit.")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    return p.parse_args()


def download_one(name: str, force: bool) -> Path:
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset: {name!r}. Run with --list to see options.")
    url, filename, kind = DATASETS[name]
    path = DATA_DIR / filename

    if path.exists() and not force:
        kb = path.stat().st_size / 1024
        print(f"  {name:<18}  cached  ({kb:>6.0f} KB)  {path.relative_to(PROJECT_ROOT)}")
        return path

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {name:<18}  downloading...", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".tmp", dir=DATA_DIR, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(["curl", "-fsSL", "-o", str(tmp_path), url], check=True)

        if kind == "gutenberg":
            text = tmp_path.read_text(encoding="utf-8", errors="replace")
            text = strip_gutenberg(text)
            path.write_text(text, encoding="utf-8")
        elif kind == "raw":
            tmp_path.replace(path)
        else:
            raise ValueError(f"unknown kind: {kind!r}")
    finally:
        tmp_path.unlink(missing_ok=True)

    kb = path.stat().st_size / 1024
    print(f"  {name:<18}  saved   ({kb:>6.0f} KB)  {path.relative_to(PROJECT_ROOT)}")
    return path


def sweep_stale_tmp_files() -> int:
    """Remove *.tmp files left behind by previous interrupted downloads.

    Our tempfile cleanup lives in a `finally` block, which handles normal
    exits and exceptions but not SIGKILL. Sweeping at startup keeps the
    data dir tidy across kills."""
    if not DATA_DIR.exists():
        return 0
    n = 0
    for p in DATA_DIR.glob("*.tmp"):
        p.unlink(missing_ok=True)
        n += 1
    return n


def main() -> None:
    args = parse_args()

    if args.list:
        print("Available datasets:")
        for name, (_, filename, _) in DATASETS.items():
            print(f"  {name:<18}  -> data/{filename}")
        return

    if args.all:
        names = list(DATASETS)
    elif args.dataset:
        names = args.dataset
    else:
        names = ["shakespeare"]

    n_stale = sweep_stale_tmp_files()
    if n_stale:
        print(f"swept {n_stale} stale .tmp file(s)")

    print(f"target: {len(names)} dataset(s) -> {DATA_DIR.relative_to(PROJECT_ROOT)}/")
    for name in names:
        download_one(name, args.force)


if __name__ == "__main__":
    main()
