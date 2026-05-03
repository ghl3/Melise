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
    webster            Webster's Unabridged Dictionary (1913)        ~29 MB
    origin-species     Darwin, On the Origin of Species              ~950 KB
    decline-fall-1     Gibbon, Decline and Fall, Volume 1            ~1.8 MB
    wealth-of-nations  Adam Smith, The Wealth of Nations             ~2.4 MB
    huckleberry-finn   Twain, Adventures of Huckleberry Finn         ~600 KB
    tale-two-cities    Dickens, A Tale of Two Cities                 ~800 KB
    treasure-island    Stevenson, Treasure Island                    ~400 KB
    wizard-of-oz       Baum, The Wonderful Wizard of Oz              ~250 KB
    walden             Thoreau, Walden                               ~600 KB
    anna-karenina      Tolstoy (Garnett tr.), Anna Karenina          ~2.0 MB
    david-copperfield  Dickens, David Copperfield                    ~1.9 MB
    voyage-beagle      Darwin, Voyage of the Beagle                  ~1.0 MB
    meditations        Marcus Aurelius (Long tr.), Meditations       ~300 KB
    relativity         Einstein, Relativity (Special + General)      ~205 KB
    treatise-light     Huygens, Treatise on Light                    ~207 KB
    discourse-method   Descartes, Discourse on the Method            ~147 KB
    descent-of-man     Darwin, The Descent of Man                    ~1.9 MB

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
    "webster": (
        "https://www.gutenberg.org/cache/epub/29765/pg29765.txt",
        "webster_dictionary.txt",
        "gutenberg",
    ),
    "origin-species": (
        "https://www.gutenberg.org/files/1228/1228-0.txt",
        "origin_of_species.txt",
        "gutenberg",
    ),
    "decline-fall-1": (
        "https://www.gutenberg.org/files/731/731-0.txt",
        "decline_and_fall_v1.txt",
        "gutenberg",
    ),
    "wealth-of-nations": (
        "https://www.gutenberg.org/files/3300/3300-0.txt",
        "wealth_of_nations.txt",
        "gutenberg",
    ),
    "huckleberry-finn": (
        "https://www.gutenberg.org/files/76/76-0.txt",
        "huckleberry_finn.txt",
        "gutenberg",
    ),
    "tale-two-cities": (
        "https://www.gutenberg.org/files/98/98-0.txt",
        "tale_of_two_cities.txt",
        "gutenberg",
    ),
    "treasure-island": (
        "https://www.gutenberg.org/files/120/120-0.txt",
        "treasure_island.txt",
        "gutenberg",
    ),
    "wizard-of-oz": (
        "https://www.gutenberg.org/files/55/55-0.txt",
        "wizard_of_oz.txt",
        "gutenberg",
    ),
    "walden": (
        "https://www.gutenberg.org/files/205/205-0.txt",
        "walden.txt",
        "gutenberg",
    ),
    "anna-karenina": (
        "https://www.gutenberg.org/files/1399/1399-0.txt",
        "anna_karenina.txt",
        "gutenberg",
    ),
    "david-copperfield": (
        "https://www.gutenberg.org/files/766/766-0.txt",
        "david_copperfield.txt",
        "gutenberg",
    ),
    "voyage-beagle": (
        "https://www.gutenberg.org/files/944/944-0.txt",
        "voyage_of_the_beagle.txt",
        "gutenberg",
    ),
    "meditations": (
        "https://www.gutenberg.org/files/2680/2680-0.txt",
        "meditations.txt",
        "gutenberg",
    ),
    "relativity": (
        "https://www.gutenberg.org/files/30155/30155-0.txt",
        "relativity.txt",
        "gutenberg",
    ),
    "treatise-light": (
        "https://www.gutenberg.org/files/14725/14725-0.txt",
        "treatise_on_light.txt",
        "gutenberg",
    ),
    "discourse-method": (
        "https://www.gutenberg.org/files/59/59-0.txt",
        "discourse_on_method.txt",
        "gutenberg",
    ),
    "descent-of-man": (
        "https://www.gutenberg.org/files/2300/2300-0.txt",
        "descent_of_man.txt",
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
