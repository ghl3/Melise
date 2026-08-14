"""Add datasets to the project's GCS bucket and register them.

The bucket (configs/gcs.json) is the canonical data store; training
machines pull from it with scripts/download_data.py. This script is the
only writer: it fetches a corpus from its origin (Gutenberg, HTTP, zip),
processes it (strip Gutenberg boilerplate / extract the zip member),
uploads the result, and records it in the bucket's manifest.json
(name, filename, bytes, sha256, source URL, kind, date added).

Usage:

    Seed the bucket from an existing local data/ directory:
        .venv/bin/python scripts/add_dataset.py --seed-from-local

    Add a known dataset (see ORIGINS below) by name:
        .venv/bin/python scripts/add_dataset.py --name middlemarch

    Add and register a brand-new dataset:
        .venv/bin/python scripts/add_dataset.py --name my-corpus \\
            --url https://example.com/corpus.txt --kind raw

    List what's registered in the bucket:
        .venv/bin/python scripts/add_dataset.py --list

Kinds: "raw" (save as-is), "gutenberg" (strip the Project Gutenberg
legal header/footer), "zip:<member>" (extract one zip member
byte-for-byte — enwik8 is a byte-exact benchmark; never re-encode it).

Requires write access to the bucket — run from an authenticated
workstation, not a read-only VM.
"""

import argparse
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path

from gcs_util import PROJECT_ROOT, gs_uri, read_manifest, run, sha256_file, write_manifest

DATA_DIR = PROJECT_ROOT / "data"


# Known origins: name -> (url, output filename, kind).
ORIGINS: dict[str, tuple[str, str, str]] = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "tinyshakespeare.txt",
        "raw",
    ),
    "alice": ("https://www.gutenberg.org/files/11/11-0.txt", "alice.txt", "gutenberg"),
    "frankenstein": ("https://www.gutenberg.org/files/84/84-0.txt", "frankenstein.txt", "gutenberg"),
    "pride-prejudice": ("https://www.gutenberg.org/files/1342/1342-0.txt", "pride_prejudice.txt", "gutenberg"),
    "sherlock": ("https://www.gutenberg.org/files/1661/1661-0.txt", "sherlock.txt", "gutenberg"),
    "moby-dick": ("https://www.gutenberg.org/files/2701/2701-0.txt", "moby_dick.txt", "gutenberg"),
    "bible-kjv": ("https://www.gutenberg.org/files/10/10-0.txt", "bible_kjv.txt", "gutenberg"),
    "shakespeare-all": ("https://www.gutenberg.org/files/100/100-0.txt", "shakespeare_complete.txt", "gutenberg"),
    "webster": ("https://www.gutenberg.org/cache/epub/29765/pg29765.txt", "webster_dictionary.txt", "gutenberg"),
    "origin-species": ("https://www.gutenberg.org/files/1228/1228-0.txt", "origin_of_species.txt", "gutenberg"),
    "decline-fall-1": ("https://www.gutenberg.org/files/731/731-0.txt", "decline_and_fall_v1.txt", "gutenberg"),
    "wealth-of-nations": ("https://www.gutenberg.org/files/3300/3300-0.txt", "wealth_of_nations.txt", "gutenberg"),
    "huckleberry-finn": ("https://www.gutenberg.org/files/76/76-0.txt", "huckleberry_finn.txt", "gutenberg"),
    "tale-two-cities": ("https://www.gutenberg.org/files/98/98-0.txt", "tale_of_two_cities.txt", "gutenberg"),
    "treasure-island": ("https://www.gutenberg.org/files/120/120-0.txt", "treasure_island.txt", "gutenberg"),
    "wizard-of-oz": ("https://www.gutenberg.org/files/55/55-0.txt", "wizard_of_oz.txt", "gutenberg"),
    "walden": ("https://www.gutenberg.org/files/205/205-0.txt", "walden.txt", "gutenberg"),
    "anna-karenina": ("https://www.gutenberg.org/files/1399/1399-0.txt", "anna_karenina.txt", "gutenberg"),
    "david-copperfield": ("https://www.gutenberg.org/files/766/766-0.txt", "david_copperfield.txt", "gutenberg"),
    "voyage-beagle": ("https://www.gutenberg.org/files/944/944-0.txt", "voyage_of_the_beagle.txt", "gutenberg"),
    "meditations": ("https://www.gutenberg.org/files/2680/2680-0.txt", "meditations.txt", "gutenberg"),
    "relativity": ("https://www.gutenberg.org/files/30155/30155-0.txt", "relativity.txt", "gutenberg"),
    "treatise-light": ("https://www.gutenberg.org/files/14725/14725-0.txt", "treatise_on_light.txt", "gutenberg"),
    "discourse-method": ("https://www.gutenberg.org/files/59/59-0.txt", "discourse_on_method.txt", "gutenberg"),
    "descent-of-man": ("https://www.gutenberg.org/files/2300/2300-0.txt", "descent_of_man.txt", "gutenberg"),
    "war-and-peace": ("https://www.gutenberg.org/files/2600/2600-0.txt", "war_and_peace.txt", "gutenberg"),
    "monte-cristo": ("https://www.gutenberg.org/files/1184/1184-0.txt", "monte_cristo.txt", "gutenberg"),
    "don-quixote": ("https://www.gutenberg.org/files/996/996-0.txt", "don_quixote.txt", "gutenberg"),
    "les-miserables": ("https://www.gutenberg.org/files/135/135-0.txt", "les_miserables.txt", "gutenberg"),
    "middlemarch": ("https://www.gutenberg.org/files/145/145-0.txt", "middlemarch.txt", "gutenberg"),
    "brothers-karamazov": ("https://www.gutenberg.org/files/28054/28054-0.txt", "brothers_karamazov.txt", "gutenberg"),
    "grimms": ("https://www.gutenberg.org/files/2591/2591-0.txt", "grimms_fairy_tales.txt", "gutenberg"),
    # Gen-4 fiction expansion (2026-08-14): diversify the fiction group —
    # new authors, era spread incl. 1920s public domain. emma and
    # great-expectations are FULL HOLDOUTS (never trained; test-only).
    "dracula": ("https://www.gutenberg.org/cache/epub/345/pg345.txt", "dracula.txt", "gutenberg"),
    "jane-eyre": ("https://www.gutenberg.org/cache/epub/1260/pg1260.txt", "jane_eyre.txt", "gutenberg"),
    "wuthering-heights": ("https://www.gutenberg.org/cache/epub/768/pg768.txt", "wuthering_heights.txt", "gutenberg"),
    "dorian-gray": ("https://www.gutenberg.org/cache/epub/174/pg174.txt", "dorian_gray.txt", "gutenberg"),
    "time-machine": ("https://www.gutenberg.org/cache/epub/35/pg35.txt", "time_machine.txt", "gutenberg"),
    "war-of-the-worlds": ("https://www.gutenberg.org/cache/epub/36/pg36.txt", "war_of_the_worlds.txt", "gutenberg"),
    "emma": ("https://www.gutenberg.org/cache/epub/158/pg158.txt", "emma.txt", "gutenberg"),
    "great-expectations": ("https://www.gutenberg.org/cache/epub/1400/pg1400.txt", "great_expectations.txt", "gutenberg"),
    "tom-sawyer": ("https://www.gutenberg.org/cache/epub/74/pg74.txt", "tom_sawyer.txt", "gutenberg"),
    "call-of-the-wild": ("https://www.gutenberg.org/cache/epub/215/pg215.txt", "call_of_the_wild.txt", "gutenberg"),
    "heart-of-darkness": ("https://www.gutenberg.org/cache/epub/219/pg219.txt", "heart_of_darkness.txt", "gutenberg"),
    "crime-and-punishment": ("https://www.gutenberg.org/cache/epub/2554/pg2554.txt", "crime_and_punishment.txt", "gutenberg"),
    "madame-bovary": ("https://www.gutenberg.org/cache/epub/2413/pg2413.txt", "madame_bovary.txt", "gutenberg"),
    "age-of-innocence": ("https://www.gutenberg.org/cache/epub/541/pg541.txt", "age_of_innocence.txt", "gutenberg"),
    "little-women": ("https://www.gutenberg.org/cache/epub/514/pg514.txt", "little_women.txt", "gutenberg"),
    "great-gatsby": ("https://www.gutenberg.org/cache/epub/64317/pg64317.txt", "great_gatsby.txt", "gutenberg"),
    "enwik8": ("https://mattmahoney.net/dc/enwik8.zip", "enwik8.txt", "zip:enwik8"),
    "wikitext-103": (
        "https://wikitext.smerity.com/wikitext-103-raw-v1.zip",
        "wikitext103_train.txt",
        "zip:wikitext-103-raw/wiki.train.raw",
    ),
}

_FILENAME_TO_NAME = {filename: name for name, (_, filename, _) in ORIGINS.items()}

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


def fetch_and_process(url: str, kind: str, dest: Path) -> None:
    """Download from origin and write the processed corpus to `dest`."""
    with tempfile.NamedTemporaryFile(suffix=".tmp", dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(["curl", "-fsSL", "-o", str(tmp_path), url], check=True)
        if kind == "gutenberg":
            dest.write_text(strip_gutenberg(
                tmp_path.read_text(encoding="utf-8", errors="replace")
            ), encoding="utf-8")
        elif kind == "raw":
            tmp_path.replace(dest)
        elif kind.startswith("zip:"):
            member = kind.split(":", 1)[1]
            with zipfile.ZipFile(tmp_path) as zf:
                dest.write_bytes(zf.read(member))
        else:
            raise SystemExit(f"unknown kind: {kind!r}")
    finally:
        tmp_path.unlink(missing_ok=True)


def register(name: str, path: Path, url: str, kind: str, force: bool) -> None:
    """Upload `path` to the bucket and record it in the manifest."""
    manifest = read_manifest()
    digest = sha256_file(path)
    entry = manifest["datasets"].get(name)
    if entry and entry.get("sha256") == digest and not force:
        print(f"  {name:<18}  already registered (same sha256)")
        return
    print(f"  {name:<18}  uploading {path.stat().st_size / 1024:.0f} KB ...", flush=True)
    run(["gcloud", "storage", "cp", str(path), gs_uri(path.name)], capture_output=True)
    manifest["datasets"][name] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "source_url": url,
        "kind": kind,
        "added": date.today().isoformat(),
    }
    write_manifest(manifest)
    print(f"  {name:<18}  registered -> {gs_uri(path.name)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add datasets to the GCS bucket and register them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--name", action="append", default=[],
                   help="Dataset to add (a known ORIGINS name, or a new name with --url)")
    p.add_argument("--url", default=None, help="Origin URL for a new dataset")
    p.add_argument("--kind", default="raw",
                   help="Processing for a new dataset: raw | gutenberg | zip:<member>")
    p.add_argument("--seed-from-local",
                   action="store_true",
                   help="Upload and register every file already in data/")
    p.add_argument("--list", action="store_true", help="Show the bucket manifest")
    p.add_argument("--force", action="store_true", help="Re-upload even if sha256 matches")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        manifest = read_manifest()
        entries = manifest["datasets"]
        print(f"{len(entries)} dataset(s) registered:")
        for name, e in sorted(entries.items()):
            print(f"  {name:<20} {e['bytes']:>12,} B  {e['filename']}")
        return

    if args.seed_from_local:
        files = sorted(DATA_DIR.glob("*.txt"))
        if not files:
            raise SystemExit("data/ has no .txt files to seed from")
        print(f"seeding bucket from {len(files)} local file(s):")
        for path in files:
            name = _FILENAME_TO_NAME.get(path.name, path.stem.replace("_", "-"))
            url, _, kind = ORIGINS.get(name, ("local", path.name, "raw"))
            register(name, path, url, kind, args.force)
        return

    if not args.name:
        raise SystemExit("nothing to do — use --name, --seed-from-local, or --list")

    for name in args.name:
        if name in ORIGINS:
            url, filename, kind = ORIGINS[name]
        elif args.url:
            url, filename, kind = args.url, f"{name.replace('-', '_')}.txt", args.kind
        else:
            raise SystemExit(f"{name!r} is not a known dataset; pass --url/--kind for new ones")
        dest = DATA_DIR / filename
        if not dest.exists() or args.force:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  {name:<18}  fetching from origin ...", flush=True)
            fetch_and_process(url, kind, dest)
        register(name, dest, url, kind, args.force)


if __name__ == "__main__":
    main()
