"""Download training corpora from the project's GCS bucket.

The bucket (configs/gcs.json) is the canonical data store; every dataset
in it is listed in its manifest.json with a sha256 that downloads are
verified against. To add or update datasets in the bucket, use
scripts/add_dataset.py (that's where origin URLs and processing live).

Examples:
    .venv/bin/python scripts/download_data.py              # everything
    .venv/bin/python scripts/download_data.py --dataset enwik8
    .venv/bin/python scripts/download_data.py --list
    .venv/bin/python scripts/download_data.py --force      # re-download all

Works anywhere `gcloud` is authenticated — including GCE VMs, whose
default service account has read access to project buckets.
"""

import argparse

from gcs_util import PROJECT_ROOT, gs_uri, read_manifest, run, sha256_file

DATA_DIR = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download training datasets from the project GCS bucket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name (repeat for multiple). Default: all registered datasets.",
    )
    p.add_argument("--list", action="store_true", help="List registered datasets and exit.")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    return p.parse_args()


def download_one(name: str, entry: dict, force: bool) -> None:
    path = DATA_DIR / entry["filename"]

    if path.exists() and not force:
        if path.stat().st_size == entry["bytes"]:
            print(f"  {name:<20} cached  ({entry['bytes'] / 1024:>9,.0f} KB)")
            return
        print(f"  {name:<20} size mismatch — re-downloading")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {name:<20} downloading ({entry['bytes'] / 1024:>9,.0f} KB) ...", flush=True)
    run(["gcloud", "storage", "cp", gs_uri(entry["filename"]), str(path)],
        capture_output=True)

    digest = sha256_file(path)
    if digest != entry["sha256"]:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"{name}: sha256 mismatch after download (got {digest[:12]}…, "
            f"manifest says {entry['sha256'][:12]}…)"
        )
    print(f"  {name:<20} saved + verified -> data/{entry['filename']}")


def main() -> None:
    args = parse_args()
    manifest = read_manifest()
    entries = manifest["datasets"]
    if not entries:
        raise SystemExit(
            "bucket manifest is empty — seed it with scripts/add_dataset.py --seed-from-local"
        )

    if args.list:
        print(f"{len(entries)} dataset(s) in {gs_uri('')}:")
        for name, e in sorted(entries.items()):
            print(f"  {name:<20} {e['bytes']:>12,} B  {e['filename']}")
        return

    names = args.dataset or sorted(entries)
    unknown = [n for n in names if n not in entries]
    if unknown:
        raise SystemExit(f"not in bucket manifest: {unknown}. See --list.")

    print(f"target: {len(names)} dataset(s) -> {DATA_DIR.relative_to(PROJECT_ROOT)}/")
    for name in names:
        download_one(name, entries[name], args.force)


if __name__ == "__main__":
    main()
