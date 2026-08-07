"""Shared helpers for the GCS-backed dataset store.

The bucket (see configs/gcs.json) is the canonical home of the training
corpora. Layout:

    gs://<bucket>/<prefix>/<filename>       processed dataset files
    gs://<bucket>/<prefix>/manifest.json    registry of what's in the bucket

The manifest maps dataset name → {filename, bytes, sha256, source_url,
kind, added}. scripts/add_dataset.py writes it; scripts/download_data.py
reads it. All bucket I/O shells out to `gcloud storage`, which is
authenticated on the laptop and via the service account on GCE VMs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GCS_CONFIG = PROJECT_ROOT / "configs" / "gcs.json"


def bucket_prefix() -> str:
    cfg = json.loads(GCS_CONFIG.read_text())
    return f"gs://{cfg['bucket']}/{cfg['prefix']}"


def gs_uri(filename: str) -> str:
    return f"{bucket_prefix()}/{filename}"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kwargs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> dict:
    """Fetch the bucket manifest. Missing manifest → empty registry."""
    try:
        out = subprocess.run(
            ["gcloud", "storage", "cat", gs_uri("manifest.json")],
            check=True, capture_output=True, text=True,
        )
        return json.loads(out.stdout)
    except subprocess.CalledProcessError:
        return {"datasets": {}}


def write_manifest(manifest: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(manifest, tmp, indent=2, sort_keys=True)
        tmp_path = Path(tmp.name)
    try:
        run(["gcloud", "storage", "cp", str(tmp_path), gs_uri("manifest.json")],
            capture_output=True)
    finally:
        tmp_path.unlink(missing_ok=True)
