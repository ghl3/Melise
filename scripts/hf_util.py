"""Plain-HTTPS HuggingFace dataset fetching shared by the prep scripts.

No datasets/huggingface_hub dependency — files come straight off the
resolve/ endpoint and directory listings off the public tree API.
certifi supplies CA certs (macOS framework Python ships without system
certs, so default urllib SSL verification fails there).
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

import certifi

HF = "https://huggingface.co"
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
_UA = {"User-Agent": "transformer-learning/1.0"}


def fetch(url: str, dest: Path) -> Path:
    """Stream a URL to disk (skips if already cached)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=_UA)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urllib.request.urlopen(req, context=SSL_CTX) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
    tmp.replace(dest)
    print(f"  fetched {dest.name} ({done / 1e6:.1f} MB)")
    return dest


def hf_list(repo: str, subdir: str) -> list[dict]:
    """List files in a HF dataset repo directory via the public API."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{subdir}"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, context=SSL_CTX) as r:
        return json.loads(r.read())


def hf_resolve(repo: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/main/{path}"
