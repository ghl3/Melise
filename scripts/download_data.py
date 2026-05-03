"""Download a small text corpus for training (TinyShakespeare).

~1MB of Shakespeare plays concatenated. Standard toy-LM dataset (used by
Karpathy's char-rnn, nanoGPT, and many tutorials). Saves to
`data/tinyshakespeare.txt` at the project root.

Run from project root:
    .venv/bin/python scripts/download_data.py

Note: uses curl rather than urllib to dodge macOS python.org SSL cert
verification issues. macOS and most Linux distros ship curl by default.
"""

import subprocess
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tinyshakespeare.txt"


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        size_kb = DATA_PATH.stat().st_size / 1024
        print(f"already downloaded: {DATA_PATH}  ({size_kb:.1f} KB)")
        return

    print(f"downloading {URL}")
    print(f"        to {DATA_PATH}")
    subprocess.run(["curl", "-fsSL", "-o", str(DATA_PATH), URL], check=True)
    size_kb = DATA_PATH.stat().st_size / 1024
    print(f"done — {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
