"""The closed-world facts table (configs/facts.json) and its
train/heldout split — shared by the probe suite (transformer.probes),
the facts RLVR task (transformer.rl.tasks), and the SFT corpus
generator (scripts/gen_fact_sft.py).

Lives in its own module so rl.tasks can consume facts without importing
transformer.probes (which imports rl.tasks — a cycle otherwise).

The split is a pure function of each entry's id (md5 % 10 < 7 → train),
so it survives table growth and reorders. The train half is what
gen_fact_sft renders into chat_facts.txt and what the facts RLVR task
draws from; the heldout half must NEVER be trained on, in any stage —
probe/facts/*/heldout measures whether the model learned facts from the
mix rather than memorized QA pairs.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = PROJECT_ROOT / "configs" / "facts.json"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains_any(text: str, answers) -> bool:
    """Word-boundary containment of any accepted answer."""
    n = f" {_norm(text)} "
    return any(f" {_norm(a)} " in n for a in answers)


def prefix_any(text: str, answers) -> bool:
    """First word matches (yes/no style)."""
    words = _norm(text).split()
    return bool(words) and any(words[0] == _norm(a) for a in answers)


def fact_split(fact_id: str) -> str:
    h = int(hashlib.md5(fact_id.encode()).hexdigest(), 16)
    return "train" if h % 10 < 7 else "heldout"


@lru_cache(maxsize=4)
def _load(path_str: str) -> tuple[dict, ...]:
    facts = json.loads(Path(path_str).read_text())["facts"]
    for f in facts:
        f["split"] = fact_split(f["id"])
        f.setdefault("mode", "contains")
    return tuple(facts)


def load_facts(path: Path = FACTS_PATH, split: str | None = None) -> list[dict]:
    """The fact table, split-annotated. `split` filters to one half
    ('train' for anything that generates training data)."""
    facts = [dict(f) for f in _load(str(path))]
    if split is not None:
        facts = [f for f in facts if f["split"] == split]
    return facts
