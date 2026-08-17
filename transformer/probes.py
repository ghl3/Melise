"""Shared capability probes, runnable at every stage of the pipeline.

Design: docs/runs/gen4-eval-suite.md. One definition of each probe;
the caller picks the surface form: `chat=False` renders raw-text
continuations (pretraining), `chat=True` renders the chat template
with a preamble (SFT/GRPO/serving checkpoints). Scoring is
deterministic — word-boundary matching against enumerated answer
sets, prefix matching for yes/no, difflib ratio for verbatim recall —
never a judge.

Scalars come back as a flat {tag: value} dict whose tags are stable
TensorBoard keys (probe/facts/capitals/heldout, probe/identity/novel,
…); dumps come back as [{id, prompt, text}] for metrics.jsonl.

The facts table lives in configs/facts.json. Its train/heldout split
is a pure function of each entry's id (md5 % 10 < 7 → train), so the
split survives table growth and reorders; the train half is what
gen_fact_sft renders into chat_facts.txt, the heldout half must never
be trained on.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from .chat import sanitize
from .generate import generate
from .rl.tasks import _NAMES as POOL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACTS_PATH = PROJECT_ROOT / "configs" / "facts.json"

# Names never present in training data (checked against POOL_NAMES at
# import). Novel names are generated per-seed on top of these.
HELDOUT_NAMES = [n for n in
                 ("Marisol", "Ingrid", "Kofi", "Saoirse", "Priya", "Bram",
                  "Yusuf", "Astrid")
                 if n not in POOL_NAMES]

_SYL_A = "ba be bi bo bu da de di do du ka ke ki ko ku la le li lo lu".split()
_SYL_B = "mar tor vex lin zan rel pol nim gar sel".split()

VERBATIM_SPEC = [
    # (data file, group). HELDOUT files are never trained on — their
    # similarity is the no-leakage floor; trained-vs-heldout gap is the
    # live memorization index.
    ("fineweb_edu.txt", "fineweb"),
    ("wikitext103_train.txt", "wikitext"),
    ("enwik8.txt", "enwik8"),
    ("war_and_peace.txt", "books"),
    ("sherlock.txt", "books"),
    ("shakespeare_complete.txt", "books"),
    ("dialogue_movies.txt", "dialogue"),
    ("code_python.txt", "code"),
    ("emma.txt", "heldout"),
    ("great_expectations.txt", "heldout"),
]
VERBATIM_KEY, VERBATIM_TARGET = 64, 128

DUMP_PROMPTS = [
    # (id, raw continuation, chat form)
    ("play", "ROMEO:\n", "Continue this play script:\n\nROMEO:"),
    ("novel", "“I am glad to see you,” said the count,",
     "Continue this story: “I am glad to see you,” said the count,"),
    ("wiki", "France is a country in", "Tell me about France."),
    ("expository", "Photosynthesis is the process by which",
     "Explain photosynthesis."),
    ("sonnet", "Shall I compare thee to a summer's day?\n"
     "Thou art more lovely and more temperate:\n",
     "Finish this line of poetry: Shall I compare thee to a summer's day?"),
    ("instances", "Common pets include cats, dogs,", "Name three animals."),
]

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
_REFUSAL_PATTERNS = ("don't know", "do not know", "no clock", "no calendar",
                     "not sure", "cannot tell", "can't tell")


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


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def fact_split(fact_id: str) -> str:
    h = int(hashlib.md5(fact_id.encode()).hexdigest(), 16)
    return "train" if h % 10 < 7 else "heldout"


def load_facts(path: Path = FACTS_PATH) -> list[dict]:
    facts = json.loads(path.read_text())["facts"]
    for f in facts:
        f["split"] = fact_split(f["id"])
        f.setdefault("mode", "contains")
    return facts


def novel_names(seed: int, n: int = 8) -> list[str]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        name = (rng.choice(_SYL_A) + rng.choice(_SYL_B)).capitalize()
        if name not in out and name not in POOL_NAMES:
            out.append(name)
    return out


def random_date(rng: random.Random) -> tuple[str, str, str, int]:
    wd, mo = rng.choice(_WEEKDAYS), rng.choice(_MONTHS)
    day, year = rng.randint(1, 28), rng.randint(2020, 2029)
    return f"{wd}, {mo} {day}, {year}", wd, mo, day


class ProbeRunner:
    """Runs the probe suite against one loaded model.

    chat=False → raw continuations; chat=True → single-turn chat
    template with a per-probe preamble (identity/date probes supply
    their own; others use `preamble`)."""

    def __init__(self, model, tok, device, *, chat: bool, seed: int = 0,
                 preamble: str = "You are Lily, a tiny language model.",
                 facts_per_family: int | None = None,
                 names_per_stratum: int = 6, verbatim_per_file: int = 2):
        self.model, self.tok, self.device = model, tok, device
        self.chat, self.seed, self.preamble = chat, seed, preamble
        self.facts_per_family = facts_per_family
        self.names_per_stratum = names_per_stratum
        self.verbatim_per_file = verbatim_per_file

    # ---- generation ----

    def _gen_ids(self, ids: list[int], max_new: int) -> list[int]:
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        return generate(self.model, t, max_new)

    def gen_raw(self, text: str, max_new: int = 40) -> str:
        return self.tok.decode(self._gen_ids(self.tok.encode(text), max_new))

    def gen_chat(self, user: str, preamble: str | None = None,
                 max_new: int = 48) -> str:
        tok = self.tok
        ids = list(tok.encode(sanitize(preamble or self.preamble).strip()))
        ids += [tok.user_id, *tok.encode(sanitize(user).strip()),
                tok.end_turn_id, tok.assistant_id]
        out = self._gen_ids(ids, max_new)
        if tok.end_turn_id in out:
            out = out[:out.index(tok.end_turn_id)]
        return tok.decode(out)

    def _ask(self, raw: str, chatform: str, max_new: int = 40) -> str:
        return (self.gen_chat(chatform, max_new=max_new) if self.chat
                else self.gen_raw(raw, max_new=max_new))

    # ---- probe families ----

    def probe_facts(self) -> dict[str, float]:
        by_key: dict[str, list[float]] = {}
        facts = load_facts()
        if self.facts_per_family:
            seen: dict[str, int] = {}
            kept = []
            for f in facts:
                if seen.get(f["family"], 0) < self.facts_per_family:
                    seen[f["family"]] = seen.get(f["family"], 0) + 1
                    kept.append(f)
            facts = kept
        for f in facts:
            reply = self._ask(f["cloze"], f["chat"], max_new=24)
            score = (prefix_any if f["mode"] == "prefix" else contains_any)(
                reply, f["answers"])
            for key in (f["family"], f"{f['family']}/{f['split']}"):
                by_key.setdefault(key, []).append(float(score))
        return {f"probe/facts/{k}": sum(v) / len(v)
                for k, v in by_key.items()}

    def probe_identity(self) -> dict[str, float]:
        strata = {"pool": list(POOL_NAMES), "heldout": list(HELDOUT_NAMES),
                  "novel": novel_names(self.seed)}
        rng = random.Random(self.seed)
        questions = ("What is your name?", "Who are you?",
                     "Tell me your name.")
        out = {}
        for label, names in strata.items():
            names = rng.sample(names, min(self.names_per_stratum, len(names)))
            hits = []
            for i, name in enumerate(names):
                pre = f"You are {name}, a tiny language model."
                if self.chat:
                    reply = self.gen_chat(questions[i % 3], preamble=pre,
                                          max_new=24)
                else:
                    reply = self.gen_raw(
                        pre + " When asked who I am, I answer: My name is",
                        max_new=8)
                hits.append(float(contains_any(reply, [name])))
            out[f"probe/identity/{label}"] = sum(hits) / len(hits)
        if self.chat:  # ask-twice consistency, pool names
            same = []
            for name in rng.sample(list(POOL_NAMES), 4):
                pre = f"You are {name}, a tiny language model."
                r1 = self.gen_chat("What is your name?", preamble=pre,
                                   max_new=24)
                r2 = self.gen_chat("Sorry, tell me your name again?",
                                   preamble=pre, max_new=24)
                n1 = self._first_name(r1)
                same.append(float(n1 is not None and n1 == self._first_name(r2)))
            out["probe/identity/consistency"] = sum(same) / len(same)
        return out

    def _first_name(self, text: str):
        known = list(POOL_NAMES) + HELDOUT_NAMES + novel_names(self.seed)
        for w in re.findall(r"[A-Za-z]+", text):
            if w.capitalize() in known:
                return w.capitalize()
        return None

    def probe_date(self) -> dict[str, float]:
        rng = random.Random(self.seed + 1)
        hits = []
        for _ in range(4):
            date, wd, mo, day = random_date(rng)
            if self.chat:
                pre = f"{self.preamble} Today is {date}."
                reply = self.gen_chat("What day is today?", preamble=pre,
                                      max_new=24)
            else:
                reply = self.gen_raw(
                    f"Today is {date}. To repeat, the date today is",
                    max_new=16)
            hits.append(float(contains_any(reply, [wd])
                              or (contains_any(reply, [mo])
                                  and contains_any(reply, [str(day)]))))
        out = {"probe/date/retrieval": sum(hits) / len(hits)}
        if self.chat:
            reply = self.gen_chat("What day is today?", max_new=32)
            out["probe/date/honesty"] = float(
                any(p in reply.lower() for p in _REFUSAL_PATTERNS))
        return out

    def probe_verbatim(self) -> dict[str, float]:
        by_group: dict[str, list[float]] = {}
        for fname, group in VERBATIM_SPEC:
            path = PROJECT_ROOT / "data" / fname
            if not path.exists():
                continue
            size = path.stat().st_size
            span = VERBATIM_KEY + VERBATIM_TARGET
            with open(path, "rb") as fh:
                for i in range(self.verbatim_per_file):
                    h = int(hashlib.sha256(f"{fname}:{i}".encode())
                            .hexdigest(), 16)
                    off = 4096 + h % max(1, size - span - 8192)
                    fh.seek(off)
                    blob = fh.read(span)
                    key = sanitize(blob[:VERBATIM_KEY]
                                   .decode("utf-8", "ignore"))
                    target = blob[VERBATIM_KEY:].decode("utf-8", "ignore")
                    cont = self.gen_raw(key, max_new=48)[:len(target)]
                    by_group.setdefault(group, []).append(
                        similarity(cont, target))
        return {f"probe/verbatim/{g}": sum(v) / len(v)
                for g, v in by_group.items()}

    def probe_dumps(self) -> list[dict]:
        out = []
        for pid, raw, chatform in DUMP_PROMPTS:
            prompt = chatform if self.chat else raw
            text = self._ask(raw, chatform, max_new=80)
            out.append({"id": pid, "prompt": prompt, "text": text})
        return out

    # ---- entry point ----

    def run(self, families=("facts", "identity", "date", "verbatim")):
        scalars: dict[str, float] = {}
        for fam in families:
            scalars.update(getattr(self, f"probe_{fam}")())
        return scalars
