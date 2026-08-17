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
…); dumps come back as [{id, prompt, temp, text}] for metrics.jsonl.

In-loop use (pretrain.py/sft.py/grpo.py --probe-every): one ProbeRunner
per run, seed fixed to the run seed so every probe round scores the
SAME prompts/names/windows — curves are comparable step-to-step and
across stages. Full-table sweeps happen offline on keeper checkpoints
(scripts/probe_checkpoint.py). The facts table itself lives in
transformer.facts; name pools and dates in transformer.identity (the
import-time asserts there are what keep the heldout/novel strata
honest).
"""

from __future__ import annotations

import difflib
import hashlib
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .chat import sanitize
from .facts import (FACTS_PATH, contains_any, fact_split,  # noqa: F401
                    load_facts, prefix_any)
from .generate import generate
from .identity import (HELDOUT_NAMES, REFUSAL_PATTERNS,  # noqa: F401
                       TRAIN_NAMES, novel_names, random_date)
from .rl.tasks import _NAMES as POOL_NAMES  # noqa: F401 (legacy re-export)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    ("dialogue", "“Where were you last night?”\n“I told you already.”\n",
     "Let's chat! How is your day going?"),
    ("code", "def fibonacci(n):\n", "Write a fibonacci function in Python."),
    ("sonnet", "Shall I compare thee to a summer's day?\n"
     "Thou art more lovely and more temperate:\n",
     "Finish this line of poetry: Shall I compare thee to a summer's day?"),
    ("instances", "Common pets include cats, dogs,", "Name three animals."),
]

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class ProbeRunner:
    """Runs the probe suite against one loaded model.

    chat=False → raw continuations; chat=True → single-turn chat
    template with a per-probe preamble (identity/date probes supply
    their own; others use `preamble`). Toggles eval mode around every
    entry point, so it can be called mid-training."""

    def __init__(self, model, tok, device, *, chat: bool, seed: int = 0,
                 preamble: str = "You are Melise, a tiny language model.",
                 facts_per_family: int | None = None,
                 names_per_stratum: int = 6, verbatim_per_file: int = 2):
        self.model, self.tok, self.device = model, tok, device
        self.chat, self.seed, self.preamble = chat, seed, preamble
        self.facts_per_family = facts_per_family
        self.names_per_stratum = names_per_stratum
        self.verbatim_per_file = verbatim_per_file
        self._sample_rng = torch.Generator().manual_seed(seed)

    # ---- generation ----

    def _gen_ids(self, ids: list[int], max_new: int,
                 temperature: float = 0.0) -> list[int]:
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        if temperature <= 0:
            return generate(self.model, t, max_new)
        # Temperature sampling through a private CPU generator: the
        # training loop's RNG streams (exact-resume contract) are never
        # touched by probe sampling.
        def fn(logits: torch.Tensor) -> int:
            probs = F.softmax(logits.float().cpu() / temperature, dim=-1)
            return int(torch.multinomial(
                probs, 1, generator=self._sample_rng).item())
        return generate(self.model, t, max_new, sample_fn=fn)

    def gen_raw(self, text: str, max_new: int = 40,
                temperature: float = 0.0) -> str:
        return self.tok.decode(
            self._gen_ids(self.tok.encode(text), max_new, temperature))

    def gen_chat(self, user: str, preamble: str | None = None,
                 max_new: int = 48, temperature: float = 0.0) -> str:
        tok = self.tok
        ids = list(tok.encode(sanitize(preamble or self.preamble).strip()))
        ids += [tok.user_id, *tok.encode(sanitize(user).strip()),
                tok.end_turn_id, tok.assistant_id]
        out = self._gen_ids(ids, max_new, temperature)
        if tok.end_turn_id in out:
            out = out[:out.index(tok.end_turn_id)]
        return tok.decode(out)

    def _ask(self, raw: str, chatform: str, max_new: int = 40,
             temperature: float = 0.0) -> str:
        return (self.gen_chat(chatform, max_new=max_new,
                              temperature=temperature) if self.chat
                else self.gen_raw(raw, max_new=max_new,
                                  temperature=temperature))

    # ---- probe families ----

    def probe_facts(self) -> dict[str, float]:
        by_key: dict[str, list[float]] = {}
        facts = load_facts()
        if self.facts_per_family:
            # Fixed per-family subset (first N in table order — stable
            # as the table grows by appending): in-loop curves stay
            # comparable; the full table runs offline on keepers.
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
            # Forced-choice entries: naming the wrong option too voids
            # credit ("hot or cold" echoes score nothing).
            if score and f.get("reject") and contains_any(reply, f["reject"]):
                score = False
            for key in (f["family"], f"{f['family']}/{f['split']}"):
                by_key.setdefault(key, []).append(float(score))
        return {f"probe/facts/{k}": sum(v) / len(v)
                for k, v in by_key.items()}

    def probe_identity(self) -> dict[str, float]:
        strata = {"pool": list(TRAIN_NAMES), "heldout": list(HELDOUT_NAMES),
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
            for name in rng.sample(list(TRAIN_NAMES), 4):
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
        known = (list(TRAIN_NAMES) + list(HELDOUT_NAMES)
                 + novel_names(self.seed))
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
                any(p in reply.lower() for p in REFUSAL_PATTERNS))
        return out

    def probe_verbatim(self) -> dict[str, float]:
        by_group: dict[str, list[float]] = {}
        by_chat: dict[str, list[float]] = {}
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
                    # Raw form always — on a post-train checkpoint this
                    # is base-skill forgetting, live (gen-3's +0.74 bpb
                    # would have been visible per-group, per-eval).
                    cont = self.gen_raw(key, max_new=48)[:len(target)]
                    by_group.setdefault(group, []).append(
                        similarity(cont, target))
                    if self.chat:  # instruction-following variant
                        reply = self.gen_chat(
                            "Continue this text exactly: " + key,
                            max_new=48)[:len(target)]
                        by_chat.setdefault(group, []).append(
                            similarity(reply, target))
        out = {f"probe/verbatim/{g}": sum(v) / len(v)
               for g, v in by_group.items()}
        out.update({f"probe/verbatim_chat/{g}": sum(v) / len(v)
                    for g, v in by_chat.items()})
        return out

    def probe_task_formats(self) -> dict[str, float]:
        """Chat-only: greedy 1-shot spot-checks of every RLVR task
        format during SFT — a cheap preview of GRPO liveness (a task
        whose format never generates will hand GRPO all-zero groups)."""
        if not self.chat:
            return {}
        from .rl.tasks import TASKS
        out = {}
        for kind, make in TASKS.items():
            scores = []
            for i in range(2):
                t = make(random.Random(f"{self.seed}:fmt:{kind}:{i}"))
                pre = getattr(t, "preamble", None) or self.preamble
                reply = self.gen_chat(t.prompt, preamble=pre, max_new=64)
                scores.append(float(t.score(reply.strip())))
            out[f"probe/task/{kind}"] = sum(scores) / len(scores)
        return out

    def probe_dumps(self, rotation: int | None = None, k: int = 3,
                    temps: tuple = (0.0, 0.8)) -> list[dict]:
        """Fixed-prompt generation dumps (the ROMEO keyhole,
        pluralized): greedy + temperature variants per prompt.
        rotation=None emits the full battery; an integer emits k
        prompts per round, cycling, so in-loop cost stays at gen-3's
        sample cadence while the full battery lands every ~len/k
        rounds."""
        prompts = DUMP_PROMPTS
        if rotation is not None:
            n = len(DUMP_PROMPTS)
            prompts = [DUMP_PROMPTS[(rotation * k + i) % n] for i in range(k)]
        was_training = self.model.training
        self.model.eval()
        try:
            out = []
            for pid, raw, chatform in prompts:
                prompt = chatform if self.chat else raw
                for temp in temps:
                    text = self._ask(raw, chatform, max_new=80,
                                     temperature=temp)
                    out.append({"id": pid, "prompt": prompt,
                                "temp": temp, "text": text})
            return out
        finally:
            if was_training:
                self.model.train()

    # ---- entry point ----

    def run(self, families=("facts", "identity", "date", "verbatim",
                            "task_formats")):
        was_training = self.model.training
        self.model.eval()
        try:
            scalars: dict[str, float] = {}
            with torch.no_grad():
                for fam in families:
                    scalars.update(getattr(self, f"probe_{fam}")())
            return scalars
        finally:
            if was_training:
                self.model.train()
