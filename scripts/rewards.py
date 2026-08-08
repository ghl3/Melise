"""Verifiable reward tasks for GRPO (scripts/grpo.py).

Each task is a prompt plus a deterministic scoring function — no reward
model, no human labels. This is the RLVR (RL with verifiable rewards)
recipe GRPO is normally paired with, shrunk to problems a 17M byte model
can actually learn: copying, tiny arithmetic, parity, letter counting,
length control. Scores are in [0, 1].

Every generator draws from its own `random.Random`, so a seeded RNG
reproduces the same task sequence — rollouts stay comparable across
resumes and eval sets stay fixed across a run.

Add a task by writing `make_<name>(rng) -> Task` and registering it in
TASKS. Keep scoring strict enough to resist reward hacking (e.g. exact
integer match, not substring match) — with G rollouts per prompt the
policy will find any slack you leave.
"""

from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass, field
from typing import Callable

_WORDS = (
    "river mountain valley harbor glacier sparrow otter wolf falcon brook "
    "meadow canyon lighthouse garden forest comet dawn willow raven tide "
    "summit apple bread candle door engine feather grape hammer island "
    "jacket kettle ladder mirror needle ocean pencil quilt ribbon saddle"
).split()

_TOPICS = (
    "the sea", "a forest walk", "your favorite meal", "a thunderstorm",
    "an old house", "a long journey", "the night sky", "a small victory",
    "a quiet morning", "a city street",
)

# Word bank for building exactly-N-word canonical answers (words task).
_FILLER = (
    "it feels calm and bright there somehow quietly holding small "
    "moments that drift past slowly like light on water"
).split()

_FIRST_INT = re.compile(r"-?\d+")


@dataclass
class Task:
    """A prompt, its deterministic scorer, and one canonical full-credit
    answer. `answer` is what cold-start SFT data trains toward
    (scripts/gen_task_sft.py) — keeping it on the Task guarantees the
    SFT target and the RL scorer can never drift apart."""

    kind: str
    prompt: str
    answer: str
    score: Callable[[str], float] = field(repr=False)


def _first_int(text: str) -> int | None:
    m = _FIRST_INT.search(text)
    return int(m.group()) if m else None


def make_copy(rng: random.Random) -> Task:
    """Repeat a short phrase exactly. Partial credit by similarity ratio,
    so early training gets a gradient before exact copying clicks."""
    phrase = " ".join(rng.sample(_WORDS, rng.randint(2, 5)))
    def score(text: str) -> float:
        if text.strip() == phrase:
            return 1.0
        return 0.5 * difflib.SequenceMatcher(None, text.strip(), phrase).ratio()
    return Task("copy", f'Repeat exactly: {phrase}', phrase, score)


def make_arith(rng: random.Random) -> Task:
    """Two-operand addition/subtraction, answer ≤ 2 digits. All-or-nothing
    on the first integer in the reply."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    if rng.random() < 0.5:
        prompt, answer = f"What is {a} + {b}?", a + b
    else:
        a, b = max(a, b), min(a, b)
        prompt, answer = f"What is {a} - {b}?", a - b
    return Task("arith", prompt, str(answer),
                lambda t: float(_first_int(t) == answer))


def make_parity(rng: random.Random) -> Task:
    """'Is N even or odd?' — verifiable single-word answer."""
    n = rng.randint(1, 999)
    answer = "even" if n % 2 == 0 else "odd"
    wrong = "odd" if n % 2 == 0 else "even"
    def score(text: str) -> float:
        t = text.strip().lower()
        # Saying both words (or the wrong one) scores nothing.
        return float(answer in t and wrong not in t.replace(answer, "", 1))
    return Task("parity", f"Is {n} even or odd? Answer with one word.",
                answer.capitalize(), score)


def make_count_letter(rng: random.Random) -> Task:
    """Count occurrences of a letter in a word — the classic tokenizer
    stumper, but byte models see every letter."""
    word = rng.choice(_WORDS)
    letter = rng.choice(sorted(set(word)))
    answer = word.count(letter)
    return Task(
        "count",
        f"How many times does the letter '{letter}' appear in '{word}'?",
        str(answer),
        lambda t: float(_first_int(t) == answer),
    )


def make_word_count(rng: random.Random) -> Task:
    """Answer in exactly N words. Content is unjudged — only the count is
    scored (linear partial credit)."""
    n = rng.randint(3, 8)
    topic = rng.choice(_TOPICS)
    # Canonical answer: topic words first, filler to exactly n words.
    base = topic.split()[-1:] if n <= 3 else topic.split()
    pool = [w for w in _FILLER if w not in base]
    words = (base + rng.sample(pool, max(n - len(base), 0)))[:n]
    canonical = " ".join(words).capitalize().rstrip(",") + "."
    def score(text: str) -> float:
        k = len(text.split())
        return max(0.0, 1.0 - abs(k - n) / n) if k else 0.0
    return Task("words", f"Describe {topic} in exactly {n} words.", canonical, score)


TASKS: dict[str, Callable[[random.Random], Task]] = {
    "copy": make_copy,
    "arith": make_arith,
    "parity": make_parity,
    "count": make_count_letter,
    "words": make_word_count,
}


def sample_tasks(names: list[str], n: int, rng: random.Random) -> list[Task]:
    """n tasks drawn round-robin-ish (uniformly) from the named kinds."""
    return [TASKS[rng.choice(names)](rng) for _ in range(n)]
