"""Verifiable reward tasks for GRPO (scripts/grpo.py).

Each task is a prompt plus a deterministic scoring function — no reward
model, no human labels. This is the RLVR (RL with verifiable rewards)
recipe GRPO is normally paired with, shrunk to problems a tiny model
can actually learn: copying, tiny arithmetic, parity, letter counting,
length control, and in-context recall. Scores are in [0, 1].

Every family draws its prompt from several paraphrase templates (added
after gen-2: skills were template-locked — parity scored 1.00 on the
seeded eval yet failed on rewordings). Note this means seeded eval sets
are NOT comparable across this change.

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

from ..facts import contains_any, load_facts, prefix_any
from ..identity import (MONTHS, REFUSAL_PATTERNS, TRAIN_NAMES, WEEKDAYS,
                        random_date)

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

_NAMES = (
    "Frank George Alice Maria Tom Sara Omar Nina Leo Ruth Ivan Ana "
    "Peter Lucy Sam Rosa Karl Vera Hugo Elena Jack Iris Noah Cora"
).split()


@dataclass
class Task:
    """A prompt, its deterministic scorer, and one canonical full-credit
    answer. `answer` is what cold-start SFT data trains toward
    (scripts/gen_task_sft.py) — keeping it on the Task guarantees the
    SFT target and the RL scorer can never drift apart.

    `preamble` is set by tasks whose reward depends on the system
    preamble's CONTENT (context_recall carries random name/date fields
    there); None means the caller applies the deployed default. Rollout
    and eval prompt assembly must honor it — see scripts/grpo.py."""

    kind: str
    prompt: str
    answer: str
    score: Callable[[str], float] = field(repr=False)
    preamble: str | None = None


def _last_int(text: str) -> int | None:
    """The LAST integer in a reply is its answer. This is what lets
    worked-steps responses ('47 + 3 = 50, 50 + 5 = 55. 55') score
    correctly — intermediate results appear before the answer, never
    after. First-int scoring would punish showing work."""
    m = _FIRST_INT.findall(text)
    return int(m[-1]) if m else None


def _worked(a: int, b: int, sub: bool) -> str:
    """Canonical worked-steps answer: decompose through the nearest ten
    when a carry/borrow is involved, then state the answer last (the
    scorer reads the last integer)."""
    if sub:
        total, step = a - b, a % 10
        if 0 < step < b:
            mid = a - step
            return f"{a} - {step} = {mid}, {mid} - {b - step} = {total}. {total}"
        return f"{a} - {b} = {total}"
    total, step = a + b, (10 - a % 10) % 10
    if 0 < step < b:
        mid = a + step
        return f"{a} + {step} = {mid}, {mid} + {b - step} = {total}. {total}"
    return f"{a} + {b} = {total}"


def make_copy(rng: random.Random) -> Task:
    """Repeat a short phrase exactly. Partial credit by similarity ratio,
    so early training gets a gradient before exact copying clicks."""
    phrase = " ".join(rng.sample(_WORDS, rng.randint(2, 5)))
    prompt = rng.choice((
        f"Repeat exactly: {phrase}",
        f"Say exactly: {phrase}",
        f"Echo this exactly: {phrase}",
        f"Repeat after me: {phrase}",
    ))
    def score(text: str) -> float:
        if text.strip() == phrase:
            return 1.0
        return 0.5 * difflib.SequenceMatcher(None, text.strip(), phrase).ratio()
    return Task("copy", prompt, phrase, score)


def make_arith(rng: random.Random) -> Task:
    """Two-operand addition/subtraction, scored on the LAST integer in
    the reply, so the model may (and the canonical answer does) show
    carry/borrow steps before answering. A wrong answer that still shows
    worked-steps evidence ('=') gets 0.25: gen-2 proved that when every
    rollout in a group scores 0, the z-scored advantage is identically
    zero and RL can never bootstrap the format — partial format credit
    gives the first worked-steps rollout a nonzero gradient."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    sub = rng.random() < 0.5
    if sub:
        a, b = max(a, b), min(a, b)
        op, answer = "-", a - b
    else:
        op, answer = "+", a + b
    prompt = rng.choice((
        f"What is {a} {op} {b}?",
        f"What's {a} {op} {b}?",
        f"Compute {a} {op} {b}.",
        f"{a} {op} {b} = ?",
    ))
    def score(text: str) -> float:
        if _last_int(text) == answer:
            return 1.0
        return 0.25 if "=" in text else 0.0
    return Task("arith", prompt, _worked(a, b, sub), score)


def make_parity(rng: random.Random) -> Task:
    """'Is N even or odd?' — verifiable single-word answer."""
    n = rng.randint(1, 999)
    answer = "even" if n % 2 == 0 else "odd"
    wrong = "odd" if n % 2 == 0 else "even"
    prompt = rng.choice((
        f"Is {n} even or odd? Answer with one word.",
        f"Is the number {n} even or odd?",
        f"Even or odd: {n}?",
        f"Tell me whether {n} is even or odd.",
    ))
    def score(text: str) -> float:
        t = text.strip().lower()
        # Saying both words (or the wrong one) scores nothing.
        return float(answer in t and wrong not in t.replace(answer, "", 1))
    return Task("parity", prompt, answer.capitalize(), score)


def make_count_letter(rng: random.Random) -> Task:
    """Count occurrences of a letter in a word — the classic tokenizer
    stumper, but byte models see every letter."""
    word = rng.choice(_WORDS)
    letter = rng.choice(sorted(set(word)))
    answer = word.count(letter)
    prompt = rng.choice((
        f"How many times does the letter '{letter}' appear in '{word}'?",
        f"Count the letter '{letter}' in '{word}'.",
        f"How many '{letter}'s are in '{word}'?",
    ))
    return Task("count", prompt, str(answer),
                lambda t: float(_last_int(t) == answer))


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
    prompt = rng.choice((
        f"Describe {topic} in exactly {n} words.",
        f"In exactly {n} words, describe {topic}.",
        f"Write exactly {n} words about {topic}.",
    ))
    def score(text: str) -> float:
        k = len(text.split())
        return max(0.0, 1.0 - abs(k - n) / n) if k else 0.0
    return Task("words", prompt, canonical, score)


def make_recall(rng: random.Random) -> Task:
    """State a name, then ask for it back — in-context retrieval plus the
    person-deixis flip gen-2 lacked (it answered 'What is my name?' with
    'My name is Frank.', claiming the name as its own). Full credit
    requires the right name WITHOUT the first-person echo; the echo gets
    half credit (right fact, wrong voice). A length cap blocks the
    list-every-name hack. Names draw from the full gen-4 training pool
    (~370) so copying, not memorizing, is what pays."""
    name = rng.choice(TRAIN_NAMES)
    prompt = rng.choice((
        f"My name is {name}. What is my name?",
        f"I'm {name}. What is my name?",
        f"Hello, my name is {name}. What's my name?",
        f"My name is {name}. Do you remember my name?",
    ))
    def score(text: str) -> float:
        t = text.strip()
        if len(t.split()) > 10 or name.lower() not in t.lower():
            return 0.0
        return 0.5 if "my name" in t.lower() else 1.0
    return Task("recall", prompt, f"Your name is {name}.", score)


def make_context_recall(rng: random.Random) -> Task:
    """Retrieve a field from the PREAMBLE — the mechanism gen-3's SFT
    trained away (identity/novel probe: 0.67 pretrain → 0.00 rlvr; 'You
    are Melise' → 'Leo' at serve time). Three variants, one conditional:
    copy the bot name when given, copy the date when given, refuse
    honestly when the date is absent. The preamble rides ON the task
    (Task.preamble) with fields randomized per prompt, so nothing here
    is memorizable."""
    bot = rng.choice(TRAIN_NAMES)
    variant = rng.choices(("name", "date", "no_date"), (2, 2, 1))[0]

    if variant == "name":
        preamble = f"You are {bot}, a tiny language model."
        prompt = rng.choice((
            "What is your name?", "Who are you?", "Tell me your name.",
            "What should I call you?",
        ))
        def score(text: str) -> float:
            t = text.strip()
            if len(t.split()) > 10 or bot.lower() not in t.lower():
                return 0.0
            return 1.0
        return Task("context_recall", prompt, f"I'm {bot}.", score,
                    preamble=preamble)

    date, wd, mo, day = random_date(rng)
    if variant == "date":
        preamble = (f"You are {bot}, a tiny language model. "
                    f"Today is {date}.")
        prompt = rng.choice((
            "What day is today?", "What's the date today?",
            "What day of the week is it?", "Do you know today's date?",
        ))
        def score(text: str) -> float:
            return float(contains_any(text, [wd])
                         or (contains_any(text, [mo])
                             and contains_any(text, [str(day)])))
        return Task("context_recall", prompt, f"Today is {date}.", score,
                    preamble=preamble)

    # no_date: the preamble carries NO date — reward the honest refusal
    # and punish an invented one. ("May" is skipped in the invented-date
    # check: as a modal verb it appears in honest refusals.)
    preamble = f"You are {bot}, a tiny language model."
    prompt = rng.choice((
        "What day is today?", "What's the date?", "What time is it?",
    ))
    invented = [w for w in (*WEEKDAYS, *MONTHS) if w != "May"]
    def score(text: str) -> float:
        t = text.lower()
        if not any(p in t for p in REFUSAL_PATTERNS):
            return 0.0
        return 0.0 if contains_any(text, invented) else 1.0
    return Task(
        "context_recall", prompt,
        "I don't know — I have no clock or calendar, only the text of "
        "this conversation.", score, preamble=preamble)


def make_facts(rng: random.Random) -> Task:
    """A closed-world fact question with an enumerable answer set
    (transformer.facts) — TRAIN split only; the heldout split is the
    probe suite's measuring stick and must never appear in a reward.
    This is the generative-access counterpart of chat_facts.txt SFT:
    gen-3 held weak fact content in the base model (cloze 0.67) that
    chat form couldn't reach (0.00)."""
    facts = load_facts(split="train")
    f = facts[rng.randrange(len(facts))]
    match = prefix_any if f["mode"] == "prefix" else contains_any
    answers, reject = f["answers"], f.get("reject")
    def score(text: str) -> float:
        if not match(text, answers):
            return 0.0
        # Forced-choice slack: echoing both options scores nothing.
        return 0.0 if reject and contains_any(text, reject) else 1.0
    canonical = answers[0].capitalize().rstrip(".") + "."
    return Task("facts", f["chat"], canonical, score)


TASKS: dict[str, Callable[[random.Random], Task]] = {
    "copy": make_copy,
    "arith": make_arith,
    "parity": make_parity,
    "count": make_count_letter,
    "words": make_word_count,
    "recall": make_recall,
    "context_recall": make_context_recall,
    "facts": make_facts,
}


# Training-rollout weights: learning headroom per family. copy/parity/
# words have sat at or near their ceiling since gen-1/2, so their groups
# are mostly zero-variance — no gradient. Weighted sampling spends
# rollouts where reward variance still exists. Eval stays uniform
# (rollout.eval_rewards passes no weights) so per-task numbers keep
# their meaning across runs — though adding families changes the eval
# SET, so aggregate eval reward is not comparable across generations
# (per-task numbers are).
TASK_WEIGHTS = {"copy": 0.5, "parity": 0.75, "words": 0.75,
                "count": 1.0, "recall": 1.0, "arith": 2.0,
                "context_recall": 1.5, "facts": 1.5}


def sample_tasks(names: list[str], n: int, rng: random.Random,
                 weights: dict[str, float] | None = None) -> list[Task]:
    """n tasks from the named kinds — uniform, or weighted if given
    (unknown names default to weight 1.0)."""
    if weights:
        kinds = rng.choices(names, [weights.get(k, 1.0) for k in names], k=n)
        return [TASKS[k](rng) for k in kinds]
    return [TASKS[rng.choice(names)](rng) for _ in range(n)]
