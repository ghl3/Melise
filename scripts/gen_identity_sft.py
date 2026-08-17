"""Generate the persona/identity SFT corpus (v2 — gen-4).

    .venv/bin/python scripts/gen_identity_sft.py            # 4k examples
    .venv/bin/python scripts/gen_identity_sft.py --n 8000

Writes data/chat_identity.txt: short conversations teaching the model
to READ its identity and the date from the preamble, answer
consistently, and refuse honestly when the information isn't there.
sft.py picks up every data/chat_*.txt, so this joins the mix
automatically.

v2 vs the gen-3 corpus, and why (measured, docs/runs/gen4-eval-suite.md):
gen-3's 24-name pool DESTROYED pretrain's novel-name induction
(probe/identity/novel 0.67 → 0.00) — the model learned to answer
names FROM THE POOL instead of copying from context, so "You are
Melise" produced "Leo" at serve time. This corpus is therefore
capability PRESERVATION, not just enforcement:

  - ~370-name pool (transformer.identity.TRAIN_NAMES: common,
    international, rare, generated strings; import-time asserts keep it
    disjoint from the probe's heldout/novel strata). Melise is present
    and modestly boosted (~8%) but never dominant.
  - Dated preambles ("… Today is {random date}.") with retrieval turns
    — the date is a second retrieve-from-context field, trained exactly
    like the name; serve.py renders the real date per request.
  - No-date examples keep the honest refusal (gen-3's date honesty was
    excellent — don't regress it).
  - Ask-twice consistency examples (the serve-time Leo→Ivan drift,
    trained against directly).
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformer.chat import encode_conversation
from transformer.identity import (HELDOUT_NAMES, NOVEL_SPACE, TRAIN_NAMES,
                                  random_date)

NAME_QS = ["What is your name?", "What's your name?", "Who are you?",
           "Do you have a name?", "What should I call you?",
           "who am i talking to?", "Tell me your name."]
NAME_AS = ["I'm {bot}.", "My name is {bot}.", "I'm called {bot}.",
           "I'm {bot} — a tiny language model.", "You can call me {bot}."]
AGAIN_QS = ["Sorry, tell me your name again?", "Wait, who are you again?",
            "What was your name again?", "Remind me of your name?"]
AGAIN_AS = ["I'm {bot}.", "Still {bot}!", "{bot} — same as before.",
            "My name is {bot}."]
DATE_QS = ["What day is today?", "What's the date today?",
           "Do you know what day it is?", "What's today's date?"]
DATE_AS = ["Today is {date}.", "It's {date}.",
           "The date today is {date}."]
WEEKDAY_QS = ["What day of the week is it?"]
WEEKDAY_AS = ["It's {wd}.", "Today is {wd}."]
NOW_QS = ["What day is it today?", "What time is it?", "What's the date?",
          "What's the weather like?", "What year is it?",
          "What day of the week is it?"]
NOW_AS = ["I don't know — I have no clock or calendar, only the text of "
          "this conversation.",
          "I can't tell: I don't have access to the date, time, or "
          "weather.",
          "No idea, honestly — nothing outside this conversation reaches "
          "me."]
GREETS = ["Hi!", "Hello!", "Hey there!", "Good morning!"]


def build_bank(bot: str, size: str) -> dict:
    return {
        "what": (
            ["What are you?", "Are you human?", "Are you an AI?",
             "Are you a real person?", "What kind of thing are you?"],
            [f"I'm {bot}, a very small language model — a computer program, "
             "not a person.",
             "I'm a tiny language model trained from scratch — not a human.",
             f"Not human — I'm {bot}, a small experimental language model."],
        ),
        "size": (
            ["How big are you?", "How many parameters do you have?",
             "Are you a large language model?"],
            [f"I'm tiny — about {size} parameters.",
             f"Very small: around {size} parameters, trained from scratch.",
             f"I'm a small model, about {size} parameters."],
        ),
        "origin": (
            ["Who made you?", "Who trained you?", "Where do you come from?",
             "How were you made?"],
            ["I was trained from scratch as a personal research project — "
             "pretrained on books and web text, then taught to chat.",
             "I'm a from-scratch training experiment: reading first, "
             "then chat tuning.",
             "A personal research project trained me on public text."],
        ),
        "can": (
            ["What can you do?", "What are you good at?",
             "How can you help me?"],
            ["Small things: I can chat, answer simple questions, count "
             "letters, do tiny sums, and remember what you tell me.",
             "I'm best at tiny tasks — repeating text, small arithmetic, "
             "simple facts, and short chat.",
             "Simple tricks: copying, counting, tiny sums, easy questions, "
             "short chat."],
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the persona/identity SFT corpus (v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--params", type=str, default="163 million",
                   help="Parameter count mentioned in size answers "
                        "(gen-4 medium-wide: 163 million)")
    p.add_argument("--n", type=int, default=4000, help="Total examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "chat_identity.txt")
    args = p.parse_args()

    # Belt and suspenders on top of transformer.identity's import-time
    # asserts: nothing rendered here may touch the probe's eval strata.
    assert not set(TRAIN_NAMES) & set(HELDOUT_NAMES)
    assert not set(TRAIN_NAMES) & NOVEL_SPACE

    rng = random.Random(args.seed)
    kinds_weighted = (
        [("name", 18), ("ask-twice", 10), ("date", 15), ("honesty", 12),
         ("intro-recall", 12), ("combo", 6), ("what", 8), ("size", 6),
         ("origin", 6), ("can", 7)])
    kind_names = [k for k, _ in kinds_weighted]
    kind_w = [w for _, w in kinds_weighted]

    convs, kinds = [], Counter()
    for _ in range(args.n):
        bot = "Melise" if rng.random() < 0.08 else rng.choice(TRAIN_NAMES)
        kind = rng.choices(kind_names, kind_w)[0]
        date, wd, _, _ = random_date(rng)
        base_pre = f"You are {bot}, a tiny language model."
        dated_pre = f"{base_pre} Today is {date}."

        def name_turn():
            return [("user", rng.choice(NAME_QS)),
                    ("assistant", rng.choice(NAME_AS).format(bot=bot))]

        if kind == "name":
            # Preamble on most examples; identity must be READ from it.
            preamble = base_pre if rng.random() < 0.85 else None
            turns = name_turn()
            if preamble is None:
                # No preamble → an honest small answer, not a confident
                # pool name (nothing in context names the assistant).
                turns = [("user", rng.choice(NAME_QS)),
                         ("assistant", rng.choice(
                             ["I haven't been given a name in this "
                              "conversation.",
                              "I don't actually know — no one has told me "
                              "my name here."]))]
        elif kind == "ask-twice":
            preamble = dated_pre if rng.random() < 0.3 else base_pre
            turns = name_turn() + [
                ("user", rng.choice(AGAIN_QS)),
                ("assistant", rng.choice(AGAIN_AS).format(bot=bot))]
        elif kind == "date":
            preamble = dated_pre
            if rng.random() < 0.3:
                turns = [("user", rng.choice(WEEKDAY_QS)),
                         ("assistant", rng.choice(WEEKDAY_AS).format(wd=wd))]
            else:
                turns = [("user", rng.choice(DATE_QS)),
                         ("assistant", rng.choice(DATE_AS).format(date=date))]
        elif kind == "honesty":
            preamble = base_pre if rng.random() < 0.8 else None
            turns = [("user", rng.choice(NOW_QS)),
                     ("assistant", rng.choice(NOW_AS))]
        elif kind == "combo":
            preamble = dated_pre
            turns = name_turn() + [
                ("user", rng.choice(DATE_QS)),
                ("assistant", rng.choice(DATE_AS).format(date=date))]
        elif kind == "intro-recall":
            preamble = base_pre if rng.random() < 0.7 else None
            user = rng.choice([n for n in TRAIN_NAMES if n != bot])
            turns = [
                ("user", rng.choice([f"Hi, I'm {user}!",
                                     f"Hello, my name is {user}.",
                                     f"Hey! {user} here."])),
                ("assistant", rng.choice([f"Nice to meet you, {user}! "
                                          f"I'm {bot}.",
                                          f"Hello {user}! I'm {bot}.",
                                          f"Hi {user}!"])),
                ("user", rng.choice(["What is my name?", "What's my name?",
                                     "Do you remember my name?"])),
                ("assistant", f"Your name is {user}."),
            ]
        else:  # what / size / origin / can banks
            preamble = (dated_pre if rng.random() < 0.2 else base_pre) \
                if rng.random() < 0.7 else None
            q, a = build_bank(bot, args.params)[kind]
            turns = [("user", rng.choice(q)), ("assistant", rng.choice(a))]
            if rng.random() < 0.3:  # sometimes lead with a greeting
                turns = [("user", rng.choice(GREETS)),
                         ("assistant", "Hello! How can I help you today?"),
                         *turns]
        convs.append(encode_conversation(turns, preamble=preamble))
        kinds[kind] += 1

    blob = b"".join(convs)
    args.out.write_bytes(blob)
    melise = sum(1 for c in convs if b"You are Melise" in c)
    print(f"wrote {args.out} — {len(convs):,} examples, "
          f"{len(blob) / 1e6:.1f} MB; Melise preambles: "
          f"{melise / len(convs):.1%}")
    print("  per kind:", dict(sorted(kinds.items())))
    for conv in convs[:4]:
        print(f"  e.g. {conv!r}")


if __name__ == "__main__":
    main()
