"""Generate the persona/identity SFT corpus.

    .venv/bin/python scripts/gen_identity_sft.py                # 3k examples
    .venv/bin/python scripts/gen_identity_sft.py --name Wren    # future renames

Writes data/chat_identity.txt: short conversations teaching the model
its name, what it is, and honest no-real-time-info answers (date/time/
weather). Gen-2 had none of this — "What is your name?" degenerated
into a repetition loop, and "What day is it?" produced circular
nonsense, because the questions sit in a distribution SFT never
covered. sft.py picks up every data/chat_*.txt, so this joins the mix
automatically.

The assistant's name is a CLI flag, not a constant: the site has been
renamed four times this week. Regenerate + re-upload on the next one.

Includes multi-turn examples that pair an introduction with a
name-recall question ("Your name is X.") — the same skill the `recall`
RL task scores, so SFT plants the exact format RL amplifies.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformer.chat import encode_conversation
from transformer.rl.tasks import _NAMES

def build_bank(bot: str) -> dict[str, tuple[list[str], list[str]]]:
    """category -> (question templates, answer templates)."""
    return {
        "name": (
            ["What is your name?", "What's your name?", "Who are you?",
             "Do you have a name?", "What should I call you?",
             "who am i talking to?", "Tell me your name."],
            [f"I'm {bot}.", f"My name is {bot}.", f"I'm called {bot}.",
             f"I'm {bot} — a tiny language model.",
             f"You can call me {bot}."],
        ),
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
            ["I'm tiny — about 19 million parameters.",
             "Very small: around 19 million parameters, trained from scratch.",
             "I'm a small model, about 19 million parameters."],
        ),
        "origin": (
            ["Who made you?", "Who trained you?", "Where do you come from?",
             "How were you made?"],
            ["I was trained from scratch as a personal research project — "
             "pretrained on classic books, then taught to chat.",
             "I'm a from-scratch training experiment: books first, "
             "then chat tuning.",
             "A personal research project trained me on public-domain text."],
        ),
        "now": (
            ["What day is it today?", "What time is it?",
             "What's the date?", "What's the weather like?",
             "What year is it?", "What day of the week is it?"],
            ["I don't know — I have no clock or calendar, only the text of "
             "this conversation.",
             "I can't tell: I don't have access to the date, time, or "
             "weather.",
             "No idea, honestly — nothing outside this conversation reaches "
             "me."],
        ),
        "can": (
            ["What can you do?", "What are you good at?",
             "How can you help me?"],
            ["Small things: I can copy phrases, answer even-or-odd "
             "questions, count letters, and chat a little.",
             "I'm best at tiny tasks — repeating text, small arithmetic, "
             "counting letters in words.",
             "Simple tricks: copying, counting, tiny sums, short chat."],
        ),
    }

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the persona/identity SFT corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", type=str, default="Lily",
                   help="Assistant persona name")
    p.add_argument("--n", type=int, default=3000, help="Total examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "chat_identity.txt")
    args = p.parse_args()

    rng = random.Random(args.seed)
    bank = build_bank(args.name)
    greets = ["Hi!", "Hello!", "Hey there!", "Good morning!"]
    convs, kinds = [], Counter()
    for _ in range(args.n):
        kind = rng.choice(sorted(bank) + ["intro-recall"])
        if kind == "intro-recall":
            user = rng.choice(_NAMES)
            turns = [
                ("user", rng.choice([f"Hi, I'm {user}!",
                                     f"Hello, my name is {user}.",
                                     f"Hey! {user} here."])),
                ("assistant", rng.choice([f"Nice to meet you, {user}! "
                                          f"I'm {args.name}.",
                                          f"Hello {user}! I'm {args.name}.",
                                          f"Hi {user}!"])),
                ("user", rng.choice(["What is my name?", "What's my name?",
                                     "Do you remember my name?"])),
                ("assistant", f"Your name is {user}."),
            ]
        else:
            q, a = bank[kind]
            turns = [("user", rng.choice(q)), ("assistant", rng.choice(a))]
            if rng.random() < 0.3:  # sometimes lead with a greeting exchange
                turns = [("user", rng.choice(greets)),
                         ("assistant", "Hello! How can I help you today?"),
                         *turns]
        convs.append(encode_conversation(turns))
        kinds[kind] += 1

    blob = b"".join(convs)
    args.out.write_bytes(blob)
    print(f"wrote {args.out} — {len(convs):,} examples, {len(blob) / 1e6:.1f} MB")
    print("  per kind:", dict(sorted(kinds.items())))
    for conv in convs[:4]:
        print(f"  e.g. {conv!r}")

if __name__ == "__main__":
    main()
