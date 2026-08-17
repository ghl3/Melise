"""Generate the basic-facts SFT corpus from the facts table.

    .venv/bin/python scripts/gen_fact_sft.py            # 4k examples
    .venv/bin/python scripts/gen_fact_sft.py --n 8000

Writes data/chat_facts.txt: short QA conversations rendered from the
TRAIN split of configs/facts.json (transformer.facts) — the heldout
split is never rendered, in any form; probe/facts/*/heldout measures
whether the model learned facts from the mix vs memorized these QA
pairs. sft.py picks up every data/chat_*.txt, so this joins the mix
automatically; the facts RLVR task (transformer.rl.tasks.make_facts)
scores the same table, so SFT plants exactly the format GRPO amplifies.

Every rendered answer is self-checked against the entry's own scorer
(match mode + reject list), so the SFT target and the reward can never
drift apart — the gen_task_sft contract, applied to facts.

Gen-3 context: the base model held weak fact content (cloze 0.67 on
instances) that chat form couldn't access (0.00; "Name an animal" →
"An animal"). This corpus is the access path; the capacity bet
(probe/facts/*/heldout off the floor) is the gen-4 headline metric.
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformer.chat import encode_conversation
from transformer.facts import contains_any, load_facts, prefix_any
from transformer.identity import TRAIN_NAMES

Q_WRAP = ["{q}", "{q}", "{q}", "Quick question: {q}", "Tell me: {q}",
          "I have a question. {q}"]

YES_TEMPLATES = ["Yes.", "Yes!", "Yes, it is.", "Yes, that's right."]
NO_TEMPLATES = ["No.", "No, it isn't.", "No, it's not."]


def _cap(ans: str) -> str:
    return " ".join(w.capitalize() for w in ans.split())


def render_answer(fact: dict, rng: random.Random) -> str:
    if fact["mode"] == "prefix":
        pool = YES_TEMPLATES if fact["answers"][0] == "yes" else NO_TEMPLATES
        return rng.choice(pool)
    # Prefer the leading answers (canonical); tail entries are accepted
    # alternates, not always natural things to SAY.
    ans = _cap(rng.choice(fact["answers"][:3]))
    return rng.choice([f"{ans}.", f"{ans}.", f"The answer is {ans}.",
                       f"{ans}!"])


def check(fact: dict, answer: str) -> bool:
    match = prefix_any if fact["mode"] == "prefix" else contains_any
    if not match(answer, fact["answers"]):
        return False
    return not (fact.get("reject") and contains_any(answer, fact["reject"]))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the basic-facts SFT corpus (train split only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=4000, help="Total examples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "chat_facts.txt")
    args = p.parse_args()

    facts = load_facts(split="train")
    rng = random.Random(args.seed)
    convs, families = [], Counter()
    for _ in range(args.n):
        turns = []
        # 20% of examples ask two facts in one conversation — multi-turn
        # retrieval practice, cheap.
        for fact in rng.sample(facts, 2 if rng.random() < 0.2 else 1):
            answer = render_answer(fact, rng)
            assert check(fact, answer), (fact["id"], answer)
            turns += [("user", rng.choice(Q_WRAP).format(q=fact["chat"])),
                      ("assistant", answer)]
            families[fact["family"]] += 1
        preamble = (f"You are {rng.choice(TRAIN_NAMES)}, a tiny language "
                    "model." if rng.random() < 0.5 else None)
        convs.append(encode_conversation(turns, preamble=preamble))

    blob = b"".join(convs)
    args.out.write_bytes(blob)
    print(f"wrote {args.out} — {len(convs):,} examples, "
          f"{len(blob) / 1e6:.1f} MB (train split: {len(facts)} facts)")
    print("  per family:", dict(sorted(families.items())))
    for conv in convs[:3]:
        print(f"  e.g. {conv!r}")


if __name__ == "__main__":
    main()
