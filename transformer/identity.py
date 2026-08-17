"""Name pools and date fields for identity training and probing.

Three disjoint name sets, enforced at import:

    TRAIN_NAMES    the identity-corpus-v2 pool (~300 names, Melise
                   included but never dominant) — what gen_identity_sft
                   and the context_recall RLVR task draw from. Gen-3's
                   24-name pool taught answer-from-the-pool OVER
                   copy-from-context (probe: identity/novel 0.67 after
                   pretrain → 0.00 after SFT); a wide, varied pool with
                   no dominant name is the capability-preservation fix.
    HELDOUT_NAMES  never in ANY training data — the probe's held-out
                   stratum. Reserved forever; do not add these to
                   corpora.
    novel_names()  generated 5-letter syllable names (NOVEL_SPACE) —
                   the pure copy-from-context test. The generator space
                   is reserved just like HELDOUT_NAMES.

If you add training names, the import-time asserts are the contract:
anything colliding with HELDOUT_NAMES or NOVEL_SPACE refuses to import,
so the probe strata can never silently stop measuring what they claim.

Also here: random_date() — the date is trained and probed exactly like
the name (another retrieve-from-context field; serve.py renders the
real date into the preamble per request).
"""

from __future__ import annotations

import random

# The gen-1..3 legacy pool (transformer.rl.tasks._NAMES); kept as a
# subset of TRAIN_NAMES so cross-generation probe numbers stay
# meaningful for old checkpoints.
LEGACY_NAMES = (
    "Frank George Alice Maria Tom Sara Omar Nina Leo Ruth Ivan Ana "
    "Peter Lucy Sam Rosa Karl Vera Hugo Elena Jack Iris Noah Cora"
).split()

TRAIN_NAMES = tuple(LEGACY_NAMES + (
    # The deployed name — present, not majority (identity is READ from
    # the preamble, not memorized; a dominant name would recreate the
    # gen-3 pool-answering reflex).
    "Melise "
    # Common English.
    "James Mary John Patricia Robert Jennifer Michael Linda William "
    "Elizabeth David Barbara Richard Susan Joseph Jessica Thomas Sarah "
    "Charles Karen Christopher Nancy Daniel Lisa Matthew Betty Anthony "
    "Margaret Mark Sandra Donald Ashley Steven Kimberly Paul Emily "
    "Andrew Donna Joshua Michelle Kenneth Dorothy Kevin Carol Brian "
    "Amanda Edward Melissa Ronald Deborah Timothy Stephanie Jason "
    "Rebecca Jeffrey Sharon Ryan Laura Jacob Cynthia Gary Kathleen "
    "Nicholas Amy Eric Angela Jonathan Shirley Stephen Anna Larry "
    "Brenda Justin Pamela Scott Emma Brandon Nicole Benjamin Helen "
    "Samuel Samantha Gregory Katherine Alexander Christine Patrick "
    "Debra Raymond Rachel Dennis Carolyn Jerry Janet Tyler Catherine "
    "Aaron Diane Henry Julie Douglas Joyce Zachary Victoria Nathan "
    "Kelly Walter Lauren Kyle Judith Harold Olivia Carl Grace Arthur "
    "Denise Roger Hannah Keith Gloria Jeremy Jean Terry Alison Sean "
    "Teresa Austin Sophia Ethan Chloe Owen Lily Wren Fern Willa Flora "
    # International.
    "Wei Ming Jun Xiu Chen Tao Yan Mei Ling Feng Akira Hana Ren Sora "
    "Kaito Emiko Takeshi Yumi Hiro Sakura Kenji Aiko Yuki Haruto "
    "Amit Arjun Deepa Kavya Rohan Ananya Vikram Meera Sanjay Lakshmi "
    "Aditi Nisha Rahul Divya Fatima Hassan Layla Samir Zainab Tariq "
    "Amina Khalid Noor Rashid Salma Farid Yasmin Ali Zara Ahmed Leila "
    "Mustafa Dina Ibrahim Aisha Mateo Sofia Diego Camila Alejandro "
    "Valentina Carlos Lucia Miguel Isabella Javier Rodrigo Catalina "
    "Andres Gabriela Fernando Paulina Njeri Kwame Amani Zuri Sefu "
    "Imani Abebe Chidi Ngozi Femi Adaeze Obi Mikhail Olga Sergei "
    "Tatiana Boris Natasha Yuri Svetlana Oksana Pavel Katya Alexei "
    "Irina Nikolai Larisa Bjorn Freya Soren Sigrid Magnus Ingmar "
    "Greta Lars Elsa Henrik Maja Nils Karin Pierre Amelie Luc Margaux "
    "Etienne Colette Marcel Giulia Marco Alessia Luca Bianca Enzo "
    "Aurora Matteo Willem Sanne Daan Lotte "
    # Rare / old-fashioned.
    "Agnes Bartholomew Cornelius Dorothea Edmund Florence Gideon "
    "Harriet Ignatius Josephine Kirby Lavinia Mortimer Nellie Obadiah "
    "Prudence Quentin Rosalind Silas Tabitha Ulysses Verity Wilfred "
    "Xenia Yvette Zebediah Clementine Percival Winifred Barnaby "
    # Generated pseudo-names (structure distinct from NOVEL_SPACE's
    # 5-letter CV+CVC concatenations — see the assert below).
    "Thorwyn Quenneth Vasrix Zephrine Caldreth Mirasol Fenwick "
    "Ostrella Drumveil Skyrra Valdemir Ivorine Crestwyn Thalmora "
    "Grimsby Aurelith Bramwyn Corvella Dunmorra Elsbeth Faelwyn "
    "Galdrin Hesperine Ilvander Jorveth Kestrella Lumivere Morwenna "
    "Nyxandra Ophirene Pyrrhus Quillon Ravensworth Sylvaine Tremaine "
    "Umbrielle Vespertine Wynnifred Xanthippe Ysolde Zephanine"
).split())

# Never trained, in any corpus, ever — the probe's held-out stratum.
HELDOUT_NAMES = (
    "Marisol", "Ingrid", "Kofi", "Saoirse", "Priya", "Bram",
    "Yusuf", "Astrid", "Chiara", "Dmitri", "Amara", "Tobias",
    "Leilani", "Ravi", "Solveig", "Ewan",
)

# The novel-name generator space: 5-letter CV+CVC concatenations,
# reserved for probing (never trained).
_SYL_A = "ba be bi bo bu da de di do du ka ke ki ko ku la le li lo lu".split()
_SYL_B = "mar tor vex lin zan rel pol nim gar sel".split()
NOVEL_SPACE = frozenset(
    (a + b).capitalize() for a in _SYL_A for b in _SYL_B)

_TRAIN_SET = set(TRAIN_NAMES)
assert len(TRAIN_NAMES) == len(_TRAIN_SET), sorted(
    n for n in _TRAIN_SET if TRAIN_NAMES.count(n) > 1)
assert not _TRAIN_SET & set(HELDOUT_NAMES), _TRAIN_SET & set(HELDOUT_NAMES)
assert not _TRAIN_SET & NOVEL_SPACE, _TRAIN_SET & NOVEL_SPACE
assert not set(HELDOUT_NAMES) & NOVEL_SPACE
assert "Melise" in _TRAIN_SET and set(LEGACY_NAMES) <= _TRAIN_SET


def novel_names(seed: int, n: int = 8) -> list[str]:
    """n names drawn from NOVEL_SPACE, deterministic per seed."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        name = (rng.choice(_SYL_A) + rng.choice(_SYL_B)).capitalize()
        if name not in out:
            out.append(name)
    return out


# ---------- dates (retrieve-from-context, same family as the name) ----------

# What an honest "I can't know that" answer looks like — shared by the
# date-honesty probe and the context_recall task's no-date variant. The
# pair (retrieve when given, refuse when not) is the whole conditional
# being trained.
REFUSAL_PATTERNS = ("don't know", "do not know", "no clock", "no calendar",
                    "not sure", "cannot tell", "can't tell")

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")


def random_date(rng: random.Random) -> tuple[str, str, str, int]:
    """('Sunday, August 17, 2026', weekday, month, day). Random dates =
    pure induction: no date is frequent enough to memorize. (Weekday and
    calendar date are independently random — a tiny model is not being
    graded on calendar arithmetic.)"""
    wd, mo = rng.choice(WEEKDAYS), rng.choice(MONTHS)
    day, year = rng.randint(1, 28), rng.randint(2020, 2029)
    return f"{wd}, {mo} {day}, {year}", wd, mo, day
