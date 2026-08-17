"""Fetch + format the dialogue corpora for the pretrain mix.

    .venv/bin/python scripts/prep_dialogue_data.py

Writes:
    data/dialogue_movies.txt   Cornell Movie-Dialogs (~220k movie-script
                               exchanges; iso-8859-1 source)
    data/dialogue_daily.txt    DailyDialog (~13k everyday conversations;
                               tokenized source, lightly detokenized)
    data/dialogue_persona.txt  PersonaChat / ConvAI2 (~18k crowdworker
                               get-to-know-you chats; gen-4 addition)
    data/dialogue_blended.txt  BlendedSkillTalk (~7k chats blending
                               personality/knowledge/empathy; gen-4)

All render as dash-prefixed turns with a blank line between
conversations — plain conversational text, NOT the chat template (these
are pretrain corpora; the `dialogue_` prefix keeps them excluded from
sft.py's chat_* glob). This is the casual-dialogue register the gen-2
mix had none of — the product register, and the gen-3 domain that
improved all run without saturating, which is why gen-4 grows it.
"""

import ast
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hf_util import fetch
from transformer.chat import sanitize

CACHE = PROJECT_ROOT / "data" / ".cache"
CORNELL_URL = ("https://www.cs.cornell.edu/~cristian/data/"
               "cornell_movie_dialogs_corpus.zip")
# The original ijcnlp zip (yanran.li) now sits behind a bot check;
# roskoN/dailydialog mirrors the original per-split files on HF.
DAILY_REPO = "roskoN/dailydialog"
DAILY_ZIPS = ("train.zip", "validation.zip", "test.zip")
HF = "https://huggingface.co"
SEP = " +++$+++ "

# DailyDialog text is whitespace-tokenized ("how about it ? great !").
_DETOK_BEFORE = re.compile(r"\s+([,.!?;:%)\]’”…])")
_DETOK_AFTER = re.compile(r"([(\[‘“$])\s+")
_DETOK_APOS = re.compile(r"\s+(['’](?:s|t|re|ve|ll|d|m)\b)")


def detok(text: str) -> str:
    text = _DETOK_APOS.sub(r"\1", text)
    text = _DETOK_BEFORE.sub(r"\1", text)
    return _DETOK_AFTER.sub(r"\1", text).strip()


def clean_turn(text: str) -> str:
    return sanitize(text).replace("\n", " ").strip()


def write_dialogues(out: Path, dialogues) -> int:
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for turns in dialogues:
            turns = [t for t in (clean_turn(t) for t in turns) if t]
            if len(turns) < 2:
                continue
            f.write("\n".join(f"- {t}" for t in turns) + "\n\n")
            n += 1
    print(f"wrote {out} — {n:,} conversations, "
          f"{out.stat().st_size / 1e6:.1f} MB")
    return n


def build_cornell(out: Path) -> None:
    raw = fetch(CORNELL_URL, CACHE / "cornell_movie_dialogs.zip")
    with zipfile.ZipFile(raw) as z:
        base = "cornell movie-dialogs corpus/"
        lines_raw = z.read(base + "movie_lines.txt").decode("iso-8859-1")
        convs_raw = z.read(base + "movie_conversations.txt").decode("iso-8859-1")
    lines = {}
    for row in lines_raw.splitlines():
        parts = row.split(SEP)
        if len(parts) >= 5:
            lines[parts[0]] = parts[4]
    def dialogues():
        for row in convs_raw.splitlines():
            parts = row.split(SEP)
            if len(parts) < 4:
                continue
            try:
                ids = ast.literal_eval(parts[3])
            except (ValueError, SyntaxError):
                continue
            yield [lines[i] for i in ids if i in lines]
    write_dialogues(out, dialogues())


def build_daily(out: Path) -> None:
    lines = []
    for name in DAILY_ZIPS:
        raw = fetch(f"{HF}/datasets/{DAILY_REPO}/resolve/main/{name}",
                    CACHE / f"dailydialog_{name}")
        with zipfile.ZipFile(raw) as z:
            inner = next(n for n in z.namelist()
                         if Path(n).name.startswith("dialogues_")
                         and n.endswith(".txt")
                         and "act" not in n and "emotion" not in n)
            text = z.read(inner).decode("utf-8", errors="replace")
        lines += [ln for ln in text.splitlines() if ln.strip()]
    dialogues = ([detok(u) for u in line.split("__eou__") if u.strip()]
                 for line in lines)
    write_dialogues(out, dialogues)


def build_persona(out: Path) -> None:
    """PersonaChat via the ParlAI S3 mirror (ConvAI2 line format:
    '<num> <utterance>\\t<reply>\\t...'; num resets to 1 per dialogue).
    The none_original files carry no persona lines — pure dialogue."""
    raw = fetch("http://parl.ai/downloads/personachat/personachat.tgz",
                CACHE / "personachat.tgz")
    dialogues, cur = [], []
    with tarfile.open(raw) as t:
        for name in ("personachat/train_none_original.txt",
                     "personachat/valid_none_original.txt",
                     "personachat/test_none_original.txt"):
            text = t.extractfile(name).read().decode("utf-8")
            for line in text.splitlines():
                num, _, rest = line.partition(" ")
                if not num.isdigit():
                    continue
                if int(num) == 1 and cur:
                    dialogues.append(cur)
                    cur = []
                parts = rest.split("\t")
                if len(parts) >= 2:
                    # Same whitespace-tokenized source style as
                    # DailyDialog ("hi , how are you ?") — detokenize.
                    cur += [detok(parts[0]), detok(parts[1])]
            if cur:
                dialogues.append(cur)
                cur = []
    write_dialogues(out, dialogues)


def build_blended(out: Path) -> None:
    """BlendedSkillTalk via the ParlAI S3 mirror (json episodes; the two
    seed utterances precede the 'dialog' turn list)."""
    raw = fetch("http://parl.ai/downloads/blended_skill_talk/"
                "blended_skill_talk.tar.gz",
                CACHE / "blended_skill_talk.tar.gz")
    dialogues = []
    with tarfile.open(raw) as t:
        for member in t.getmembers():
            if not member.name.endswith((".json",)):
                continue
            episodes = json.load(t.extractfile(member))
            if not isinstance(episodes, list):
                continue
            for ep in episodes:
                turns = []
                for key in ("free_turker_utterance",
                            "guided_turker_utterance"):
                    if ep.get(key):
                        turns.append(ep[key])
                for item in ep.get("dialog", []):
                    turns.append(item[1] if isinstance(item, (list, tuple))
                                 else str(item))
                if turns:
                    dialogues.append(turns)
    write_dialogues(out, dialogues)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    data = PROJECT_ROOT / "data"
    build_cornell(data / "dialogue_movies.txt")
    build_daily(data / "dialogue_daily.txt")
    build_persona(data / "dialogue_persona.txt")
    build_blended(data / "dialogue_blended.txt")


if __name__ == "__main__":
    main()
