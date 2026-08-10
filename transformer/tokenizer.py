"""Tokenizers: raw bytes (the default) and trained BPE, one interface.

Every training/inference path speaks token ids; which ids depends on the
model config's `tokenizer` field ("bytes", or the name/path of a trained
artifact like "tokenizer-bpe4k"). Checkpoints embed the config, so a
loaded model always knows its own encoding.

Both tokenizers expose the same contract:

    vocab_size            256 for bytes, trained size for BPE
    encode(text) -> ids   text may be str or utf-8 bytes; NEVER emits
                          special ids for content (sanitize() strips the
                          control chars that spell them)
    decode(ids) -> str    lenient utf-8; skips special ids
    pad_id, user_id, assistant_id, end_turn_id, end_conv_id
                          ids 0-4 in BOTH tokenizers — for bytes these
                          are literally the control bytes of the chat
                          template (transformer.chat), and the BPE
                          trainer pins the same control characters to
                          the same ids (scripts/train_tokenizer.py)

Because the special ids and their on-disk byte forms coincide, chat
corpora stored in the byte template encode correctly under either
tokenizer with no re-parsing.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SPECIAL_IDS = (0, 1, 2, 3, 4)  # pad, user, assistant, end_turn, end_conv

# encode() strips the specials' own characters from content so no input
# text can ever produce a special id — structure comes only from code
# that injects the ids explicitly (transformer.chat).
_SPECIAL_CHARS = re.compile(r"[\x00-\x04]")


class ByteTokenizer:
    """The identity tokenizer: token ids are utf-8 byte values."""

    name = "bytes"
    vocab_size = 256
    pad_id, user_id, assistant_id, end_turn_id, end_conv_id = _SPECIAL_IDS

    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return list(_SPECIAL_CHARS.sub("", text).encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i not in _SPECIAL_IDS and 0 <= i < 256) \
            .decode("utf-8", errors="replace")

    def byte_lengths(self) -> list[int]:
        """Raw byte length per token id — the bpb denominators. Every
        byte token is exactly one byte."""
        return [1] * 256


class BPETokenizer:
    """A trained byte-level BPE artifact (scripts/train_tokenizer.py)."""

    pad_id, user_id, assistant_id, end_turn_id, end_conv_id = _SPECIAL_IDS

    def __init__(self, path: Path):
        from tokenizers import Tokenizer  # runtime dep only for BPE models

        self._tok = Tokenizer.from_file(str(path))
        self.name = path.stem.replace("tokenizer-", "")
        self.vocab_size = self._tok.get_vocab_size()
        for i, ch in enumerate("\x00\x01\x02\x03\x04"):
            got = self._tok.token_to_id(ch)
            assert got == i, f"special {ch!r} has id {got}, expected {i}"

    def encode(self, text: str | bytes) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return self._tok.encode(_SPECIAL_CHARS.sub("", text)).ids

    def decode(self, ids) -> str:
        return self._tok.decode(
            [i for i in ids if i not in _SPECIAL_IDS], skip_special_tokens=True
        )

    def byte_lengths(self) -> list[int]:
        """Raw byte length per token id. ByteLevel vocab strings use one
        character per underlying byte, so the char count IS the byte
        count — this makes logged bpb exact, not estimated."""
        return [max(1, len(self._tok.id_to_token(i) or ""))
                for i in range(self.vocab_size)]


def load_tokenizer(name: str | None):
    """Resolve a config's `tokenizer` field. "bytes"/None -> the identity
    byte tokenizer; otherwise a name (resolved in configs/) or path of a
    trained artifact."""
    if name in (None, "bytes"):
        return ByteTokenizer()
    path = Path(name)
    if not path.exists():
        path = PROJECT_ROOT / "configs" / f"{name}.json"
    if not path.exists():
        path = PROJECT_ROOT / "configs" / f"tokenizer-{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no tokenizer artifact for {name!r}")
    return BPETokenizer(path)
