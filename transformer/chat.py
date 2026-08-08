"""Byte-level chat template (used by SFT, RL rollouts, and data prep).

Byte models have no tokenizer and no spare vocab slots (all 256 byte
values are real data), so role markers are ASCII control bytes that are
verified absent from every corpus in data/ (checked 2026-08-08):

    0x01  start of a user turn
    0x02  start of an assistant turn
    0x03  end of turn (the model learns to emit this — it's the stop byte)
    0x04  end of conversation (separates conversations in a corpus file)

A two-turn conversation is laid out as:

    \\x01 <user bytes> \\x03 \\x02 <assistant bytes> \\x03 \\x04

At inference, the prompt is everything up to and including \\x02; decoding
stops at the first \\x03. SFT masks the loss to assistant spans: the
bytes after each \\x02 up to and including its \\x03, so the model is
trained to answer and to stop, but not to imitate user text.
"""

from __future__ import annotations

import re

USER = 0x01
ASSISTANT = 0x02
END_TURN = 0x03
END_CONV = 0x04

# Strip every C0 control byte except \t \n \r from dataset text so the
# markers above stay unambiguous (and stray terminal-control junk dies).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize(text: str) -> str:
    return _CONTROL.sub("", text)


def encode_conversation(turns: list[tuple[str, str]]) -> bytes:
    """turns: [(role, text), ...] with role 'user' or 'assistant'."""
    out = bytearray()
    for role, text in turns:
        out.append(USER if role == "user" else ASSISTANT)
        out += sanitize(text).strip().encode("utf-8")
        out.append(END_TURN)
    out.append(END_CONV)
    return bytes(out)


def make_prompt(user_text: str) -> bytes:
    """Prompt bytes for generation: user turn + assistant marker."""
    return bytes([USER]) + sanitize(user_text).strip().encode("utf-8") + bytes(
        [END_TURN, ASSISTANT]
    )


def split_conversations(blob: bytes) -> list[bytes]:
    """Split a corpus file back into per-conversation byte strings
    (END_CONV terminator re-appended to each non-empty piece)."""
    return [c + bytes([END_CONV]) for c in blob.split(bytes([END_CONV])) if c]


def assistant_mask(conv: bytes) -> list[bool]:
    """Per-byte loss mask for a conversation laid out by
    encode_conversation. True marks bytes an SFT loss should train on as
    *targets*: the content of each assistant turn plus its END_TURN. The
    0x02 marker itself is False — it's given, not predicted; END_CONV is
    True so the model also learns the conversation boundary."""
    mask = [False] * len(conv)
    in_assistant = False
    for i, b in enumerate(conv):
        if b == ASSISTANT:
            in_assistant = True
        elif b == END_TURN:
            if in_assistant:
                mask[i] = True
            in_assistant = False
        elif b == USER:
            in_assistant = False
        elif b == END_CONV:
            mask[i] = True
        elif in_assistant:
            mask[i] = True
    return mask


def parse_turns(conv: bytes) -> list[tuple[str, bytes]]:
    """Inverse of encode_conversation: [(role, content bytes), ...].
    Stops at END_CONV; bytes outside a marker...END_TURN span are ignored."""
    turns = []
    role, buf = None, bytearray()
    for b in conv:
        if b in (USER, ASSISTANT):
            role, buf = ("user" if b == USER else "assistant"), bytearray()
        elif b == END_TURN:
            if role is not None:
                turns.append((role, bytes(buf)))
            role = None
        elif b == END_CONV:
            break
        elif role is not None:
            buf.append(b)
    return turns


def assemble(turns: list[tuple[str, bytes]]) -> bytes:
    """encode_conversation for content that is already bytes (no
    sanitization — used to reassemble parsed turns)."""
    out = bytearray()
    for role, content in turns:
        out.append(USER if role == "user" else ASSISTANT)
        out += content
        out.append(END_TURN)
    out.append(END_CONV)
    return bytes(out)


def completion_text(raw: bytes) -> str:
    """Decode a generated assistant completion: cut at the stop byte,
    drop any stray markers, decode utf-8 leniently."""
    cut = raw.split(bytes([END_TURN]))[0]
    cut = bytes(b for b in cut if b not in (USER, ASSISTANT, END_CONV))
    return cut.decode("utf-8", errors="replace").strip()
