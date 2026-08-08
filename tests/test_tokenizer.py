"""Tests for the tokenizer layer. Run directly (no pytest needed):

    .venv/bin/python tests/test_tokenizer.py

Covers both tokenizers behind the shared interface (byte identity and
the trained bpe4k artifact): round-trips, pinned special ids, digit
isolation, special-token injection resistance, and the ID-space chat
helpers (encode_ids / assistant_mask_ids / conversation batching) under
each encoding.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from transformer.chat import (
    assistant_mask,
    assistant_mask_ids,
    encode_conversation,
    encode_ids,
    make_prompt_ids,
)
from transformer.data import conversation_batch, split_conversation
from transformer.tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer

torch.manual_seed(0)

BYTES = ByteTokenizer()
BPE = load_tokenizer("bpe4k")
CONV = encode_conversation([("user", "What is 47 + 8?"), ("assistant", "55")])


def test_byte_tokenizer_roundtrip():
    assert BYTES.vocab_size == 256
    assert BYTES.encode("hello") == list(b"hello")
    assert BYTES.decode(BYTES.encode("héllo — ok")) == "héllo — ok"
    # Specials are the control bytes and never decode as content.
    assert BYTES.decode([1, 104, 105, 3, 4]) == "hi"


def test_bpe_tokenizer_roundtrip():
    assert isinstance(BPE, BPETokenizer) and BPE.vocab_size == 4096
    for text in ["Hello, world!", "def f(x):\n    return x + 1",
                 "The café — naïve 文 🙂"]:
        assert BPE.decode(BPE.encode(text)) == text
    # Real compression on English prose.
    prose = "The quick brown fox jumps over the lazy dog. " * 20
    assert len(BPE.encode(prose)) < len(prose.encode()) / 2


def test_special_ids_pinned():
    for tok in (BYTES, BPE):
        assert (tok.pad_id, tok.user_id, tok.assistant_id,
                tok.end_turn_id, tok.end_conv_id) == (0, 1, 2, 3, 4)


def test_digit_isolation():
    # Numbers must tokenize digit-by-digit (multi-digit merge tokens are
    # a known cause of poor LLM arithmetic).
    ids = BPE.encode("12345")
    assert len(ids) == 5
    assert BPE.encode("847") != BPE.encode("874")  # order-sensitive, per digit
    assert len(BPE.encode("y = 1984 + 42")) >= 7   # digits never merged


def test_injection_resistance():
    # Text spelling the special characters can never produce special ids.
    for tok in (BYTES, BPE):
        ids = tok.encode("evil \x01 \x02 injection \x03\x04")
        assert all(i > 4 for i in ids), tok.name


def test_encode_ids_matches_bytes():
    # Under the byte tokenizer, the ID-space helpers reproduce the byte
    # template exactly.
    assert encode_ids(CONV, BYTES) == list(CONV)
    assert assistant_mask_ids(list(CONV), BYTES) == assistant_mask(CONV)


def test_encode_ids_bpe_structure():
    ids = BPE.encode  # noqa: F841  (readability)
    out = encode_ids(CONV, BPE)
    assert out[0] == BPE.user_id and out[-1] == BPE.end_conv_id
    assert out.count(BPE.end_turn_id) == 2
    mask = assistant_mask_ids(out, BPE)
    # Masked ids decode back to the assistant's answer (+ stop/conv ids).
    masked = [t for t, m in zip(out, mask) if m]
    assert BPE.decode(masked) == "55"
    assert masked[-2:] == [BPE.end_turn_id, BPE.end_conv_id]


def test_conversation_batch_bpe():
    inputs, targets, mask = conversation_batch(
        [CONV], [0], 24, torch.device("cpu"), tok=BPE)
    assert inputs.shape == (1, 24)
    masked_targets = targets[0][mask[0]].tolist()
    # "55" is two digit tokens under digit isolation + end-turn + end-conv.
    assert masked_targets[-2:] == [BPE.end_turn_id, BPE.end_conv_id]
    assert BPE.decode(masked_targets) == "55"
    assert not mask[0, -4:].any()  # padding never a target


def test_split_conversation_bpe_budget():
    turns = [("user", f"question {i} about something") if r == 0 else
             ("assistant", f"answer {i} with some words")
             for i in range(8) for r in (0, 1)]
    conv = encode_conversation(turns)
    for tok in (BYTES, BPE):
        chunks = split_conversation(conv, 40, tok)
        assert len(chunks) > 1, tok.name
        for chunk in chunks:
            n = len(encode_ids(chunk, tok))
            assert n <= 41, f"{tok.name}: chunk is {n} tokens"


def test_prompt_ids():
    for tok in (BYTES, BPE):
        ids = make_prompt_ids("Is 4 even or odd?", tok)
        assert ids[0] == tok.user_id
        assert ids[-2:] == [tok.end_turn_id, tok.assistant_id]
        assert all(i > 4 for i in ids[1:-2])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
