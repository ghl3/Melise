"""Tests for the post-training stack. Run directly (no pytest needed):

    .venv/bin/python tests/test_posttraining.py

Covers the byte chat template (round-trip, masking, turn parsing), the
SFT data path (turn-boundary splitting, conversation batching), the GRPO
math (group advantages, rollout tensorization, clipped+KL loss), and the
verifiable reward tasks (every canonical answer earns full credit from
its own scorer).
"""

import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from transformer.chat import (
    ASSISTANT,
    END_CONV,
    END_TURN,
    USER,
    assemble,
    assistant_mask,
    completion_text,
    encode_conversation,
    make_prompt,
    parse_turns,
    split_conversations,
)
from transformer.data import conversation_batch, split_conversation
from transformer.rl import TASKS, group_advantages, grpo_loss, pad_rollouts

torch.manual_seed(0)


def test_chat_roundtrip():
    turns = [("user", "Hi there"), ("assistant", "Hello!"),
             ("user", "Bye"), ("assistant", "See ya")]
    conv = encode_conversation(turns)
    assert parse_turns(conv) == [(r, t.encode()) for r, t in turns]
    # assemble is the byte-level inverse of parse_turns.
    assert assemble(parse_turns(conv)) == conv
    # A corpus blob splits back into its conversations, each intact.
    blob = conv + encode_conversation([("user", "a"), ("assistant", "b")])
    parts = split_conversations(blob)
    assert len(parts) == 2 and parts[0] == conv


def test_chat_prompt_and_completion():
    assert make_prompt("Hi") == bytes([USER]) + b"Hi" + bytes([END_TURN, ASSISTANT])
    # Completion decoding cuts at the stop byte and drops stray markers.
    assert completion_text(b"Yes" + bytes([END_TURN]) + b"garbage") == "Yes"
    assert completion_text(bytes([ASSISTANT]) + b"ok") == "ok"


def test_assistant_mask():
    conv = encode_conversation([("user", "ab"), ("assistant", "cd")])
    # bytes: 0x01 a b 0x03 0x02 c d 0x03 0x04
    #        0    1 2 3    4    5 6 7    8
    # Trainable targets: assistant content (5,6), its END_TURN (7), END_CONV (8).
    assert assistant_mask(conv) == [False] * 5 + [True] * 4


def test_split_conversation():
    turns = [("user", f"question {i} " + "x" * 20) if r == 0 else
             ("assistant", f"answer {i} " + "y" * 20)
             for i in range(6) for r in (0, 1)]
    conv = encode_conversation(turns)
    chunks = split_conversation(conv, 120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120
        assert any(r == "assistant" for r, _ in parse_turns(chunk))
    # No assistant content is lost across the split.
    got = [t for c in chunks for t in parse_turns(c) if t[0] == "assistant"]
    want = [(r, t.encode()) for r, t in turns if r == "assistant"]
    assert got == want
    # Short conversations pass through untouched.
    assert split_conversation(conv, len(conv)) == [conv]


def test_conversation_batch():
    conv = encode_conversation([("user", "ab"), ("assistant", "cd")])
    inputs, targets, mask = conversation_batch([conv], [0], 16, torch.device("cpu"))
    assert inputs.shape == (1, 16) and targets.shape == (1, 16)
    # target[t] = conv[t+1]; trainable where that byte is an assistant target.
    assert mask[0].tolist() == [False] * 4 + [True] * 4 + [False] * 8
    assert targets[0, 4:8].tolist() == [ord("c"), ord("d"), END_TURN, END_CONV]
    # Padding is never a target.
    assert not mask[0, 8:].any()


def test_group_advantages():
    adv = group_advantages(torch.tensor([[1.0, 0.0], [0.5, 0.5]]))
    # Group 1 z-scores to ±1/std; group 2 is constant -> zero advantage.
    assert torch.allclose(adv[:2], torch.tensor([0.7071, -0.7071]), atol=1e-2)
    assert torch.allclose(adv[2:], torch.zeros(2))


def test_pad_rollouts():
    seqs = [
        (b"AB", [67, END_TURN], torch.tensor([-0.1, -0.2])),
        (b"CD", [69, 70, END_TURN], torch.tensor([-0.3, -0.4, -0.5])),
    ]
    b = pad_rollouts(seqs)
    assert b.ids[0].tolist() == [65, 66, 67, END_TURN, 0]
    assert b.ids[1].tolist() == [67, 68, 69, 70, END_TURN]
    assert b.mask.sum(dim=1).tolist() == [2, 3]
    assert b.total_tokens == 5
    # pos points at the logit that predicts tok: ids[pos[t] + 1] == tok[t].
    for i in range(2):
        for t in range(int(b.mask[i].sum())):
            assert b.ids[i, b.pos[i, t] + 1] == b.tok[i, t]
    assert torch.allclose(b.old_lp[1, :3], torch.tensor([-0.3, -0.4, -0.5]))


def test_grpo_loss_at_identity():
    # new == old == ref: ratio 1, KL 0 -> loss is minus the masked mean
    # advantage; z-scored groups make that ~0 across a full group.
    lp = torch.randn(2, 3)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3, dtype=torch.bool)
    loss, stats = grpo_loss(lp, lp, lp, adv, mask,
                            clip_eps=0.2, kl_coef=0.05, total_tokens=6)
    assert abs(float(loss)) < 1e-6
    assert stats["kl"] < 1e-6 and stats["clipped"] == 0


def test_grpo_loss_direction():
    # Raising the log-prob of positive-advantage tokens must lower the loss.
    old = torch.zeros(1, 2)
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 2, dtype=torch.bool)
    lo, _ = grpo_loss(old, old, old, adv, mask,
                      clip_eps=0.2, kl_coef=0.0, total_tokens=2)
    hi, _ = grpo_loss(old + 0.1, old, old + 0.1, adv, mask,
                      clip_eps=0.2, kl_coef=0.0, total_tokens=2)
    assert float(hi) < float(lo)


def test_task_canonical_answers():
    rng = random.Random(0)
    for name, gen in TASKS.items():
        for _ in range(25):
            task = gen(rng)
            score = task.score(task.answer)
            assert score >= 0.99, f"{name}: {task.prompt!r} -> {task.answer!r} = {score}"


def test_arith_format_credit():
    # Wrong-but-worked rollouts must get partial credit (nonzero group
    # variance is what lets GRPO bootstrap the format — gen-2 lesson).
    rng = random.Random(3)
    for _ in range(50):
        t = TASKS["arith"](rng)
        assert t.score(t.answer) == 1.0
        assert t.score("well it = hmm") == 0.25   # format shown, no answer
        assert t.score("no idea") == 0.0


def test_recall_scoring():
    rng = random.Random(4)
    for _ in range(50):
        t = TASKS["recall"](rng)
        name = t.answer.split()[-1].rstrip(".")
        assert t.score(f"Your name is {name}.") == 1.0
        assert t.score(name) == 1.0                      # bare name is fine
        assert t.score(f"My name is {name}.") == 0.5     # right fact, wrong voice
        assert t.score("Beatrice") == 0.0                # wrong name
        assert t.score(" ".join([name] * 12)) == 0.0     # list-spam blocked


def test_weighted_task_sampling():
    from transformer.rl import TASK_WEIGHTS, sample_tasks
    rng = random.Random(5)
    names = sorted(TASKS)
    kinds = [t.kind for t in sample_tasks(names, 600, rng, weights=TASK_WEIGHTS)]
    counts = {k: kinds.count(k) for k in names}
    assert counts["arith"] > counts["copy"] * 2   # 2.0 vs 0.5 weight
    assert all(counts[k] > 0 for k in names)      # nothing starved
    uniform = [t.kind for t in sample_tasks(names, 600, random.Random(5))]
    assert abs(uniform.count("copy") - 100) < 40  # no-weights path stays uniform


def test_preamble_encoding():
    from transformer.chat import (assistant_mask_ids, encode_conversation,
                                  encode_ids, make_prompt_ids, preamble_of)
    from transformer.tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    pre = "You are X."
    conv = encode_conversation([("user", "hi"), ("assistant", "yo")],
                               preamble=pre)
    assert preamble_of(conv) == pre.encode()
    ids = encode_ids(conv, tok)
    n = len(pre)
    assert ids[:n] == list(pre.encode())      # preamble survives encode_ids
    assert ids[n] == tok.user_id
    assert not any(assistant_mask_ids(ids, tok)[:n + 1])  # never a target
    pid = make_prompt_ids("hi", tok, preamble=pre)
    assert pid[:n] == list(pre.encode()) and pid[n] == tok.user_id
    # Back-compat: no preamble -> identical to the old layout.
    plain = encode_conversation([("user", "hi"), ("assistant", "yo")])
    assert preamble_of(plain) == b""
    assert encode_ids(plain, tok)[0] == tok.user_id


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
