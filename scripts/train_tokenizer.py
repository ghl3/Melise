"""Train the byte-level BPE tokenizer on the project corpora.

Example:

    .venv/bin/python scripts/train_tokenizer.py                # 4096 vocab
    .venv/bin/python scripts/train_tokenizer.py --vocab-size 8192

Writes configs/tokenizer-bpe4k.json (name follows the vocab size) and
prints compression stats per corpus. Models reference the artifact via
their config's `tokenizer` field, so checkpoints stay self-describing;
upload alongside data with scripts/add_dataset.py so VMs can fetch it.

Design decisions (see transformer/tokenizer.py for the runtime wrapper):

  - Byte-level BPE: the base alphabet is all 256 bytes, so any text —
    any language, emoji, binary junk — is always encodable; merges only
    ever ADD compression. Nothing is OOV, same as the raw-bytes setup.
  - Special tokens are the chat template's control characters
    ("\\x00" pad, "\\x01" user, "\\x02" assistant, "\\x03" end-turn,
    "\\x04" end-conv), pinned to ids 0-4. Chat corpora already use these
    bytes as markers, so encoding a stored conversation maps its
    structure to special ids with no re-parsing — and because
    sanitize() strips control chars from all content, no user-provided
    text can ever spell a special token (no prompt injection through
    the tokenizer).
  - Digits are pre-split into single characters before BPE, so numbers
    tokenize digit-by-digit ("55" -> "5","5"). Multi-digit merge tokens
    are a known cause of poor LLM arithmetic, and the RLVR arith task
    is exactly where we need the number system compositional.

Trained on everything in data/ (pretraining corpora AND chat corpora) so
the vocabulary reflects prose, wiki markup, code, math, and chat alike.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["\x00", "\x01", "\x02", "\x03", "\x04"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the byte-level BPE tokenizer on the project corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data",
                   help="Directory of .txt corpora to train on")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: configs/tokenizer-bpe<N>k.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (PROJECT_ROOT / "configs" /
                       f"tokenizer-bpe{args.vocab_size // 1024}k.json")
    files = sorted(str(p) for p in args.data.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt corpora in {args.data}")

    tokenizer = Tokenizer(models.BPE())
    # Isolate digits first (single-digit number tokens), then standard
    # GPT-2-style byte-level pre-tokenization for everything else.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"training {args.vocab_size}-token BPE on {len(files)} corpora...")
    tokenizer.train(files, trainer)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"wrote {out}")

    # Compression stats: bytes per token on a sample of each corpus.
    print(f"\n{'corpus':<24} {'bytes/token':>11}")
    for f in files:
        sample = Path(f).read_bytes()[:2_000_000].decode("utf-8", errors="replace")
        n_tok = len(tokenizer.encode(sample).ids)
        print(f"{Path(f).name:<24} {len(sample.encode('utf-8')) / n_tok:>11.2f}")


if __name__ == "__main__":
    main()
