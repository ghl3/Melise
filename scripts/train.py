"""Train a small TransformerLM on TinyShakespeare (byte-level).

Run from the project root:
    .venv/bin/python scripts/download_data.py     # once, downloads ~1 MB
    .venv/bin/python scripts/train.py             # ~minutes on MPS

The model is the same `transformer.TransformerLM` we use for inference.
Training runs in fp32 for stability (then we cast back to bf16 if needed
at inference time). Hyperparameters are tuned for "watch the loss come
down on a Mac in a few minutes," not for SOTA.
"""

import sys
import time
from pathlib import Path

# Make the `transformer` package importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from transformer import Config, TransformerLM, generate

# ---------- Hyperparameters ----------

BATCH_SIZE = 16
SEQ_LEN = 128                    # must be ≤ cfg.max_seq_len
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
NUM_STEPS = 500
LOG_EVERY = 25
SAMPLE_EVERY = 100               # generate a sample to watch progress
SAMPLE_TOKENS = 200
SEED = 0

DATA_PATH = PROJECT_ROOT / "data" / "tinyshakespeare.txt"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "tinyshakespeare.pt"


# ---------- Data ----------

def load_data(device: torch.device) -> torch.Tensor:
    """Read the dataset as raw bytes and return a 1D int64 tensor on `device`."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python scripts/download_data.py` first."
        )
    raw = DATA_PATH.read_bytes()
    return torch.tensor(list(raw), dtype=torch.long, device=device)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random crops; return (inputs, targets) where targets is shifted by 1."""
    starts = torch.randint(0, data.shape[0] - seq_len - 1, (batch_size,), device=data.device)
    inputs = torch.stack([data[s : s + seq_len] for s in starts])
    targets = torch.stack([data[s + 1 : s + 1 + seq_len] for s in starts])
    return inputs, targets


# ---------- Training ----------

def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("mps")

    # Train in fp32 for numerical stability — bf16 master weights are doable
    # but require care (mixed-precision optimizer state, gradient scaling).
    # Inference can later cast to bf16.
    cfg = Config(dtype=torch.float32, max_seq_len=SEQ_LEN)
    model = TransformerLM(cfg).to(device)
    print(f"model: {model.num_parameters():,} params, dtype={cfg.dtype}, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    data = load_data(device)
    print(f"dataset: {data.numel():,} bytes")
    print(f"training {NUM_STEPS} steps, batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
    print()

    model.train()
    t_start = time.perf_counter()
    for step in range(1, NUM_STEPS + 1):
        inputs, targets = get_batch(data, BATCH_SIZE, SEQ_LEN)

        logits = model(inputs)                                    # (B, S, V)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            tok_per_sec = step * BATCH_SIZE * SEQ_LEN / elapsed
            print(f"step {step:>4}/{NUM_STEPS}  loss={loss.item():.4f}  "
                  f"({tok_per_sec:>6.0f} tok/s)")

        if step % SAMPLE_EVERY == 0:
            sample(model, device)
            model.train()

    print(f"\ntraining took {time.perf_counter() - t_start:.1f}s")

    # Final sample.
    print("\nfinal sample:")
    sample(model, device, n_tokens=SAMPLE_TOKENS * 2)

    # Checkpoint.
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, CKPT_PATH)
    print(f"\nsaved checkpoint to {CKPT_PATH}")


@torch.no_grad()
def sample(model: TransformerLM, device: torch.device, n_tokens: int = SAMPLE_TOKENS) -> None:
    """Generate a sample from a Shakespeare-y prompt and print it."""
    model.eval()
    prompt = b"ROMEO:\n"
    ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
    out_ids = generate(model, ids, max_new_tokens=n_tokens)
    out_bytes = bytes(b if 0 <= b < 256 else 0x3f for b in out_ids)
    print("---")
    # Show prompt + generation, decoded as UTF-8 (with replacement for bad bytes).
    full = prompt + out_bytes
    print(full.decode("utf-8", errors="replace"))
    print("---")


if __name__ == "__main__":
    main()
