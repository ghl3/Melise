"""Sample text from a trained checkpoint.

Examples:

    Default (latest checkpoint, ROMEO prompt):
        .venv/bin/python scripts/sample.py

    From a specific checkpoint:
        .venv/bin/python scripts/sample.py --checkpoint checkpoints/step_2500.pt

    Custom prompt and length:
        .venv/bin/python scripts/sample.py --prompt "Once upon a time" --tokens 500

    Multi-line prompt (\\n is interpreted):
        .venv/bin/python scripts/sample.py --prompt "ROMEO:\\nO Juliet,"

    Temperature sampling instead of greedy:
        .venv/bin/python scripts/sample.py --temperature 0.8

Loads a checkpoint saved by any stage script (which embeds the Config so
we know how to reconstruct the model). Greedy by default; temperature > 0
enables stochastic sampling.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from transformer import build_model, generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample text from a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path,
                   default=PROJECT_ROOT / "checkpoints" / "latest.pt",
                   help="Path to a checkpoint saved by pretrain.py/sft.py/grpo.py")
    p.add_argument("--prompt", type=str, default="ROMEO:\\n",
                   help="Prefix text. Backslash escapes (\\n, \\t) are interpreted")
    p.add_argument("--tokens", type=int, default=300,
                   help="Number of new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature. 0 = greedy argmax. Higher = more random.")
    p.add_argument("--top-k", type=int, default=0,
                   help="Limit sampling to top-K logits (0 = no limit). Only used when temperature > 0.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (only relevant when temperature > 0)")
    p.add_argument("--device", type=str, default="mps")
    return p.parse_args()


def make_sampler(temperature: float, top_k: int):
    """Build a (logits) -> token_id function."""
    if temperature <= 0:
        def sampler(logits: torch.Tensor) -> int:
            return int(logits.argmax().item())
        return sampler

    def sampler(logits: torch.Tensor) -> int:
        scaled = logits / temperature
        if top_k > 0:
            kth = torch.topk(scaled, k=top_k).values[-1]
            scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return sampler


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    print(f"loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    step = ckpt.get("step", "?")

    # The config's type identifies the architecture (see transformer.models).
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  {type(model).__name__}  step={step}  {model.num_parameters():,} params  "
          f"dtype={cfg.dtype}  device={device}")

    # Interpret backslash escapes (\n, \t) in the user-provided prompt.
    prompt_str = args.prompt.encode().decode("unicode_escape")
    prompt_bytes = prompt_str.encode("utf-8")
    print(f"  prompt: {prompt_bytes!r}")
    print(f"  generating {args.tokens} tokens "
          f"(temperature={args.temperature}{', top_k=' + str(args.top_k) if args.top_k else ''})")
    print()

    ids = torch.tensor([list(prompt_bytes)], device=device, dtype=torch.long)
    sampler = make_sampler(args.temperature, args.top_k)
    out_ids = generate(model, ids, max_new_tokens=args.tokens, sample_fn=sampler)

    # Decode bytes to text. Replace bytes outside the printable range gracefully.
    out_bytes = bytes(b if 0 <= b < 256 else 0x3f for b in out_ids)
    full = prompt_bytes + out_bytes
    print(full.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
