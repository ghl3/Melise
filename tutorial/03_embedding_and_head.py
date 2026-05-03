"""
Piece 3a: The frame of a language model.

A transformer LM is, at its outermost layer, a sandwich:

    token IDs --[embed]--> vectors --[ ... layers ... ]--> vectors --[unembed]--> logits

The "..." is the transformer stack (attention + MLP layers). We'll add it in
the next piece. First, build the bun: just the embedding and the LM head.

Why this is worth seeing in isolation:
  - You can read the entire forward pass top-to-bottom in three lines.
  - You can see exactly which tensors are *parameters* (weights, learned)
    vs. *activations* (computed each call).
  - You see the I/O shapes that every transformer must respect, regardless
    of how clever the middle is.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    vocab_size: int = 256    # one token per byte → we don't need a tokenizer
    d_model: int = 128       # the "residual stream" width — tokens flow as 128-vectors
    max_seq_len: int = 64    # we'll need this later for positions and the KV cache


class TinyLM(nn.Module):
    """An LM with no transformer layers — just embed and unembed."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # A (vocab_size, d_model) parameter matrix. Looking up token id `t`
        # returns row `t`. There is no math here — it's a gather.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # A (d_model, vocab_size) linear projection. For each position it
        # produces one score per possible next token — the "logits".
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq) int64
        x = self.embed(token_ids)        # (batch, seq, d_model)
        logits = self.lm_head(x)         # (batch, seq, vocab_size)
        return logits


device = torch.device("mps")
cfg = Config()
model = TinyLM(cfg).to(device)

# `.to(device)` walked every parameter and moved it to the GPU. Confirm.
print("model parameters:")
for name, p in model.named_parameters():
    print(f"  {name:<14} shape={str(tuple(p.shape)):<14} device={p.device}  dtype={p.dtype}")
total = sum(p.numel() for p in model.parameters())
print(f"  total parameters: {total:,}")

# Encode a string as raw bytes → token IDs. shape: (batch=1, seq_len)
prompt = "hello"
ids = torch.tensor([list(prompt.encode("utf-8"))], device=device)
print(f"\ninput  shape={tuple(ids.shape)}   device={ids.device}")

# One forward pass.
logits = model(ids)
print(f"output shape={tuple(logits.shape)}   device={logits.device}")

# Logits at the last position = the model's prediction for the next byte.
# This model is randomly initialized, so the answers are meaningless — but
# the *shape* of the output is exactly what a real LM produces.
next_token_logits = logits[0, -1]                     # (vocab_size,)
top = torch.topk(next_token_logits, k=3)
print("\ntop-3 (random, untrained) predictions for the byte after 'hello':")
for score, idx in zip(top.values.tolist(), top.indices.tolist()):
    char = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
    print(f"  byte {idx:3d}  {char!r:>6}  logit={score:+.3f}")
