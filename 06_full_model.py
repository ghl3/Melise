"""
Piece 4: A full transformer language model.

The complete architecture: token embedding + N transformer blocks + final
norm + LM head. Plus one new piece: positional embedding.

WHY POSITIONAL EMBEDDING. Self-attention is permutation-invariant. Q, K, V
depend only on each token's identity, not its position; the lower-triangular
mask is the only thing in the model that knows about order, and it only
*restricts* who sees whom — it doesn't *distinguish* positions. So without
some position-aware signal, the model treats `"the cat sat"` and
`"sat cat the"` identically. We fix that by adding a learned vector per
position to the token embeddings.

This is the simplest possible positional scheme — "absolute learned"
embeddings, used by GPT-2. Modern models use Rotary Position Embedding (RoPE)
applied inside attention, which has nicer properties (relative, generalizes
better to longer contexts). Same role though: tell the model "where you are
in the sequence."
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 64


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, H, Dh).transpose(1, 2)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores = scores.masked_fill(~self.mask[:S, :S], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_delta = self.attn(self.norm1(x))
        mlp_delta = self.mlp(self.norm2(x + attn_delta))
        return x + attn_delta + mlp_delta


class TransformerLM(nn.Module):
    """The full language model: embed -> N blocks -> final norm -> LM head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Token embedding: vocab_size rows, each a D-vector for that token.
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Position embedding: max_seq_len rows, each a D-vector for that position.
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # The transformer stack. N blocks composed in sequence.
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # One final norm before the read-out, standard in modern models.
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, S = token_ids.shape
        # Two embeddings, summed: "what token is at this position?" plus
        # "where in the sequence is this position?"
        positions = torch.arange(S, device=token_ids.device)         # (S,)
        x = self.token_embed(token_ids) + self.pos_embed(positions)  # (B, S, D)

        # Push x through the residual stream. Each block reads x, writes back.
        for block in self.blocks:
            x = block(x)

        # Read out: final norm, then project to vocab-sized logits.
        x = self.final_norm(x)
        return self.lm_head(x)                                       # (B, S, V)


# --- Demo ---
device = torch.device("mps")
cfg = Config()
model = TransformerLM(cfg).to(device)

# Parameter breakdown by component.
print("model parameters:")
groups = {"token_embed": 0, "pos_embed": 0, "blocks": 0, "final_norm": 0, "lm_head": 0}
for name, p in model.named_parameters():
    for k in groups:
        if name.startswith(k):
            groups[k] += p.numel()
            break
for k, n in groups.items():
    print(f"  {k:<14} {n:>10,}")
print(f"  {'total':<14} {sum(p.numel() for p in model.parameters()):>10,}")

# Encode a real string as bytes; (1, S) shape.
prompt = b"hello world"
ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
print(f"\ninput  shape={tuple(ids.shape)}   = (B, S)")

# One forward pass through the whole thing. (1, S) -> (1, S, V).
logits = model(ids)
print(f"output shape={tuple(logits.shape)}  = (B, S, V)")

# Greedy decode the next byte. Untrained -> meaningless content, but a real
# end-to-end pipeline.
next_id = logits[0, -1].argmax().item()
char = chr(next_id) if 32 <= next_id < 127 else f"\\x{next_id:02x}"
print(f"\ngreedy next byte after {prompt!r}: id={next_id}  char={char!r}")
