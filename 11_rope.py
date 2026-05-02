"""
Piece 9: Rotary Position Embedding (RoPE).

Replaces our learned `pos_embed` lookup. Instead of *adding* position info
to token vectors before they enter the model, RoPE *rotates* the Q and K
vectors inside attention by an angle proportional to their position.

Why this is the modern default:
  - The attention score Q·K^T after rotation depends on the *relative
    offset* (i - j), not absolute (i, j). Language is much more about
    "what's 3 tokens back?" than "what's at slot 47?", so this matches
    the natural structure of the task.
  - No learned parameters. The rotation frequencies are fixed sinusoids.
    Nothing to fit, nothing to overfit.
  - Generalizes to longer sequences than seen in training, often with
    just a base-frequency tweak (NTK-aware scaling, YaRN).
  - The rotation is per-token-per-head and *additive* in the sense that
    each token's K is rotated by its own position before caching, so
    cached entries naturally retain their position info.

The math:
  - Treat consecutive *pairs* of the head vector as 2D points.
  - Rotate each pair by angle θ_d(pos) = pos / base^(2d / Dh).
  - Different pairs use different frequencies — low-index pairs rotate
    fast, high-index pairs rotate slowly. Multi-scale encoding.
  - cos and sin lookup tables are precomputed once.

Used in: Llama 2/3, Mistral, Mixtral, DeepSeek (decoupled RoPE), Qwen,
Gemma, Phi — essentially every modern decoder LLM since ~2023.
"""

from __future__ import annotations
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


# ---------- RoPE primitives ----------

def precompute_rotary(d_head: int, max_seq_len: int, base: float = 10000.0,
                      device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin tables.

    For each (position, pair_index) we get an angle θ.
    Returns:
      cos: (max_seq_len, d_head / 2)
      sin: (max_seq_len, d_head / 2)
    """
    # Each *pair* of dims gets its own frequency. Low-index pairs rotate fast,
    # high-index pairs rotate slowly. Standard "10000 base" from the RoPE paper.
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)            # (max_seq_len, d_head/2)
    return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs of x's last dim by the given angles.

    x: (..., S, Dh)
    cos, sin: (S, Dh/2)

    Treats x[..., 0], x[..., 1] as a 2D vector; rotates by angle.
    Then x[..., 2], x[..., 3]; rotates by a different angle. Etc.
    """
    x1 = x[..., 0::2]   # even-indexed dims
    x2 = x[..., 1::2]   # odd-indexed dims
    # Broadcast cos/sin across leading dims (B, H).
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    # 2D rotation: [x1] = [cos -sin] [x1]
    #              [x2]   [sin  cos] [x2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # Re-interleave even/odd back into the last dim.
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)


# ---------- KV cache (unchanged from piece 5) ----------

class KVCache:
    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        Dh = cfg.d_model // cfg.n_heads
        self.k = torch.zeros(cfg.n_layers, batch_size, cfg.n_heads, cfg.max_seq_len, Dh, device=device)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def write(self, layer_idx, new_k, new_v):
        s = new_k.shape[-2]
        self.k[layer_idx, :, :, self.length : self.length + s] = new_k
        self.v[layer_idx, :, :, self.length : self.length + s] = new_v

    def advance(self, s):
        self.length += s


# ---------- Model with RoPE ----------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # RoPE tables: shared across heads, indexed by absolute position.
        cos, sin = precompute_rotary(self.head_dim, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, S, H, Dh).transpose(1, 2)             # (B, H, S, Dh)
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)

        # Rotate Q and K by their absolute positions. V is *not* rotated —
        # only the dot-product side gets position info; values are left alone.
        pos_offset = kv_cache.length if kv_cache is not None else 0
        cos = self.rope_cos[pos_offset : pos_offset + S]
        sin = self.rope_sin[pos_offset : pos_offset + S]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Cache the *already-rotated* K. Future steps' Qs (rotated by their
        # positions) dotted with these cached Ks (rotated by theirs) produce
        # scores that depend on the relative offset. That's the whole magic.
        if kv_cache is not None:
            kv_cache.write(self.layer_idx, k, v)
            L = pos_offset + S
            k = kv_cache.k[self.layer_idx, :, :, :L]
            v = kv_cache.v[self.layer_idx, :, :, :L]
        else:
            L = S

        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores = scores.masked_fill(~self.mask[pos_offset : pos_offset + S, :L], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
    def forward(self, x, kv_cache=None):
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        mlp_delta = self.mlp(self.norm2(x + attn_delta))
        return x + attn_delta + mlp_delta


class TransformerLM(nn.Module):
    """Same as before, but NO learned `pos_embed`. RoPE handles positions
    entirely from inside attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # No self.pos_embed!
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids, kv_cache=None):
        x = self.token_embed(token_ids)            # (B, S, D) — no positions added here
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(token_ids.shape[1])
        return self.lm_head(self.final_norm(x))


# ---------- Demo ----------

torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()
model = TransformerLM(cfg).to(device).eval()

# Param counts: notice no pos_embed entry.
print("model parameters (no pos_embed — RoPE adds none):")
groups = {"token_embed": 0, "blocks": 0, "final_norm": 0, "lm_head": 0}
for name, p in model.named_parameters():
    for k in groups:
        if name.startswith(k):
            groups[k] += p.numel()
            break
for k, n in groups.items():
    print(f"  {k:<14} {n:>10,}")
print(f"  {'total':<14} {sum(p.numel() for p in model.parameters()):>10,}")

# Verify the inference pipeline still works end-to-end with RoPE.
prompt = b"hello world"
ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
cache = KVCache(cfg, 1, device)
with torch.no_grad():
    logits = model(ids, kv_cache=cache)
print(f"\nforward pass: input {tuple(ids.shape)} -> logits {tuple(logits.shape)}")
print(f"cache length after prefill: {cache.length}")

# Demonstrate the RELATIVE-position property numerically.
# Take two random head vectors q and k, rotate them at different absolute
# positions but the same relative offset, and show that q·k stays the same.
print("\n--- Relative position property of RoPE ---")
print("Random q at pos i, k at pos j:")
print("  the dot product q·k after RoPE depends ONLY on (i - j), not absolute (i, j)")
Dh = cfg.d_model // cfg.n_heads
cos_full, sin_full = precompute_rotary(Dh, cfg.max_seq_len, device=device)

torch.manual_seed(42)
q = torch.randn(1, 1, 1, Dh, device=device)
k = torch.randn(1, 1, 1, Dh, device=device)
print(f"  q,k drawn once:  shape per vec = {tuple(q.shape[-2:])}")
print()
for offset in [1, 2, 5]:
    print(f"  offset = q_pos - k_pos = {offset}:")
    for k_pos in [0, 5, 20, 40]:
        q_pos = k_pos + offset
        if q_pos >= cfg.max_seq_len:
            continue
        q_rot = apply_rotary(q, cos_full[q_pos:q_pos+1], sin_full[q_pos:q_pos+1])
        k_rot = apply_rotary(k, cos_full[k_pos:k_pos+1], sin_full[k_pos:k_pos+1])
        score = (q_rot * k_rot).sum().item()
        print(f"    q_pos={q_pos:3d}, k_pos={k_pos:3d}  →  q·k = {score:+.6f}")
