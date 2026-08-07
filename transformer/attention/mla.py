"""Multi-head Latent Attention (MLA) — DeepSeek-V2/V3, Kimi K2/K3.

Instead of caching per-head K and V (2 * n_heads * head_dim per token),
MLA down-projects each token into a small shared latent c = W_dkv @ x and
caches only that. Per-head keys and values are re-expanded from the
latent with learned up-projections during attention. The KV cache shrinks
from O(n_heads * head_dim) to O(kv_lora_rank) per token.

Position encoding — two modes, chosen by `rope`:

  Decoupled RoPE (DeepSeek, rope=True): RoPE can't be applied to keys
    that are reconstructed from a compressed latent (the rotation depends
    on position, so it can't be folded into a position-independent
    cache). DeepSeek's fix: give each query head a small extra RoPE slice
    (qk_rope_head_dim), and derive a single shared RoPE key slice
    directly from x, cached alongside the latent. Attention scores are
    the sum of the "content" (NoPE) part and the positional part.

  NoPE (Kimi K3, rope=False): no position encoding at all. In the Kimi
    hybrid stack the interleaved KDA layers carry position (their decay
    is inherently recency-aware), so the periodic global-attention MLA
    layers can be purely content-addressed. This is also what lets Kimi
    extend context without touching RoPE frequencies.

Output gate (`gated=True`, Kimi K3 "Gated MLA"): an input-dependent,
channel-wise sigmoid gate on the attention output before the output
projection — y = W_o(sigmoid(W_g x) ⊙ attn_out). Kimi K3 applies this
full-rank gate to all MLA layers (Eq. 7 of the K3 report).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..norm import RMSNorm
from .cache import LatentLayerCache, ModelCache
from .rope import apply_rotary, precompute_rotary


class MultiLatentAttention(nn.Module):
    """MLA with a latent KV cache, optional decoupled RoPE, optional output gate.

    Projections (per token x ∈ R^d):
        q                 = W_q x                  → per-head [q_nope | q_rope]
        [c | k_rope_raw]  = W_dkv x                → c is normed and cached
        [k_nope | v]      = W_ukv RMSNorm(c)       → per-head, expanded on the fly
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        max_seq_len: int,
        *,
        layer_idx: int = 0,
        rope: bool = True,
        gated: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        d_nope = qk_nope_head_dim
        d_rope = qk_rope_head_dim if rope else 0
        if rope:
            assert d_rope % 2 == 0, "qk_rope_head_dim must be even (RoPE rotates pairs)"

        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.d_nope, self.d_rope, self.d_v = d_nope, d_rope, v_head_dim
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx

        # Queries carry both the content slice and (optionally) the RoPE slice.
        self.q_proj = nn.Linear(d_model, n_heads * (d_nope + d_rope), bias=False)

        # Down-projection produces the cached latent and the shared RoPE key.
        self.kv_down = nn.Linear(d_model, kv_lora_rank + d_rope, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)

        # Up-projection re-expands the latent into per-head K (content) and V.
        self.kv_up = nn.Linear(kv_lora_rank, n_heads * (d_nope + v_head_dim), bias=False)

        self.o_proj = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

        # Kimi-style full-rank output gate.
        self.gate_proj = nn.Linear(d_model, n_heads * v_head_dim, bias=False) if gated else None

        if d_rope > 0:
            cos, sin = precompute_rotary(d_rope, max_seq_len, rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def make_cache(self, batch_size: int, device: torch.device) -> LatentLayerCache:
        return LatentLayerCache(
            batch_size, self.max_seq_len, self.kv_lora_rank, self.d_rope,
            device, self.q_proj.weight.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H = self.n_heads
        d_nope, d_rope, d_v = self.d_nope, self.d_rope, self.d_v

        pos_offset = kv_cache.length if kv_cache is not None else 0

        q = self.q_proj(x).view(B, S, H, d_nope + d_rope).transpose(1, 2)  # (B, H, S, dq)

        down = self.kv_down(x)                                    # (B, S, d_c + d_rope)
        c_new = self.kv_norm(down[..., : self.kv_lora_rank])      # normed latent — cached
        k_rope_new = down[..., self.kv_lora_rank :]               # (B, S, d_rope)

        if d_rope > 0:
            # Rotate the query RoPE slice per head and the single shared key
            # slice; the key rotation happens BEFORE caching, so cached
            # entries are already position-stamped.
            cos = self.rope_cos[pos_offset : pos_offset + S]
            sin = self.rope_sin[pos_offset : pos_offset + S]
            q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)
            q = torch.cat([q_nope, apply_rotary(q_rope, cos, sin)], dim=-1)
            k_rope_new = apply_rotary(k_rope_new.unsqueeze(1), cos, sin).squeeze(1)

        # Cache write/read: the latent (and RoPE key) get extended; everything
        # per-head below is freshly re-expanded for this forward pass.
        if kv_cache is not None:
            layer = kv_cache.layers[self.layer_idx]
            layer.write(c_new, k_rope_new if d_rope > 0 else None, pos_offset)
            L = pos_offset + S
            c_full, k_rope_full = layer.read(L)
        else:
            L = S
            c_full, k_rope_full = c_new, k_rope_new

        kv = self.kv_up(c_full).view(B, L, H, d_nope + d_v).transpose(1, 2)
        k_nope, v = kv.split([d_nope, d_v], dim=-1)               # (B, H, L, ·)

        if d_rope > 0:
            # The shared RoPE key is broadcast to every head.
            k_rope = k_rope_full.unsqueeze(1).expand(B, H, L, d_rope)
            k = torch.cat([k_nope, k_rope], dim=-1)
        else:
            k = k_nope

        bool_mask = self.causal_mask[pos_offset : pos_offset + S, :L]
        attn_mask = torch.zeros(S, L, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(~bool_mask, float("-inf"))
        # SDPA scales by 1/sqrt(d_nope + d_rope) — the full Q/K width. V may
        # have a different width; SDPA allows that.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).reshape(B, S, H * d_v)
        if self.gate_proj is not None:
            out = torch.sigmoid(self.gate_proj(x)) * out
        return self.o_proj(out)
