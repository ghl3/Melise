"""
Piece 12: The final model.

Everything we've built, integrated into one self-contained file. Decisions:

  ATTENTION: GQA with n_q_heads=16, n_kv_heads=4 (4:1 grouping)
  POSITION:  RoPE applied inside attention to Q and K
  MLP:       MoE — 8 routed experts (top-2) + 1 shared always-on expert,
             SwiGLU inside each expert
  BLOCK:     sequential (Pre-RMSNorm + residual), MLP sees post-attention stream
  KERNEL:    F.scaled_dot_product_attention (FlashAttention on supported HW)
  PRECISION: bf16 throughout (weights, KV cache, activations)
  INFERENCE: prefill once, then decode token-by-token with the KV cache

The model has the *shape* of a modern Llama / Mixtral-class transformer with
a DeepSeek-style shared expert added on top. Compared to a real production
model we're missing only:
  - Tokenization (we use raw bytes — vocab=256)
  - Training (random weights, gibberish output)
  - Quantization (we serve as bf16, not int4)
  - Multi-device parallelism (single MPS device)

The arithmetic is right; the engineering is right; only the scale is small.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- Config --------------------

@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 512
    n_heads: int = 16
    n_kv_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128

    # MoE
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 1024
    shared_expert_hidden: int = 1024

    # Storage
    dtype: torch.dtype = torch.bfloat16


# -------------------- RoPE --------------------

def precompute_rotary(d_head: int, max_seq_len: int, base: float = 10000.0,
                      device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., S, Dh)  cos, sin: (S, Dh/2). Rotates consecutive dim-pairs of x."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)


# -------------------- KV cache (GQA-aware) --------------------

class KVCache:
    """Cache shape: (n_layers, B, n_kv_heads, max_seq_len, head_dim).

    Stores n_kv_heads worth of K and V — the smaller GQA count, not n_heads.
    This is exactly the cache size reduction GQA buys."""

    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        head_dim = cfg.d_model // cfg.n_heads
        self.k = torch.zeros(cfg.n_layers, batch_size, cfg.n_kv_heads, cfg.max_seq_len, head_dim,
                             device=device, dtype=cfg.dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    def write(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        s = new_k.shape[-2]
        self.k[layer_idx, :, :, self.length : self.length + s] = new_k
        self.v[layer_idx, :, :, self.length : self.length + s] = new_v

    def advance(self, s: int) -> None:
        self.length += s


# -------------------- Building blocks --------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote to fp32 for the normalization to keep numerical stability,
        # then cast back. Standard pattern in real bf16 model code.
        in_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (self.weight * (x32 / rms)).to(in_dtype)


class CausalSelfAttention(nn.Module):
    """GQA + RoPE + SDPA. Three parts of the modern attention recipe."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.d_model // cfg.n_heads

        # Q at full n_heads; K and V at smaller n_kv_heads.
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # RoPE tables. Cast to model dtype so multiplications stay in bf16.
        cos, sin = precompute_rotary(self.head_dim, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos.to(cfg.dtype), persistent=False)
        self.register_buffer("rope_sin", sin.to(cfg.dtype), persistent=False)

        # Bool causal mask for the explicit-mask SDPA path. Used for any case
        # where Q and K positions don't align (i.e., decoding).
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        B, S, D = x.shape
        H, H_kv, Dh = self.cfg.n_heads, self.cfg.n_kv_heads, self.head_dim

        # Project. Q: (B, H, S, Dh). K, V: (B, H_kv, S, Dh).
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)
        kv = self.kv_proj(x).view(B, S, 2, H_kv, Dh)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        # RoPE on Q and K only (not V). Use the absolute-position slice.
        pos_offset = kv_cache.length if kv_cache is not None else 0
        cos = self.rope_cos[pos_offset : pos_offset + S]
        sin = self.rope_sin[pos_offset : pos_offset + S]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # KV cache: write the new (rotated) K, V; read the full prefix.
        if kv_cache is not None:
            kv_cache.write(self.layer_idx, k, v)
            L = pos_offset + S
            k = kv_cache.k[self.layer_idx, :, :, :L]
            v = kv_cache.v[self.layer_idx, :, :, :L]
        else:
            L = S

        # SDPA. enable_gqa=True tells PyTorch to broadcast K, V across the
        # query-head groups internally. Pass an explicit additive mask: this
        # works for both prefill (S == L, lower-triangular) and decode (S < L,
        # rectangular slice).
        bool_mask = self.causal_mask[pos_offset : pos_offset + S, :L]
        attn_mask = torch.zeros(S, L, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(~bool_mask, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class SwiGLUExpert(nn.Module):
    """One MoE expert. Standard SwiGLU FFN — three projections, gate × up,
    one down. Same shape used for both routed and shared experts."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Top-k routed experts + 1 shared always-on expert.

    Forward pattern: loop over experts, batch the chosen tokens through each,
    accumulate weighted outputs. Shared expert runs over all tokens.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(cfg.d_model, cfg.expert_hidden) for _ in range(cfg.n_experts)
        ])
        self.shared_expert = SwiGLUExpert(cfg.d_model, cfg.shared_expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Routed experts — top-k selection and weighting.
        # Router runs in fp32 for stable softmax; weights cast back to model dtype.
        router_logits = self.router(x_flat).float()
        topk_logits, topk_idx = torch.topk(router_logits, k=self.cfg.top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1).to(x.dtype)

        out = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            chosen = (topk_idx == expert_idx)
            token_has_expert = chosen.any(dim=-1)
            if not token_has_expert.any():
                continue
            token_ids = token_has_expert.nonzero(as_tuple=True)[0]
            w = (chosen[token_ids].to(x.dtype) * topk_weights[token_ids]).sum(dim=-1)
            expert_out = expert(x_flat[token_ids])
            out[token_ids] += w.unsqueeze(-1) * expert_out

        # Shared expert: every token, no routing.
        out = out + self.shared_expert(x_flat)

        return out.view(B, S, D)


class Block(nn.Module):
    """Sequential Pre-RMSNorm block: attention then MoE, both residual-added."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        moe_delta = self.moe(self.norm2(x + attn_delta))
        return x + attn_delta + moe_delta


class TransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # No pos_embed — RoPE handles positions inside attention.
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        x = self.token_embed(token_ids)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(token_ids.shape[1])
        return self.lm_head(self.final_norm(x))


# -------------------- Inference --------------------

@torch.no_grad()
def generate(model: TransformerLM, prompt_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    cache = KVCache(model.cfg, prompt_ids.shape[0], prompt_ids.device)

    # Prefill
    logits = model(prompt_ids, kv_cache=cache)
    next_id = logits[0, -1].argmax().item()
    new_tokens = [next_id]

    # Decode
    for _ in range(max_new_tokens - 1):
        if cache.length >= model.cfg.max_seq_len:
            break
        next_input = torch.tensor([[next_id]], device=prompt_ids.device, dtype=torch.long)
        logits = model(next_input, kv_cache=cache)
        next_id = logits[0, -1].argmax().item()
        new_tokens.append(next_id)
    return new_tokens


# -------------------- Demo --------------------

torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()

# Build the model in bf16 from the start. (We could instead build in fp32 and
# .to(bf16); we do it here by constructing then casting — simpler in eager
# code. The "with torch.device('meta'):" pattern is the production way.)
model = TransformerLM(cfg).to(device).to(cfg.dtype).eval()

# --- Parameter breakdown ---
def count_params(prefix: str) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))

print(f"FINAL MODEL — d_model={cfg.d_model}, layers={cfg.n_layers}, dtype={cfg.dtype}")
print(f"             {cfg.n_heads} Q heads, {cfg.n_kv_heads} KV heads (GQA), head_dim={cfg.d_model // cfg.n_heads}")
print(f"             MoE: {cfg.n_experts} routed + 1 shared, top_k={cfg.top_k}, expert_hidden={cfg.expert_hidden}")
print()

# Per-component counts
attn_per_block = sum(p.numel() for n, p in model.blocks[0].named_parameters() if n.startswith(("attn", "norm1")))
moe_per_block = sum(p.numel() for n, p in model.blocks[0].named_parameters() if n.startswith(("moe", "norm2")))
moe_routed_per_block = sum(p.numel() for n, p in model.blocks[0].moe.named_parameters() if n.startswith("experts"))
moe_shared_per_block = sum(p.numel() for n, p in model.blocks[0].moe.named_parameters() if n.startswith("shared_expert"))
moe_router_per_block = sum(p.numel() for n, p in model.blocks[0].moe.named_parameters() if n.startswith("router"))

print(f"per-block parameter breakdown:")
print(f"  attention + norm1:          {attn_per_block:>10,}")
print(f"  MoE (total):                {moe_per_block:>10,}")
print(f"    router:                   {moe_router_per_block:>10,}")
print(f"    {cfg.n_experts} routed experts:        {moe_routed_per_block:>10,}  ({moe_routed_per_block // cfg.n_experts:,} each)")
print(f"    shared expert:            {moe_shared_per_block:>10,}")
print()

# Active vs total params per token (for the MoE block).
expert_params_each = moe_routed_per_block // cfg.n_experts
moe_active = moe_router_per_block + cfg.top_k * expert_params_each + moe_shared_per_block
print(f"per-block, per-token compute (active params):")
print(f"  attention:                  {attn_per_block:>10,}  (always all active)")
print(f"  MoE active:                 {moe_active:>10,}  ({cfg.top_k} routed + 1 shared = {cfg.top_k + 1} experts)")
print(f"  MoE total:                  {moe_per_block:>10,}")
print(f"  total → active per token:   {(attn_per_block + moe_active) / (attn_per_block + moe_per_block):.1%}")
print()

total = sum(p.numel() for p in model.parameters())
embedding_params = count_params("token_embed") + count_params("lm_head") + count_params("final_norm")
print(f"whole model: {total:,} params total")
print(f"  embeddings + LM head + final norm: {embedding_params:,}")
print(f"  {cfg.n_layers} blocks × {attn_per_block + moe_per_block:,} = {cfg.n_layers * (attn_per_block + moe_per_block):,}")
print()

# KV cache size for full max_seq_len.
cache = KVCache(cfg, 1, device)
cache_bytes = cache.k.numel() * cache.k.element_size() + cache.v.numel() * cache.v.element_size()
mha_equivalent = 2 * cfg.n_layers * 1 * cfg.n_heads * cfg.max_seq_len * (cfg.d_model // cfg.n_heads) * 2  # bf16 = 2 bytes
print(f"KV cache, full {cfg.max_seq_len}-position capacity, batch=1, bf16:")
print(f"  GQA cache:           {cache_bytes / 1024:>8.1f} KB  ({cfg.n_kv_heads} KV heads)")
print(f"  MHA-equivalent:      {mha_equivalent / 1024:>8.1f} KB  ({cfg.n_heads} KV heads)")
print(f"  reduction:           {mha_equivalent / cache_bytes:.1f}×")
print()

# --- Inference ---
prompt = b"hello world. once upon a time"
prompt_ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)
print(f"prompt: {prompt!r} ({prompt_ids.shape[1]} tokens)")

new_tokens = generate(model, prompt_ids, max_new_tokens=40)
out_bytes = bytes(b if 0 <= b < 256 else 0x3f for b in new_tokens)
print(f"generated ({len(new_tokens)} tokens):")
print(f"  {prompt!r}")
print(f"  +{out_bytes!r}")
print()
print("(content is gibberish — model is randomly initialized — but the entire pipeline is real)")
