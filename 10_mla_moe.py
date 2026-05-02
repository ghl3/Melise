"""
Piece 8: DeepSeek-style block — Multi-Latent Attention + Mixture of Experts.

Two architectural changes from our standard block:

  MLA (Multi-Latent Attention) replaces standard MHA. Instead of caching K
  and V (each of size D per token), we project x through a small "latent"
  dim d_kv (with d_kv ≪ D) and *cache only the latent*. K and V are
  decompressed from the latent on the fly inside attention. The cache
  shrinks from 2*D per token to d_kv per token — for our config below,
  256 → 32, an 8× reduction. (DeepSeek-V3 chooses d_kv such that the
  reduction is ~28×.)

  MoE (Mixture of Experts) replaces the dense MLP. Instead of one big
  feedforward through all parameters, we have N small experts and a
  router that picks the top-k experts per token. Each token only flows
  through k of N experts. Total parameter count scales with N; per-token
  compute scales with k. DeepSeek-V3: N=256, k=8.

We keep:
  - Causal mask, position embeddings, KV-style caching, residual stream.
  - The (B, S, D) flow.

We skip:
  - RoPE (DeepSeek's "decoupled RoPE" trick complicates MLA; we use
    learned positional embeddings as in our other files).
  - Shared expert (DeepSeek-V3 has 1 always-on expert + 256 routed; we
    use routed only).
  - Load-balancing aux losses (training-only; doesn't affect inference).
  - Tensor / expert parallelism (single device only here).
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

    # MLA: latent dim that the KV cache stores. d_kv << d_model.
    d_kv: int = 32

    # MoE: number of experts, how many run per token, hidden dim per expert.
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 256        # tuned so that top_k experts ≈ dense MLP compute


# -------------------- KV cache for MLA --------------------

class LatentKVCache:
    """KV cache that stores the per-layer LATENT vector c, not full K/V.

    Memory: (n_layers, B, max_seq_len, d_kv) — single d_kv-sized vector per
    token per layer. No n_heads axis: the latent is shared across heads,
    decompressed inside attention.
    """

    def __init__(self, cfg: Config, batch_size: int, device: torch.device):
        self.c = torch.zeros(cfg.n_layers, batch_size, cfg.max_seq_len, cfg.d_kv, device=device)
        self.length = 0

    def write(self, layer_idx: int, new_c: torch.Tensor) -> None:
        s_new = new_c.shape[-2]
        self.c[layer_idx, :, self.length : self.length + s_new] = new_c

    def advance(self, s_new: int) -> None:
        self.length += s_new


# -------------------- Multi-Latent Attention --------------------

class MultiLatentAttention(nn.Module):
    """Q is computed at full D as usual. K and V are derived from a small
    cached latent c via two upward projections. The cache stores only c."""

    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.head_dim = cfg.d_model // cfg.n_heads

        # Q: regular full-D projection, same as standard MHA.
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # KV: down-project x into a small latent c (THIS gets cached).
        self.kv_down = nn.Linear(cfg.d_model, cfg.d_kv, bias=False)

        # Decompression: latent → full K, full V (run on the fly, not cached).
        self.k_up = nn.Linear(cfg.d_kv, cfg.d_model, bias=False)
        self.v_up = nn.Linear(cfg.d_kv, cfg.d_model, bias=False)

        # Output projection: same as before.
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache: LatentKVCache | None = None) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.cfg.n_heads, self.head_dim

        # New tokens' Q, full D.
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)         # (B, H, S, Dh)

        # New tokens' compressed latent.
        c_new = self.kv_down(x)                                       # (B, S, d_kv)

        if kv_cache is not None:
            pos_offset = kv_cache.length
            kv_cache.write(self.layer_idx, c_new)
            c_full = kv_cache.c[self.layer_idx, :, : pos_offset + S]  # (B, L, d_kv)
        else:
            pos_offset = 0
            c_full = c_new
        L = c_full.shape[1]

        # Decompress latent into full K, V on the fly. These are NOT cached;
        # they're fresh activations for this forward pass.
        k = self.k_up(c_full).view(B, L, H, Dh).transpose(1, 2)       # (B, H, L, Dh)
        v = self.v_up(c_full).view(B, L, H, Dh).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
        scores = scores.masked_fill(~self.mask[pos_offset : pos_offset + S, :L], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


# -------------------- Mixture of Experts --------------------

class Expert(nn.Module):
    """A single expert: just a small two-layer MLP."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.expert_hidden, bias=False)
        self.down = nn.Linear(cfg.expert_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    """Top-k routing. Each token sees k of N experts.

    Forward pattern: loop *over experts*, not over tokens. Each expert
    receives a batch of all tokens that selected it (in any of their top-k
    slots) and processes them in a single batched matmul.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(B * S, D)
        T = B * S

        # Router scores. Higher logit = better match for this token.
        router_logits = self.router(x_flat)                                   # (T, E)
        # Pick the top-k experts per token, then softmax-normalize *only those k*
        # so each token's chosen-expert weights sum to 1.
        topk_logits, topk_idx = torch.topk(router_logits, k=self.cfg.top_k, dim=-1)  # both (T, K)
        topk_weights = torch.softmax(topk_logits, dim=-1)                     # (T, K)

        out = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            # Tokens that chose this expert (in any of their top-K slots).
            chosen = (topk_idx == expert_idx)                  # (T, K) bool
            token_has_expert = chosen.any(dim=-1)              # (T,) bool
            if not token_has_expert.any():
                continue

            token_ids = token_has_expert.nonzero(as_tuple=True)[0]   # indices into x_flat
            # Weight for this expert per chosen token. Usually one of the K slots
            # matched; .sum() handles the rare case where the same expert appears twice.
            w = (chosen[token_ids].float() * topk_weights[token_ids]).sum(dim=-1)   # (n_chosen,)

            expert_in = x_flat[token_ids]                       # (n_chosen, D)
            expert_out = expert(expert_in)                      # (n_chosen, D)
            out[token_ids] += w.unsqueeze(-1) * expert_out

        return out.view(B, S, D)


# -------------------- Block + Model --------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiLatentAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg)

    def forward(self, x, kv_cache=None):
        attn_delta = self.attn(self.norm1(x), kv_cache=kv_cache)
        moe_delta = self.moe(self.norm2(x + attn_delta))
        return x + attn_delta + moe_delta


class TransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids, kv_cache=None):
        B, S = token_ids.shape
        pos_offset = kv_cache.length if kv_cache is not None else 0
        positions = torch.arange(pos_offset, pos_offset + S, device=token_ids.device)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        if kv_cache is not None:
            kv_cache.advance(S)
        return self.lm_head(self.final_norm(x))


# -------------------- Demo --------------------

torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()
model = TransformerLM(cfg).to(device).eval()

prompt = b"hello world"
ids = torch.tensor([list(prompt)], device=device, dtype=torch.long)

cache = LatentKVCache(cfg, batch_size=1, device=device)
with torch.no_grad():
    logits = model(ids, kv_cache=cache)
print(f"prompt: {prompt!r}  →  logits {tuple(logits.shape)}, cache.length={cache.length}\n")

# --- KV cache size: MLA vs hypothetical MHA ---
mla_floats = cfg.n_layers * 1 * cfg.max_seq_len * cfg.d_kv
mha_floats = 2 * cfg.n_layers * 1 * cfg.n_heads * cfg.max_seq_len * (cfg.d_model // cfg.n_heads)
print(f"KV cache (full {cfg.max_seq_len}-token capacity):")
print(f"  MLA cache:  {mla_floats:>7,} floats   ({mla_floats * 4 / 1024:.1f} KB)")
print(f"  MHA cache:  {mha_floats:>7,} floats   ({mha_floats * 4 / 1024:.1f} KB)")
print(f"  reduction:  {mha_floats / mla_floats:.1f}×\n")

# --- MoE parameter count: total vs per-token active ---
expert_params_each = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
expert_params_total = expert_params_each * cfg.n_experts
expert_params_active = expert_params_each * cfg.top_k
router_params = sum(p.numel() for p in model.blocks[0].moe.router.parameters())
print(f"MoE per layer:")
print(f"  experts: {cfg.n_experts}  top_k: {cfg.top_k}  hidden: {cfg.expert_hidden}")
print(f"  per expert:        {expert_params_each:>7,} params")
print(f"  total expert:      {expert_params_total:>7,} params  ({cfg.n_experts}× one expert)")
print(f"  active per token:  {expert_params_active:>7,} params  ({cfg.top_k}× one expert)")
print(f"  ratio:             {expert_params_total / expert_params_active:.1f}× more total than active")
print(f"  router:            {router_params:>7,} params\n")

# --- What did the routers actually do? ---
# Re-run forward, this time hooking into the routing decisions of layer 0.
# (For demo; real code wouldn't use a hook, but it shows what's happening.)
captured = {}
def hook(module, inputs, output):
    x_flat = inputs[0].view(-1, cfg.d_model)
    logits = module.router(x_flat)
    captured["topk_idx"] = torch.topk(logits, k=cfg.top_k, dim=-1).indices.cpu()

handle = model.blocks[0].moe.register_forward_hook(hook)
with torch.no_grad():
    _ = model(ids, kv_cache=LatentKVCache(cfg, 1, device))
handle.remove()

print(f"layer 0 routing decisions (one row per prompt token):")
print(f"  {'token':<7} {'top-k experts':<20}")
for t, byte in enumerate(prompt):
    chars = chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}"
    chosen = captured["topk_idx"][t].tolist()
    print(f"  {chars:<7} {chosen}")

# Tally how often each expert was chosen, across all tokens.
all_choices = captured["topk_idx"].flatten().tolist()
from collections import Counter
counts = Counter(all_choices)
print(f"\n  expert usage (across {len(prompt)} tokens × top_k={cfg.top_k} = {len(prompt) * cfg.top_k} routings):")
for e in range(cfg.n_experts):
    n = counts.get(e, 0)
    bar = "█" * n
    print(f"    expert {e}:  {n:>2}  {bar}")
