"""
Piece 7: Multi-GPU sharding (simulated on a single GPU).

You have one GPU. Real serving of large models uses many. Three patterns:

  TENSOR PARALLELISM (TP). Split each layer's weight matrices across N
  devices; every forward pass does the matmul *cooperatively* and exchanges
  partial results. Used in essentially every multi-GPU dense-model deployment.
  Megatron-LM is the canonical reference.

  PIPELINE PARALLELISM (PP). Split the *layers* across devices: layers 1-10
  on device A, 11-20 on B, etc. Activations cascade between devices.
  Communication is point-to-point; the challenge is keeping all devices busy
  (microbatching, GPipe, 1F1B schedules).

  EXPERT PARALLELISM (EP). For Mixture-of-Experts models. Different experts
  live on different devices; tokens get routed (all-to-all communication) to
  the device hosting the experts they need. DeepSeek-V3, Mixtral, Switch-Transformer.

Real systems combine these. DeepSeek-V3 inference uses TP inside a node and
EP across nodes; training systems add data parallelism (DP) to that.

This file demonstrates TP for attention. We fake N devices by holding N
weight shards on the single MPS GPU; the collective ops are local sum/concat.
The math is identical to a real cluster — what changes in production is the
implementation of `all_reduce` (NCCL over NVLink) and `all_gather`.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 128
    n_heads: int = 4              # must be divisible by N_SHARDS
    max_seq_len: int = 64


# ---------- Faked collective operations ----------
# In real distributed code these are NCCL primitives (one line each, but
# under the hood: ring algorithms over NVLink/InfiniBand). The semantics
# are identical: every device ends up with the same final tensor.

def all_reduce_sum(per_shard: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sum tensors across shards; broadcast result to every shard."""
    total = sum(per_shard)
    return [total for _ in per_shard]


def all_gather(per_shard: list[torch.Tensor], dim: int) -> list[torch.Tensor]:
    """Concatenate tensors across shards along `dim`; broadcast to every shard."""
    full = torch.cat(per_shard, dim=dim)
    return [full for _ in per_shard]


# ---------- Reference (unsharded) attention ----------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
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


# ---------- Tensor-parallel attention ----------

class TPCausalSelfAttention(nn.Module):
    """Megatron-style tensor-parallel attention.

    Weight sharding rule:
      qkv_proj: COLUMN-PARALLEL.  Output dim split N ways → each shard owns a
                subset of heads. Linear is (D → 3D/N) per shard. No comm in
                forward — each shard's heads are self-contained.
      o_proj:   ROW-PARALLEL.    Input dim split N ways → each shard's input
                is the local head outputs (D/N). Linear is (D/N → D) per shard.
                Each shard produces a partial sum; one all-reduce combines them.

    Net communication per attention block: ONE all_reduce (D-width tensor).
    Same pattern for MLP: column-parallel up + row-parallel down, one all_reduce.
    So a full transformer block does TWO all_reduces per layer.
    """

    def __init__(self, cfg: Config, n_shards: int):
        super().__init__()
        assert cfg.n_heads % n_shards == 0, "heads must divide evenly across shards"
        self.cfg = cfg
        self.n_shards = n_shards
        self.head_dim = cfg.d_model // cfg.n_heads
        self.heads_per_shard = cfg.n_heads // n_shards

        # Per-shard qkv linear. Each holds (D × 3D/N). Together they reconstitute
        # the full (D × 3D) weight, sliced along the output axis.
        per_shard_qkv_out = (3 * cfg.d_model) // n_shards
        self.qkv_shards = nn.ModuleList([
            nn.Linear(cfg.d_model, per_shard_qkv_out, bias=False)
            for _ in range(n_shards)
        ])

        # Per-shard o_proj linear. Each holds (D/N × D). Together they tile
        # the full (D × D) weight along the input axis.
        per_shard_in = cfg.d_model // n_shards
        self.o_shards = nn.ModuleList([
            nn.Linear(per_shard_in, cfg.d_model, bias=False)
            for _ in range(n_shards)
        ])

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def load_from(self, attn: CausalSelfAttention) -> None:
        """Slice an unsharded attention's weights into our shards. For verification.

        qkv_proj is subtle: the full weight has shape (3D, D), with rows
        arranged as [Q-rows | K-rows | V-rows]. Each TP shard wants its OWN
        slice of Q, K, AND V — not "all of Q + half of K." So we split the
        weight into (W_Q, W_K, W_V) first, take this shard's slice from each,
        then concatenate. Real TP frameworks (Megatron, vLLM) often pre-pack
        the weight in a TP-aware order to avoid this re-shuffle.
        """
        D = self.cfg.d_model
        per_shard = D // self.n_shards

        # Split the (3D, D) weight into Q, K, V row blocks.
        W_qkv = attn.qkv_proj.weight
        W_Q, W_K, W_V = W_qkv[:D], W_qkv[D:2 * D], W_qkv[2 * D:]

        for i, shard in enumerate(self.qkv_shards):
            rows = slice(i * per_shard, (i + 1) * per_shard)
            shard_W = torch.cat([W_Q[rows], W_K[rows], W_V[rows]], dim=0)
            shard.weight.copy_(shard_W)

        # o_proj.weight has shape (D, D). Row-parallel = split along input dim.
        # Here a contiguous chunk *is* correct because o_proj's input columns
        # correspond directly to head-aligned slices of D.
        for shard, chunk in zip(self.o_shards, attn.o_proj.weight.chunk(self.n_shards, dim=1)):
            shard.weight.copy_(chunk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is replicated on every shard.
        B, S, D = x.shape
        Dh, H_local = self.head_dim, self.heads_per_shard

        # 1) Column-parallel qkv_proj. Each shard computes its own slice of Q,K,V
        #    independently. No communication: each shard ends up with H_local heads
        #    of Q, K, V — completely self-contained.
        per_shard_out = []
        for qkv_shard in self.qkv_shards:
            qkv = qkv_shard(x)                                     # (B, S, 3D/N)
            q, k, v = qkv.chunk(3, dim=-1)                          # each (B, S, D/N)
            q = q.view(B, S, H_local, Dh).transpose(1, 2)
            k = k.view(B, S, H_local, Dh).transpose(1, 2)
            v = v.view(B, S, H_local, Dh).transpose(1, 2)

            # 2) Attention math on this shard's heads only — no cross-shard interaction.
            scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
            scores = scores.masked_fill(~self.mask[:S, :S], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out = (weights @ v).transpose(1, 2).contiguous().view(B, S, -1)   # (B, S, D/N)
            per_shard_out.append(out)

        # 3) Row-parallel o_proj. Each shard's input is its local head outputs;
        #    each shard produces a partial sum (full D-width but missing the
        #    contribution from other shards' heads).
        partials = [o(out_i) for o, out_i in zip(self.o_shards, per_shard_out)]

        # 4) All-reduce: sum the partials across shards. After this, every shard
        #    holds the same full output.
        summed = all_reduce_sum(partials)
        return summed[0]


# ---------- Demo: verify TP attention reproduces the unsharded forward ----------

torch.manual_seed(0)
device = torch.device("mps")
cfg = Config()
N_SHARDS = 2

# Reference attention.
attn = CausalSelfAttention(cfg).to(device)

# TP attention loaded from the same weights.
tp_attn = TPCausalSelfAttention(cfg, n_shards=N_SHARDS).to(device)
tp_attn.load_from(attn)

# Same input on both.
x = torch.randn(1, 7, cfg.d_model, device=device)
with torch.no_grad():
    y_ref = attn(x)
    y_tp  = tp_attn(x)

diff = (y_ref - y_tp).abs().max().item()
print(f"max |unsharded - TP|: {diff:.2e}")
print("✓ TP attention reproduces unsharded output" if diff < 1e-5 else "✗ MISMATCH")

# Per-shard parameter footprint.
print(f"\nN_SHARDS = {N_SHARDS}")
print(f"\nUnsharded attention weights:")
for n, p in attn.named_parameters():
    print(f"  {n:<18} {tuple(p.shape)}   {p.numel():,} params")

print(f"\nPer-shard weights (one shard, in TP):")
for n, p in tp_attn.qkv_shards[0].named_parameters():
    print(f"  qkv_shard.{n:<8} {tuple(p.shape)}   {p.numel():,} params")
for n, p in tp_attn.o_shards[0].named_parameters():
    print(f"  o_shard.{n:<10} {tuple(p.shape)}   {p.numel():,} params")

per_shard_total = sum(p.numel() for s in tp_attn.qkv_shards[:1] for p in s.parameters())
per_shard_total += sum(p.numel() for s in tp_attn.o_shards[:1] for p in s.parameters())
total_unsharded = sum(p.numel() for p in attn.parameters())
print(f"\nper-shard total: {per_shard_total:,} ({per_shard_total / total_unsharded * 100:.0f}% of unsharded)")
print(f"summed across {N_SHARDS} shards: {per_shard_total * N_SHARDS:,} (matches unsharded)")
