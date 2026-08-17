"""Kimi K3 in miniature (arXiv:2607.24653).

The layer stack, exactly as the K3 report draws it (Fig. 2), repeated
n_blocks times with one extra global-attention layer at the end so the
backbone always finishes with global token mixing:

    token embedding
      ┌─ per block (×n_blocks): ─────────────────────────────┐
      │   KDA        → Stable LatentMoE                       │
      │   KDA        → Stable LatentMoE                       │  3:1 hybrid:
      │   KDA        → Stable LatentMoE                       │  KDA carries
      │   Gated MLA  → Stable LatentMoE                       │  position (NoPE)
      └──────────────────────────────────────────────────────┘
        Gated MLA    → Stable LatentMoE                          final global layer
      → final norm → head

(The very first FFN is a dense SiTU-GLU MLP instead of MoE — K3, like
DeepSeek-V3 and K2, keeps one dense layer because routing right off the
embedding is unstable.)

There is no plain residual stream: K3 uses Attention Residuals (§2.2).
Every module above — each attention layer AND each FFN — is a *stage*
whose input is an attention-weighted mix over the embedding and all
preceding stage outputs, computed with a learnable per-stage pseudo-query
(see residual/attnres.py). The LM head reads one final learned mix. The
forward pass below keeps the list of stage outputs explicitly.

Kimi K3 at full scale vs this miniature:

                       K3            here (defaults)
    layers             93            9  (2 blocks + final)
    d_model            7168          256
    routed experts     896 / 16 act  16 / 4 act
    MoE latent ℓ       3584 (d/2)    128 (d/2)
    shared experts     2             2

Deliberately not reproduced from the report: multi-token prediction (a
training objective, not an architecture component), the chunkwise KDA
kernel (we run the reference sequential scan — keep seq_len modest when
training), Block AttnRes (only pays off at ~100-layer depth; we use the
full form), the vision pathway (MoonViT-V2), and the Per-Head Muon
optimizer (training here uses AdamW).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..attention import KimiDeltaAttention, ModelCache, MultiLatentAttention
from ..ffn import GatedMLP, StableLatentMoE
from ..norm import RMSNorm
from ..residual import AttentionResiduals


@dataclass
class Kimi3Config:
    vocab_size: int = 256
    d_model: int = 256
    n_blocks: int = 2               # each block = 3 KDA layers + 1 Gated MLA layer
    max_seq_len: int = 512
    n_heads: int = 8                # head_dim 32 → KDA state is 32×32 per head
    # KDA.
    kda_conv_kernel: int = 4
    kda_decay_rank: int = 16
    kda_g_min: float = -5.0
    # Gated MLA (NoPE — position comes from KDA).
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 32
    v_head_dim: int = 32
    # Stable LatentMoE (K3: 896 routed / 16 active on ℓ = d/2).
    n_experts: int = 16
    top_k: int = 4
    moe_latent_dim: int = 128       # ℓ = d_model / 2
    expert_hidden: int = 128
    n_shared_experts: int = 2       # K3 fixes N_s = 2
    shared_expert_hidden: int = 512
    dense_hidden: int = 1024        # the single dense FFN (first layer)
    dtype: torch.dtype = torch.bfloat16

    # Which tokenizer this model's ids come from: "bytes" (the 256
    # byte values) or a trained artifact name like "bpe4k"
    # (transformer.tokenizer.load_tokenizer resolves it). Serialized into
    # checkpoints via the config; read with getattr(cfg, "tokenizer",
    # "bytes") for pre-field checkpoints.
    tokenizer: str = "bytes"

    @property
    def n_layers(self) -> int:
        """Attention layers: n_blocks × (3 KDA + 1 MLA) + the final MLA."""
        return 4 * self.n_blocks + 1


def kimi3_small(**overrides) -> Kimi3Config:
    """~17M params — the default Kimi3Config (d=256, 9 attention layers)."""
    return Kimi3Config(**overrides)


def kimi3_medium(**overrides) -> Kimi3Config:
    """~67M params: d=384, 13 attention layers, 24 experts.

    Scaled from small by the K3 shape rules: head_dim stays 32, the MoE
    latent stays d/2, expert hidden stays ≈ latent, shared/dense hidden
    stay 2d/4d."""
    kwargs = dict(
        d_model=384, n_blocks=3, n_heads=12,
        kv_lora_rank=96, kda_decay_rank=24,
        n_experts=24, top_k=4, moe_latent_dim=192, expert_hidden=192,
        shared_expert_hidden=768, dense_hidden=1536,
    )
    kwargs.update(overrides)
    return Kimi3Config(**kwargs)


def kimi3_medium_wide(**overrides) -> Kimi3Config:
    """~163M params (78M active/token): d=512, 13 attention layers,
    40 experts — the gen-4 shape (docs/runs/gen4-ideas.md, Proposal v1).

    Large's width with medium's depth: 3 blocks, deliberately NOT
    large's 4, to cap FLOPs. The routed pool triples vs medium (31.9M →
    94.4M params) while active/token grows only 1.72× — width fattens
    each expert (1.33M → 2.36M) while the count grows 24 → 40. This
    targets gen-3's measured failure mode: small-domain eviction lives
    in FFN storage, and experts are the cheapest storage per unit
    wall-clock. Per-layer shapes follow the K3 rules at d=512 (head_dim
    32, latent d/2, expert hidden = latent, shared 2d, dense 4d)."""
    kwargs = dict(
        d_model=512, n_blocks=3, n_heads=16,
        kv_lora_rank=128, kda_decay_rank=32,
        n_experts=40, top_k=4, moe_latent_dim=256, expert_hidden=256,
        shared_expert_hidden=1024, dense_hidden=2048,
    )
    kwargs.update(overrides)
    return Kimi3Config(**kwargs)


def kimi3_large(**overrides) -> Kimi3Config:
    """~180M params: d=512, 17 attention layers, 32 experts."""
    kwargs = dict(
        d_model=512, n_blocks=4, n_heads=16,
        kv_lora_rank=128, kda_decay_rank=32,
        n_experts=32, top_k=4, moe_latent_dim=256, expert_hidden=256,
        shared_expert_hidden=1024, dense_hidden=2048,
    )
    kwargs.update(overrides)
    return Kimi3Config(**kwargs)


class _Stage(nn.Module):
    """Pre-norm wrapper around one module. Under Attention Residuals each
    module is its own stage in the depth stream — there is no two-module
    block; the residual mixing lives in AttentionResiduals."""

    def __init__(self, d_model: int, module: nn.Module, is_attn: bool):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mod = module
        self.is_attn = is_attn

    def forward(self, x: torch.Tensor, kv_cache: ModelCache | None = None) -> torch.Tensor:
        h = self.norm(x)
        return self.mod(h, kv_cache=kv_cache) if self.is_attn else self.mod(h)


class Kimi3LM(nn.Module):
    """token embedding → AttnRes over [KDA/MLA + LatentMoE stages] → head."""

    def __init__(self, cfg: Kimi3Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Build the layer stack. Each attention layer is followed by its FFN;
        # both are separate AttnRes stages. `layer_idx` numbers attention
        # layers for the cache.
        stages: list[_Stage] = []
        self._attn_modules: list[nn.Module] = []

        def kda(layer_idx: int) -> KimiDeltaAttention:
            return KimiDeltaAttention(
                cfg.d_model, cfg.n_heads, layer_idx=layer_idx,
                conv_kernel=cfg.kda_conv_kernel, decay_rank=cfg.kda_decay_rank,
                g_min=cfg.kda_g_min,
            )

        def gated_mla(layer_idx: int) -> MultiLatentAttention:
            return MultiLatentAttention(
                cfg.d_model, cfg.n_heads, cfg.kv_lora_rank,
                cfg.qk_nope_head_dim, qk_rope_head_dim=0, v_head_dim=cfg.v_head_dim,
                max_seq_len=cfg.max_seq_len, layer_idx=layer_idx,
                rope=False, gated=True,                       # NoPE + output gate
            )

        def add_layer(attn: nn.Module) -> None:
            layer_idx = len(self._attn_modules)
            self._attn_modules.append(attn)
            stages.append(_Stage(cfg.d_model, attn, is_attn=True))
            if layer_idx == 0:                                # the one dense FFN
                ffn = GatedMLP(cfg.d_model, cfg.dense_hidden, activation="situ")
            else:
                ffn = StableLatentMoE(
                    cfg.d_model, cfg.moe_latent_dim, cfg.n_experts, cfg.top_k,
                    cfg.expert_hidden, cfg.n_shared_experts, cfg.shared_expert_hidden,
                    activation="situ",
                )
            stages.append(_Stage(cfg.d_model, ffn, is_attn=False))

        for _ in range(cfg.n_blocks):
            for _ in range(3):
                add_layer(kda(len(self._attn_modules)))       # 3× KDA ...
            add_layer(gated_mla(len(self._attn_modules)))     # ... then 1× global
        add_layer(gated_mla(len(self._attn_modules)))         # final global layer

        self.stages = nn.ModuleList(stages)
        self.attnres = AttentionResiduals(cfg.d_model, n_stages=len(stages))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: ModelCache | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(token_ids)

        # Attention Residuals: keep every stage output; each stage (and the
        # head) reads its own learned mix of all of them.
        outputs = [x]
        for idx, stage in enumerate(self.stages):
            h = self.attnres(idx, outputs)
            outputs.append(stage(h, kv_cache=kv_cache))
        x = self.attnres(len(self.stages), outputs)

        if kv_cache is not None:
            kv_cache.advance(token_ids.shape[1])
        return self.lm_head(self.final_norm(x))

    def new_cache(self, batch_size: int, device: torch.device | str) -> ModelCache:
        device = torch.device(device)
        return ModelCache([a.make_cache(batch_size, device) for a in self._attn_modules])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
