"""Tests for the component library and model architectures. Run directly
(no pytest needed):

    .venv/bin/python tests/test_transformer.py

Covers every architecture (base, vanilla, deepseek, kimi3): forward and
backward shapes, causality, incremental-decode vs full-forward
equivalence, MoE balancing updates, KDA decay bounds, and pre-refactor
checkpoint compatibility.
"""

import io
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from transformer import (
    Config,
    DeepSeekConfig,
    DeepSeekMoE,
    Kimi3Config,
    KimiDeltaAttention,
    StableLatentMoE,
    VanillaConfig,
    build_model,
    generate,
)

torch.manual_seed(0)

# Tiny fp32 versions of each architecture — small enough for CPU in seconds.
TINY = {
    "base": Config(
        d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=48,
        n_experts=4, top_k=2, expert_hidden=32, shared_expert_hidden=32,
        dtype=torch.float32,
    ),
    "vanilla": VanillaConfig(
        d_model=64, n_layers=2, n_heads=4, max_seq_len=48,
        ffn_hidden=96, dtype=torch.float32,
    ),
    "deepseek": DeepSeekConfig(
        d_model=64, n_layers=3, n_heads=4, max_seq_len=48,
        kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
        dense_hidden=96, n_experts=4, top_k=2, expert_hidden=32,
        shared_expert_hidden=32, dtype=torch.float32,
    ),
    "kimi3": Kimi3Config(
        d_model=32, n_blocks=1, n_heads=2, max_seq_len=48,
        kv_lora_rank=16, qk_nope_head_dim=16, v_head_dim=16, kda_decay_rank=8,
        dense_hidden=48, n_experts=4, top_k=2, moe_latent_dim=16,
        expert_hidden=16, shared_expert_hidden=32, dtype=torch.float32,
    ),
}


def test_forward_backward_shapes():
    for name, cfg in TINY.items():
        model = build_model(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(ids)
        assert logits.shape == (2, 16, cfg.vocab_size), f"{name}: {logits.shape}"
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), ids.view(-1)
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"{name}: no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"{name}: non-finite grads"


def test_causality():
    """Changing the last token must not change any earlier position's logits."""
    for name, cfg in TINY.items():
        model = build_model(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 12))
        with torch.no_grad():
            a = model(ids)
            ids2 = ids.clone()
            ids2[0, -1] = (ids2[0, -1] + 1) % cfg.vocab_size
            b = model(ids2)
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5), f"{name}: causality broken"
        assert not torch.allclose(a[:, -1], b[:, -1]), f"{name}: last position insensitive"


def test_cache_matches_full_forward():
    """Prefill + one-token decode must reproduce the full-sequence logits."""
    for name, cfg in TINY.items():
        model = build_model(cfg).eval()
        T, prefill = 12, 5
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            full = model(ids)
            cache = model.new_cache(1, ids.device)
            out = model(ids[:, :prefill], kv_cache=cache)
            steps = [out[:, -1]]
            for t in range(prefill, T):
                out = model(ids[:, t : t + 1], kv_cache=cache)
                steps.append(out[:, -1])
        # steps[i] is the prediction after seeing tokens [0..prefill-1+i].
        for i, step_logits in enumerate(steps):
            ref = full[:, prefill - 1 + i]
            err = (step_logits - ref).abs().max().item()
            assert err < 1e-3, f"{name}: cache mismatch at step {i}: {err:.2e}"


def test_generate_smoke():
    for name, cfg in TINY.items():
        model = build_model(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        out = generate(model, ids, max_new_tokens=8)
        assert 1 <= len(out) <= 8 and all(isinstance(t, int) for t in out), name


def test_kda_decay_bounds():
    g_min = -5.0
    kda = KimiDeltaAttention(32, 2, decay_rank=8, g_min=g_min)
    x = torch.randn(2, 10, 32) * 3
    alpha = kda._log_decay(x).exp()
    lower = torch.tensor(g_min).exp().item()
    assert (alpha > lower - 1e-6).all(), "decay below e^{g_min}"
    assert (alpha < 1.0).all(), "decay must stay < 1"
    # The bias init should give a spread of timescales, not one constant.
    assert alpha.std() > 0.01, "decay has no channel diversity"


def test_quantile_balancing():
    """QB should pull a skewed router toward uniform expert load."""
    moe = StableLatentMoE(
        d_model=32, latent_dim=16, n_experts=4, top_k=2,
        expert_hidden=16, n_shared_experts=2, shared_expert_hidden=32,
    ).train()
    # Skew the router hard toward expert 0 and away from expert 1. The
    # inputs get a positive mean so the weight shift becomes a consistent
    # logit shift (with zero-mean inputs it would just be noise).
    with torch.no_grad():
        moe.router.weight[0] += 2.0
        moe.router.weight[1] -= 2.0
    x = torch.randn(4, 64, 32) + 1.0

    def max_load():
        scores = torch.sigmoid(moe.router(x.view(-1, 32)).float())
        idx = (scores + moe.route_bias).topk(moe.top_k, dim=-1).indices
        counts = torch.bincount(idx.reshape(-1), minlength=moe.n_experts).float()
        return (counts / counts.sum()).max().item()

    before = max_load()
    for _ in range(3):
        moe(x)  # train mode → QB update after each forward
    after = max_load()
    assert abs(moe.route_bias.mean().item()) < 1e-5, "bias not centered"
    assert after < before - 0.05, f"QB did not rebalance: {before:.3f} -> {after:.3f}"
    # Inference must freeze the bias.
    moe.eval()
    bias = moe.route_bias.clone()
    moe(x)
    assert torch.equal(bias, moe.route_bias), "bias moved in eval mode"


def test_deepseek_bias_update():
    moe = DeepSeekMoE(
        d_model=32, n_experts=4, top_k=2, expert_hidden=16,
        n_shared_experts=1, shared_expert_hidden=32,
    ).train()
    x = torch.randn(2, 32, 32)
    moe(x)
    b = moe.route_bias
    assert (b != 0).any(), "sign update did not move the bias"
    assert b.abs().max().item() <= moe.bias_update_rate + 1e-9, "step larger than γ"


def test_old_checkpoint_layout_unchanged():
    """The base model must keep the exact pre-refactor state-dict layout
    (token_embed / blocks.N.norm1 / attn.{q,kv,o}_proj / norm2 /
    moe.{router,experts.N.{gate,up,down},shared_expert}) so old
    checkpoints load without remapping."""
    model = build_model(TINY["base"])
    keys = set(model.state_dict().keys())
    expected_subset = {
        "token_embed.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.q_proj.weight",
        "blocks.0.attn.kv_proj.weight",
        "blocks.0.attn.o_proj.weight",
        "blocks.0.norm2.weight",
        "blocks.0.moe.router.weight",
        "blocks.0.moe.experts.0.gate.weight",
        "blocks.0.moe.experts.0.up.weight",
        "blocks.0.moe.experts.0.down.weight",
        "blocks.0.moe.shared_expert.gate.weight",
        "final_norm.weight",
        "lm_head.weight",
    }
    missing = expected_subset - keys
    assert not missing, f"missing old-layout keys: {sorted(missing)}"


def test_configs_pickle_roundtrip():
    """Checkpoints embed the config; every config class must pickle."""
    import pickle

    for name, cfg in TINY.items():
        clone = pickle.loads(pickle.dumps(cfg))
        assert clone == cfg, name
        # And the clone must still build a working model.
        model = build_model(clone)
        assert model.num_parameters() > 0


def test_kimi3_layer_layout():
    """The stack must be 3× (KDA, MoE) + (MLA, MoE) per block + final MLA,
    with exactly one dense FFN (the first)."""
    from transformer.ffn import GatedMLP, StableLatentMoE
    from transformer.models.kimi3 import Kimi3LM

    model = Kimi3LM(Kimi3Config(n_blocks=2))
    attn_kinds = [type(m).__name__ for m in model._attn_modules]
    assert attn_kinds == (
        ["KimiDeltaAttention"] * 3 + ["MultiLatentAttention"]
    ) * 2 + ["MultiLatentAttention"]
    ffn_mods = [s.mod for s in model.stages if not s.is_attn]
    assert isinstance(ffn_mods[0], GatedMLP)
    assert all(isinstance(f, StableLatentMoE) for f in ffn_mods[1:])
    # Every MLA layer must be NoPE + gated; every FFN SiTU-GLU.
    mlas = [m for m in model._attn_modules if type(m).__name__ == "MultiLatentAttention"]
    assert all(m.d_rope == 0 and m.gate_proj is not None for m in mlas)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
