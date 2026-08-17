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


def test_medium_wide_preset():
    """The gen-4 preset must match the frozen proposal numbers: 163.2M
    total / 94.4M routed pool / 78.3M active per token at vocab 8192
    (docs/runs/gen4-ideas.md, Proposal v1 — counted, not estimated)."""
    from transformer.ffn import StableLatentMoE
    from transformer.models import MODELS
    from transformer.models.kimi3 import Kimi3LM, kimi3_medium_wide

    assert "kimi3-medium-wide" in MODELS
    cfg = kimi3_medium_wide(vocab_size=8192, dtype=torch.float32)
    assert (cfg.d_model, cfg.n_blocks, cfg.n_experts, cfg.top_k) == (512, 3, 40, 4)
    assert cfg.n_layers == 13
    model = Kimi3LM(cfg)
    total = model.num_parameters()
    routed = sum(
        p.numel()
        for mod in model.modules() if isinstance(mod, StableLatentMoE)
        for p in mod.experts.parameters()
    )
    active = total - routed + routed * cfg.top_k / cfg.n_experts
    assert abs(total / 1e6 - 163.2) < 0.15, total
    assert abs(routed / 1e6 - 94.4) < 0.15, routed
    assert abs(active / 1e6 - 78.3) < 0.15, active


def test_data_mix_groups():
    """Group shares compile to per-file multipliers: within-group files
    sample by byte size (uniform epochs), group share is exact, and the
    ambiguous/overlapping configs are rejected."""
    import json
    import shutil
    import tempfile

    from transformer.data import PROJECT_ROOT, load_data_mix

    tmp = Path(tempfile.mkdtemp(dir=PROJECT_ROOT, prefix="_tmp_mix_"))
    rel = tmp.relative_to(PROJECT_ROOT)
    try:
        (tmp / "a.txt").write_bytes(b"x" * 100)
        (tmp / "b.txt").write_bytes(b"x" * 300)
        (tmp / "c.txt").write_bytes(b"x" * 600)

        def load(mix):
            mp = tmp / "mix.json"
            mp.write_text(json.dumps(mix))
            return load_data_mix(mp)

        # Two grouped files at share 0.4, one ungrouped at multiplier 0.5:
        # fixed = 600*0.5 = 300 → total = 300/(1-0.4) = 500 → m_g = 0.4*500/400.
        paths, mults, _, groups = load({
            "include": f"{rel}/[abc].txt",
            "groups": {"g": {"include": [f"{rel}/a.txt", f"{rel}/b.txt"],
                             "share": 0.4}},
            "multipliers": {f"{rel}/c.txt": 0.5},
        })
        assert [p.name for p in paths] == ["a.txt", "b.txt", "c.txt"]
        weights = [p.stat().st_size * m for p, m in zip(paths, mults)]
        total = sum(weights)
        assert abs((weights[0] + weights[1]) / total - 0.4) < 1e-9
        assert abs(mults[0] - mults[1]) < 1e-12  # same multiplier within group
        # Membership map: grouped files labeled, ungrouped absent.
        assert {p.name: g for p, g in groups.items()} == {"a.txt": "g", "b.txt": "g"}

        # All files grouped: shares are the relative weights directly.
        paths, mults, _, groups = load({
            "include": f"{rel}/[abc].txt",
            "groups": {
                "g1": {"include": [f"{rel}/a.txt", f"{rel}/b.txt"], "share": 0.7},
                "g2": {"include": f"{rel}/c.txt", "share": 0.3},
            },
        })
        weights = [p.stat().st_size * m for p, m in zip(paths, mults)]
        total = sum(weights)
        assert abs((weights[0] + weights[1]) / total - 0.7) < 1e-9

        # Rejected: file in two groups; grouped file with explicit
        # multiplier; shares >= 1 alongside ungrouped files.
        for bad in (
            {"include": f"{rel}/[abc].txt",
             "groups": {"g1": {"include": f"{rel}/a.txt", "share": 0.2},
                        "g2": {"include": f"{rel}/[ab].txt", "share": 0.2}}},
            {"include": f"{rel}/[abc].txt",
             "multipliers": {f"{rel}/a.txt": 2.0},
             "groups": {"g": {"include": f"{rel}/a.txt", "share": 0.2}}},
            {"include": f"{rel}/[abc].txt",
             "groups": {"g": {"include": f"{rel}/a.txt", "share": 1.0}}},
        ):
            try:
                load(bad)
                raise AssertionError(f"config should have been rejected: {bad}")
            except SystemExit:
                pass
    finally:
        shutil.rmtree(tmp)


def test_fixed_window_eval():
    """Pinned windows: identical results across calls (zero draw noise),
    sane metric ranges, byte-weighted aggregation, stable window starts
    across processes (seeded by file stem)."""
    from transformer.eval import EVAL_METRICS, fixed_window_eval, pin_val_windows
    from transformer.models.kimi3 import Kimi3LM

    torch.manual_seed(0)
    model = Kimi3LM(TINY["kimi3"]).eval()
    seq_len = 24
    val_data = [torch.randint(0, 256, (2000,)), torch.randint(0, 256, (600,))]
    val_paths = [Path("data/aaa.txt"), Path("data/bbb.txt")]

    pinned = pin_val_windows(val_data, val_paths, seq_len, 4, seed=0)
    assert pinned == pin_val_windows(val_data, val_paths, seq_len, 4, seed=0)
    assert all(len(p) == 4 for p in pinned)
    assert pin_val_windows(val_data, val_paths, seq_len, 4, seed=1) != pinned

    agg1, dom1 = fixed_window_eval(model, val_data, val_paths, pinned,
                                   seq_len, 2, agg_weights=[2000.0, 600.0])
    agg2, dom2 = fixed_window_eval(model, val_data, val_paths, pinned,
                                   seq_len, 2, agg_weights=[2000.0, 600.0])
    assert agg1 == agg2 and dom1 == dom2          # deterministic
    assert set(dom1) == {"aaa", "bbb"}
    for d in dom1.values():
        assert set(d) == set(EVAL_METRICS)
        assert 0.0 <= d["acc"] <= d["top5"] <= 1.0
        assert d["loss"] > 0 and d["ent"] > 0
    # Aggregate is the byte-weighted mean of the domain values.
    expect = (dom1["aaa"]["bpb"] * 2000 + dom1["bbb"]["bpb"] * 600) / 2600
    assert abs(agg1["bpb"] - expect) < 1e-9
    # A too-short slice yields no windows and is skipped, not crashed.
    pinned3 = pin_val_windows([val_data[0], torch.randint(0, 256, (10,))],
                              val_paths, seq_len, 4)
    assert pinned3[1] == []
    _, dom3 = fixed_window_eval(model, [val_data[0], torch.randint(0, 256, (10,))],
                                val_paths, pinned3, seq_len, 2)
    assert set(dom3) == {"aaa"}


def test_wsd_schedule_and_windowed_best():
    """WSD: warmup → flat hold → linear decay over the final decay_frac;
    extending --steps mid-hold leaves the current LR untouched. Windowed
    best: no single eval can take the title."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from types import SimpleNamespace

    from run_utils import lr_at, prune_old_checkpoints, windowed_best_val

    args = SimpleNamespace(lr=1e-3, lr_schedule="wsd", warmup_frac=0.01,
                           min_lr_frac=0.1, decay_frac=0.2, steps=1000)
    assert lr_at(5, args) < 1e-3                       # warming up
    assert lr_at(100, args) == lr_at(700, args) == 1e-3  # hold at peak
    assert lr_at(900, args) == 1e-3 + (1e-4 - 1e-3) * 0.5  # mid-decay
    assert abs(lr_at(1000, args) - 1e-4) < 1e-12       # floor at min_lr
    # Extending the run while in the hold does not change today's LR.
    longer = SimpleNamespace(**{**vars(args), "steps": 2000})
    assert lr_at(700, longer) == 1e-3
    # Cosine (unchanged behavior) still reshapes on extension.
    cos = SimpleNamespace(**{**vars(args), "lr_schedule": "cosine"})
    assert lr_at(700, cos) != lr_at(700, SimpleNamespace(
        **{**vars(cos), "steps": 2000}))

    assert windowed_best_val([1.0]) == 1.0
    assert windowed_best_val([3.0, 2.0, 1.0, 100.0]) == (2.0 + 1.0 + 100.0) / 3

    # Keeper checkpoints survive pruning.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for s in (1000, 2000, 25000, 26000, 27000, 50000, 51000):
            (d / f"step_{s}.pt").write_bytes(b"x")
        deleted = prune_old_checkpoints(d, keep_last=2, keep_every=25000)
        kept = sorted(p.name for p in d.glob("step_*.pt"))
        assert "step_25000.pt" in kept and "step_50000.pt" in kept
        assert kept == ["step_25000.pt", "step_27000.pt", "step_50000.pt",
                        "step_51000.pt"], kept
        assert all(int(p.stem.split("_")[1]) % 25000 for p in deleted)


def test_probes_scoring():
    """Probe scorers and the facts table: deterministic split, schema,
    word-boundary matching, echo exclusion."""
    from transformer.probes import (HELDOUT_NAMES, contains_any, fact_split,
                                    load_facts, novel_names, prefix_any,
                                    similarity)
    from transformer.rl.tasks import _NAMES

    facts = load_facts()
    assert len(facts) >= 50
    families = {f["family"] for f in facts}
    assert families >= {"instances", "capitals", "attributes", "counts",
                        "opposites", "membership"}
    splits = {f["split"] for f in facts}
    assert splits == {"train", "heldout"}
    assert all(fact_split(f["id"]) == f["split"] for f in facts)

    assert contains_any("I think it is Paris!", ["paris"])
    assert not contains_any("A capital is on the capital.", ["paris"])
    assert not contains_any("category", ["cat"])          # word boundary
    assert not contains_any("Name an animal", ["dog"])    # echo scores 0
    assert prefix_any("Yes, of course.", ["yes"])
    assert not prefix_any("I would say yes.", ["yes"])
    assert similarity("abcdef", "abcdef") == 1.0
    assert similarity("abcdef", "zzzzzz") < 0.2

    assert not set(HELDOUT_NAMES) & set(_NAMES)
    nn = novel_names(7)
    assert nn == novel_names(7) and not set(nn) & set(_NAMES)

    # Forced-choice entries: naming both options scores nothing (the
    # reject list closes the echo-the-question slack).
    by_id = {f["id"]: f for f in facts}
    sun = by_id["attr-sun"]
    assert contains_any("hot", sun["answers"])
    assert contains_any("hot or cold", sun["reject"])

    # The identity-module contract: training pool disjoint from both
    # eval strata, Melise present, legacy pool preserved.
    from transformer.identity import (HELDOUT_NAMES as IH, NOVEL_SPACE,
                                      TRAIN_NAMES)
    assert not set(TRAIN_NAMES) & set(IH)
    assert not set(TRAIN_NAMES) & NOVEL_SPACE
    assert "Melise" in TRAIN_NAMES and set(_NAMES) <= set(TRAIN_NAMES)
    assert len(TRAIN_NAMES) >= 300


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
