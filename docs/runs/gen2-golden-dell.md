# Gen-2 — golden-dell (bpe4k small, long context)

Full narrative: `docs/NOTEBOOK.md` 2026-08-10 entry.

## Recipe

| stage | run | config |
|---|---|---|
| pretrain | `kimi3-19M-golden-dell-20260808-204157` | kimi3-small 18.8M, **bpe4k** (vocab 4096, digit-isolated), seq 2048, batch 12, 33,333 steps ≈ 819M tokens (2× Chinchilla), mix-downweight-wiki + code/math @ ×0.15 |
| SFT | `sft-kimi3-19M-golden-dell-20260809-170701` | 12,000 steps @ seq 2048 batch 12, chat + 25k worked-steps arith cold start |
| GRPO | `rlvr-kimi3-19M-golden-dell-20260810-004132` | 600 steps, lr 1e-5, P=16 G=8, KV-batched rollouts |

Hardware: L4 on-demand. Wall-clock 20.4h / 7.6h / 2.7h + 3min evals;
~$27. Throughput 11.1–11.3k tok/s; peak 21.2–21.7 GiB.

## Results

| checkpoint | test bpb |
|---|---|
| pretrain best (step 31,400) | **1.247** (gen-1: 1.560, −20%) |
| sft best | 1.694 (+0.45 forgetting) |
| rlvr best (step 40) | 1.702 (GRPO +0.008 — KL leash works) |

GRPO eval reward **0.700, frozen from step 40**: copy 1.00, parity
1.00 (lr halving cured the oscillation), words 1.00, count 0.75,
**arith 0.00**. dead_frac ~50% all run.

## Incidents / learnings

- **Launch OOM**: batch 16 @ seq 2048 fp32 doesn't fit the L4 — died
  on step 1. Fixed to batch 12 with token budget held (commit
  b5c0494); probe-before-launch is now doctrine.
- **Arith post-mortem**: SFT modeled worked-steps perfectly
  (tasks_bpb 1.04) but never *generated* the format (greedy
  "What is 47 + 12?" → "12"); all-wrong groups ⇒ zero z-scored
  advantage ⇒ RL can't bootstrap. Teacher-forced mastery ≠ generative
  behavior.
- Eval saturation by step 40 wasted 560 GRPO steps (KL drift 0.40,
  no gain).
- Training-log "bpb" was bits-per-TOKEN under BPE (mislabeled);
  fixed for gen-3 with exact byte normalization.
