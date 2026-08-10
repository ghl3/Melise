# Gen-1 — scarlet-harbor (byte-level small)

Backfilled from `docs/NOTEBOOK.md` (2026-08-08 entries) and run dirs in
`gs://…/runs/`.

## Recipe

| stage | run | config |
|---|---|---|
| pretrain | `kimi3-small-17M-scarlet-harbor-20260808-010540` | kimi3-small 16.9M, **bytes** (vocab 256), seq 256, batch 48, 27,500 steps ≈ 338M tokens (D/N 20), mix-downweight-wiki |
| SFT | `sft-kimi3-small-17M-scarlet-harbor-20260808-153022` | 3,000 steps @ seq 1024, chat corpora (SmolTalk/Dolly/OASST1) + 25k simple-answer task cold start |
| GRPO | `rlvr-…-153022` | 200 steps, lr 2e-5, P=16 G=8, unbatched rollouts, 5 task families |

Hardware: L4 on-demand, ~8h + 73min + 2h15m.

## Results

- Pretrain test bpb (enwik8 virgin slice): **1.560** (val 1.523). The
  size sweep it came from: at fixed 1.6 GPU-h, small beat under-fed
  medium (1.927) and large (2.004) — under-training dominates size.
- SFT val 1.359 → 1.043 bits/assistant-byte, still improving at end.
- GRPO eval reward 0.600 → **0.683**: copy 1.00, parity 0.83
  (oscillated 0.83↔0.17), count 0.75, words 0.67, arith 0.20.
- Stop-rate ~100% from SFT onward.

## Incidents / learnings

- Parity instability across evals → gen-2 halved GRPO lr.
- Arith 0.20 = format compliance, not computation; simple-answer cold
  start conflicted with worked-steps scoring → gen-2 redesign.
- VM bucket scope was read-only for a night (silent write failures);
  fixed to storage-rw. First bucket op after boot can hit a stale
  token (retry once).
