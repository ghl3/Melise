# Gen-3 — medium on the chat mix (IN PREP — launch pending)

The capacity generation: same pipeline, 3.5× the model, retargeted at
casual chat. Recipe frozen 2026-08-11; results section to be filled
during/after the run. Decisions and rationale: `docs/NOTEBOOK.md` and
user calls logged there (skip gen-2.5; keep the Victorian book
register; d=384 over 512 for the ~4.5-day budget).

## Recipe

| knob | value | why |
|---|---|---|
| model | **kimi3-medium** — d=384, 13 attn layers, 12 heads (head_dim 32), 24 experts top-4 | balanced width+depth via K3 shape rules; ~74.2M params with the 8k vocab |
| tokenizer | **bpe8k** (8192, digit-isolated, specials pinned 0–4) | 8–12% fewer tokens/byte than bpe4k (enwik8 2.95 B/tok); trained on the gen-3 corpus |
| context | 2048 | chat needs no more; memory feeds batch instead |
| pretrain | **145,000 steps × batch 5 × 2048 ≈ 1.48B tokens** (Chinchilla for 74M) | b5 probed: peak 20.1/22.5 GiB @ 4,098 tok/s (b6 OOMs, b4 wastes throughput) |
| data mix | `configs/mix-gen3-chat.json` | +fineweb-edu 400MB ×0.07 (16.8%), +dialogue 24MB ×0.3 (4.7%), math ×0.05; per-domain val splits (enwik8/fineweb/dialogue/war-and-peace) |
| SFT | 20,000 steps × batch 5, cleaned corpora (190,515 convs) | filtered SmolTalk + all-paths OASST + identity/tasks with **preambles** |
| SFT tail | 1,500 steps @ lr 3e-5, tasks+identity only | make formats generative (gen-2 lesson) |
| GRPO | 600 steps, lr 1e-5, headroom-weighted rollouts, preamble on every prompt | arith 2.0 … copy 0.5; eval uniform |
| metrics | byte-true bpb everywhere, train entropy_bits, pretrain per-domain val, SFT per-source val (`val_source/*`, every 4th eval) | long-run visibility; curves comparable to gen-1/2 and the literature |
| preamble | `You are Lily, a tiny language model.` (varied names in training) | identity read from context, not memorized; serve gen-3 with `--preamble` |

## Infrastructure

- **Spot VM** `kimi3-train` (converted 2026-08-11, same boot disk,
  ~1/3 cost). Boot-resume crontab installed (`--resume` on reboot).
- At launch: create the paused Cloud Scheduler job that calls
  `instances start` every 15 min (Spot preemptions STOP the VM;
  something must start it), then unpause.
- Launch command (VM, after a 10-step re-probe at vocab 8192):

      PRESET=kimi3-medium TOKENIZER=bpe8k \
      DATA_MIX=configs/mix-gen3-chat.json \
      PT_STEPS=145000 PT_BATCH=5 SFT_BATCH=5 SFT_STEPS=20000 \
      nohup bash scripts/pipeline.sh > ~/pipeline_nohup.log 2>&1 &

- Estimated wall-clock: pretrain ~4.2d + SFT ~14h + tail ~1h +
  GRPO ~6h + evals ≈ **5.3 days**; ~$40 Spot.

## Reference numbers to beat (gen-2)

- Pretrain test bpb 1.247 (byte-normalized — only `eval_checkpoint.py`
  numbers are cross-generation comparable).
- GRPO per-task: copy 1.00, parity 1.00, words 1.00, count 0.75,
  arith 0.00, recall n/a (new). Eval rewards are NOT comparable in
  aggregate (template variety + recall changed the eval set).
- Watch: `val_domain/dialogue_movies` and `val_domain/fineweb_edu`
  (is the new register being learned?), `rollout/dead_frac` (arith
  groups coming alive = the format-credit fix working), sampled
  generations for register drift; daily laptop `eval_checkpoint.py`
  on the synced best.pt.

## Results (fill during/after)

- [ ] wall-clock + preemption count per stage
- [ ] pretrain test bpb + per-domain val curves
- [ ] SFT val/tasks bpb; forgetting eval
- [ ] GRPO per-task eval incl. recall; dead_frac trajectory
- [ ] chat battery vs gen-2 (identity, recall, date honesty, arith)
- [ ] incidents
