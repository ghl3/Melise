# Gen-3 — medium on the chat mix (IN PREP — launch pending)

The capacity generation: same pipeline, 3.5× the model, retargeted at
casual chat. Recipe frozen 2026-08-10; results section to be filled
during/after the run. Decisions and rationale: `docs/NOTEBOOK.md` and
user calls logged there (skip gen-2.5; keep the Victorian book
register; d=384 over 512 for the ~4.5-day budget).

## Recipe

| knob | value | why |
|---|---|---|
| model | **kimi3-medium** — d=384, 13 attn layers, 12 heads (head_dim 32), 24 experts top-4 | balanced width+depth via K3 shape rules; **72.0M params** measured with the 8k vocab (145k steps ≈ 20.6 tok/param) |
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

- **Spot VM** `kimi3-train` (converted 2026-08-10, same boot disk,
  ~1/3 cost). Boot-resume crontab is installed **at launch, after the
  pipeline starts** — never earlier: on a pre-launch boot it resumes
  whatever half-state is on disk (learned the hard way during toy
  validation, when a boot fired a real 1500-step tail off gen-2's
  checkpoints while the toy run was tokenizing).
- Auto-restarter: Cloud Scheduler job `kimi3-spot-restart`
  (us-central1, every 15 min → `instances start`; harmless no-op when
  already running). Created 2026-08-10, **PAUSED** — unpause at
  launch, and the DONE_CMD below re-pauses it and halts the VM when
  the pipeline finishes, so a completed run cleans itself up.
- Checkpoint/eval cadence: `PT_EVAL_EVERY=250 PT_SAVE_EVERY=500`
  (pipeline defaults; ~20 min of work at risk per preemption).
  BucketSync mirrors deletions, so pruned checkpoints don't
  accumulate in the bucket.
- Launch sequence: start VM → on VM, launch the pipeline:

      DONE_CMD="gcloud scheduler jobs pause kimi3-spot-restart --location=us-central1; sudo shutdown -h +2" \
      PRESET=kimi3-medium TOKENIZER=bpe8k \
      DATA_MIX=configs/mix-gen3-chat.json \
      PT_STEPS=145000 PT_BATCH=5 SFT_BATCH=5 SFT_STEPS=20000 \
      nohup bash scripts/pipeline.sh > ~/pipeline_nohup.log 2>&1 &

  then, only once it's stepping: `bash scripts/pipeline.sh
  --install-boot-resume` (the @reboot hook needs DONE_CMD too — see
  note below) and unpause `kimi3-spot-restart`.
- **Boot-resume + DONE_CMD**: the crontab entry must carry the same
  DONE_CMD so a post-preemption resume that reaches the end also
  cleans up.

- Estimated wall-clock: pretrain ~4.2d + SFT ~14h + tail ~1h +
  GRPO ~6h + evals ≈ **5.3 days**; ~$40 Spot.

## Pre-launch validation (2026-08-10)

Toy full-pipeline run on the VM (30/20/10/3 steps, real
preset/tokenizer/batch): every stage transition green — pretrain →
SFT → tail (first execution) → GRPO (recall + weighted sampling +
paraphrase prompts + preamble confirmed in rollout logs) → evals
(missing toy rlvr best.pt skipped gracefully) → DONE_CMD executed.
Peak memory across stages **21.3 GiB / 22.5** (worst at GRPO:
policy + frozen reference + rollout caches). bpe8k token cache for
the full mix is pre-warmed (43 cache entries), so the real run
starts stepping immediately. Three bugs found and fixed by the
validation itself: boot-resume firing on pre-launch boots, bare
`--resume` losing the recipe env, and untunable SFT eval cadence.

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
