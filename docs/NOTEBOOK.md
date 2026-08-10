# Experiment notebook

A running log of experimental sessions, newest entry last. Each entry is
dated and captures one session's work: **Goal** (what we set out to do),
**Setup** (models, data, hardware, key parameters), **Results** (numbers,
tables, artifacts), **Learnings** (what the data taught us, incidents
included), and **Next steps** (decisions made and work queued). Metric
context: `bpb` is bits per byte (pretrain: on enwik8 slices; SFT: per
assistant byte — the two are not comparable); RLVR reward is mean task
score in [0, 1] on a fixed greedy-decoded eval set.

---

## 2026-08-08 — Post-training harness + first end-to-end SFT → RLVR run

**Goal.** Build the complete post-training pipeline (chat SFT, then GRPO
on verifiable rewards) for the byte-level kimi3 models, and dry-run it
end-to-end on the strongest existing base model so the next major
pretrain→RLVR generation runs smoothly.

**Setup.**
- Base model: `kimi3-small-17M-scarlet-harbor` (17M params, Chinchilla-fed
  at 338M tokens, seq 256; best val 1.523 bpb).
- Built this session: byte chat template (control bytes 0x01–0x04 as
  role/stop markers; verified absent from all corpora), chat corpus
  (SmolTalk train-only 232k convs + Dolly-15k + OASST1 English 3.7k ≈
  895 MB), cold-start corpus (25k worked examples generated from the RL
  reward tasks themselves; every canonical answer self-checked against
  its scorer), `sft.py` (assistant-masked loss, model rebuilt at seq
  1024), `transformer/rl/` (GRPO: group rollouts, z-scored group
  advantages, clipped surrogate + k3 KL to frozen reference; 5
  programmatic reward tasks: copy / arith / parity / letter-count /
  word-count), offline eval `eval_checkpoint.py` (deterministic bpb on
  the virgin enwik8 test slice), stage layout
  `checkpoints/{pretrain,sft,rlvr}/<run>` mirrored to
  `runs/<stage>/<run>` in the bucket, and lineage metadata
  (identity/stage/lineage) embedded in checkpoints.
- Hardware: one L4 VM (SFT ~11.6k tok/s at seq 1024, batch 16).

**Results.**

First-ever numbers on the reserved enwik8 test slice (final 5 MB, never
touched by training or model selection); test ≈ val + 0.02–0.04, a
healthy selection gap:

| pretrain run | params | tokens | val bpb | test bpb |
|---|---|---|---|---|
| scarlet-harbor | 17M | 338M | 1.523 | **1.560** |
| crisp-harbor | 17M | 65M | 1.864 | 1.880 |
| stormy-summit | 66M | 25M | 1.894 | 1.927 |
| fierce-tide | 180M | 11M | 1.959 | 2.004 |

Pipeline (single chained launch, both stages exit 0):

| stage | steps | wall clock | result |
|---|---|---|---|
| SFT @ seq 1024 | 3000 | 73 min | val 1.359 → **1.043** bpb/assistant-byte; all 15 evals new bests |
| GRPO (P=16, G=8, lr 2e-5, β=0.05) | 200 | 2 h 15 m | eval reward 0.600 → **0.683**, best at final step |

Final per-task eval rewards: **copy 1.00 · parity 0.83 · count 0.75 ·
words 0.67 · arith 0.20**; stop-byte emission ~100% from SFT onward.
Runs: `sft-kimi3-small-17M-scarlet-harbor-20260808-153022` →
`rlvr-kimi3-small-17M-scarlet-harbor-20260808-153022` (bucket-mirrored,
full lineage recorded).

**Learnings.**
- *Cold-start SFT teaches format, not computation.* The 25k synthetic
  examples gave every task a nonzero start except in substance: arith
  went 0 → 0.20 in RL — real but weak. GRPO can only amplify what
  occasionally succeeds. Next round needs worked-steps arithmetic data,
  which conflicts with the current first-integer scorer (design decision
  pending).
- *Byte-level tokenization delivered where predicted:* letter-counting
  (the classic tokenizer stumper) hit 0.75–1.00.
- *RL instability is visible at this scale:* parity oscillated 0.83 ↔
  0.17 across evals — the policy repeatedly finds and loses the answer
  format. Candidate mitigations: lower LR, higher KL coefficient.
- *Both stages were schedule-limited, not capacity-limited* — every SFT
  eval and the last GRPO eval were bests. Longer runs are cheap headroom
  (~$3 for 3× SFT).
- *Never bidirectionally rsync a live run dir.* The TB-aggregation pull
  loop replaced growing event files under their writers (rename →
  writer appends to an unlinked inode → visible file freezes ~2 min in).
  Fix was topological: writers push to the bucket, viewers read the
  bucket (`tensorboard --logdir gs://…/runs` via gcsfs); metrics.jsonl
  is now the authoritative record, `rebuild_tb.py` regenerates TB from
  it, `recover_metrics.py` rebuilds it from stdout logs. Both damaged
  runs were fully recovered.
- *NoPE length extension just works:* base trained at seq 256 fine-tuned
  at 1024 with a strict state-dict load; and L4 throughput at 1024 ≈
  throughput at 256 (KDA is O(L); only the MLA quarter pays O(L²)).
- *GRPO wall-clock is decode-bound* (2 h 15 m for 200 steps vs 73 min
  for 3000 SFT steps) — rollout batching is the lever if RL gets slow.
- ~89% of chat conversations exceeded seq 512 and were being truncated;
  fixed with turn-boundary splitting at load time.

**Next steps.**
- Thread the trained BPE tokenizer (`configs/tokenizer-bpe4k.json`:
  4096 vocab, ~3 bytes/token on prose, digits isolated to single tokens
  for arithmetic, chat control chars pinned to special ids 0–4) through
  data/eval/chat/stage scripts. Decision: fold into the next pretrain
  candidate directly, no A/B experiment.
- Next-gen pretrain candidate: vocab 4096, seq 2048 (new default),
  enriched mix (100 MB Python + 101 MB OpenWebMath at 0.15 multiplier ≈
  21% of sampling).
- GPU quota resubmission window opens 2026-08-09 ~19:20 UTC
  (GPUS_ALL_REGIONS 1→3) — enables parallel runs.
- Round-two post-training: 3× longer SFT; GRPO with lower LR / higher
  KL; redesigned arith cold-start.
- Deferred by choice: DPO trainer; chat UI ("full site" planned).

---

## 2026-08-08 (evening) — Generation-2 readiness: tokenizer, hardened pipeline

**Goal.** Execute the post-dry-run improvement list so the next major
pretrain→RLVR generation launches clean: build and integrate the BPE
tokenizer end-to-end, fix the arith cold-start design flaw, speed up
GRPO rollouts, add the missing diagnostics, and make long runs
self-healing.

**Setup / work done.**
- **Tokenizer**: `configs/tokenizer-bpe4k.json` — byte-level BPE, 4096
  vocab, trained on all 40 corpora. Digits pre-split to single tokens
  (multi-digit merges are a known arithmetic killer). Special tokens are
  the chat template's control characters pinned to ids 0–4, identical
  under the byte tokenizer — chat corpora need no re-encoding, and
  content can never spell a special id (encode strips those chars).
  Threaded everywhere: config `tokenizer` field serialized into
  checkpoints; token-cached corpus loading (byte-slice first, so the
  enwik8 test boundaries stay byte-identical); token-budget conversation
  splitting; eval bpb normalized by slice BYTES (comparable across
  encodings); rollouts stop on the end-turn id. kimi3-small becomes
  18.8M params with the 4096-vocab tables. Compression ≈ 3.0 bytes/token
  prose, 2.45 code, 2.63 enwik8.
- **Arith cold-start v2**: scorers read the LAST integer (showing work
  is rewarded, not punished); canonical answers decompose through the
  nearest ten on carry/borrow ("47 + 3 = 50, 50 + 5 = 55. 55");
  chat_tasks.txt regenerated and re-uploaded.
- **GRPO rollout batching**: equal-length prompts decode as one KV-cache
  batch (several whole groups per forward pass; rectangular prefill only
  — KDA has no clean left-padding).
- **New diagnostics**: GRPO `rollout/dead_frac` (zero-variance groups =
  the cold-start gauge) and `rollout/entropy` (collapse early-warning);
  SFT `val/tasks_bpb` (cold-start progress apart from chat loss).
- **`scripts/pipeline.sh`**: resume-aware 3-stage runner (env-var
  recipe; `--resume` continues any interrupted stage;
  `--install-boot-resume` crontab hook makes Spot preemption self-heal);
  finishes with test-slice evals of all three stages' best.pt — the
  catastrophic-forgetting check.
- **Ops**: laptop TensorBoard reads the bucket directly
  (`scripts/tb_bucket.sh`; ADC + gcsfs + certifi). Repo history now
  backed up as a git bundle in the bucket (was laptop-only). A100 quota
  preference filed (us-central1, pending). gcloud SDK 437→579 replaced
  gcloud-crc32c without re-triggering Gatekeeper.

**Results.** 30/30 tests across three suites (architectures,
post-training, tokenizer). Full toy pretrain→SFT→GRPO chain verified
under bpe4k on MPS: checkpoints carry `tokenizer=bpe4k, vocab=4096`,
lineage threads, new metrics live (dead_frac correctly read 100% on an
untrained policy). VM updated and passing the same tests.

**Learnings.**
- Digit-isolated BPE + last-integer scoring resolves the conflict
  between worked-steps arithmetic data and reward scoring — the design
  flaw behind arith's 0.20 plateau.
- Pinning special-token ids identically across tokenizers (0–4) made
  the entire chat/RL stack encoding-agnostic almost for free.
- macOS framework Python's missing CA certs bit a third time (gcsfs);
  `SSL_CERT_FILE` + certifi is the once-and-for-all wrapper fix.

**Next steps.** Launch generation-2 with `scripts/pipeline.sh`
(kimi3/bpe4k, seq 2048, 25k/9k/600 steps, GRPO lr 1e-5) — on Spot with
the boot-resume hook. GPU quota window (1→3) opens 2026-08-09 ~19:20 UTC;
add medium-BPE at Chinchilla in parallel if granted.

## 2026-08-10 — Generation-2 run: pretrain −20% bpb; arith cold start failed generatively

**Goal.** First full generation on the new stack (bpe4k, seq 2048,
enriched mix, worked-steps arith, batched GRPO @ lr 1e-5) via
`scripts/pipeline.sh`; beat gen-1 (scarlet-harbor) on byte-normalized
test bpb and task rewards.

**Setup.** One command on the L4 (on-demand). Launch found a recipe
bug: batch 16 @ seq 2048 fp32 OOMs the 22 GiB L4 on step 1. Recipe
fixed to batch 12 with the token budget held (~819M: 33,333 steps;
SFT 12,000 × b12 = same examples as 9,000 × b16), batch now forwarded
on resume paths, `expandable_segments` on (commit b5c0494). A 30-step
probe measured peak 21.2/22.5 GiB, 11.3k tok/s before relaunch.
Runs: `kimi3-19M-golden-dell-20260808-204157` → sft → rlvr.
Wall-clock: pretrain 20.4h, SFT 7.6h, GRPO 2.7h (600 steps — 2.2×
faster per step than gen-1 thanks to rollout batching + shorter BPE
sequences), evals 3m. GPU held 86–100% at 20.5–21.7 GiB throughout.

**Results.** (enwik8 test slice, bpb normalized by bytes — the
cross-tokenizer axis)

| checkpoint | test bpb | gen-1 |
|---|---|---|
| pretrain best (step 31,400) | **1.247** | 1.560 |
| sft best | 1.694 | — |
| rlvr best | 1.702 | — |

- **Pretrain −0.31 bpb (−20%)** — tokenizer + code/math mix + long
  context + 2× Chinchilla, compounded. (Gen-1 medium/large sat at
  1.93/2.00; a 19M model now beats them by half a bit.)
- SFT val (bits/assistant-token) 2.762, still every-eval-best at
  12k steps; `tasks_bpb` 1.040 — the worked-steps corpus was learned
  *as data*.
- GRPO eval reward **0.700 from step 40 and frozen thereafter**
  (gen-1: 0.683): copy 1.00, parity **1.00** (gen-1 oscillated
  0.83↔0.17 — the lr halving cured it), words **1.00** (was 0.67),
  count 0.75 (unchanged), **arith 0.00** (was 0.20).
- Forgetting check: SFT costs +0.45 bpb on raw text; GRPO costs
  +0.008 — the k3-KL leash preserves the LM almost exactly.

**Learnings.**
- **Teacher-forced mastery ≠ generative behavior.** tasks_bpb 1.04
  says the model models worked-steps text; greedy probes say it never
  *produces* it: "What is 47 + 12?" → "12" (sft and rlvr both emit a
  short wrong integer; rollout len 5–7 tokens). The ~12% task share
  lost the first-token branch to the short-answer chat prior.
- With all-wrong arith groups, z-scored group advantage is
  identically zero — RL cannot bootstrap a behavior that never
  appears in rollouts (dead_frac ~50% all run ≈ all-zero arith
  groups + saturated-perfect groups). Gen-1's 0.20 came from *short*
  canonical answers matching the generative prior, not from better
  math.
- Eval saturation: 3 of 5 tasks at 1.00 by step 40 → 560 further
  steps only drifted KL (0.40 at end, grad 4–5). Next run needs
  either a harder eval set or early stop on frozen eval.
- Batch 16 @ 2048 fp32 never fit the L4; the toy chain validated
  correctness, not memory. Probe memory at full shape before any
  recipe change.

**Next steps.** (a) Fix arith cold start *generatively*: raise task
share late in SFT (anneal) or add a short task-only SFT tail; and/or
give GRPO partial format credit (e.g. reward containing "=") so rare
worked-steps rollouts get nonzero advantage. (b) Quota: both GPU
preferences denied again 2026-08-10 — medium-BPE stays blocked;
options are later resubmission, sequential medium on the existing L4
(~4 days), or another region/GPU. (c) flora now serves the gen-2
chain locally (worker auto-discovers newest best.pt per stage).
