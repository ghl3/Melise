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

## 2026-08-10 — Gen-3 prep: chat-retargeted data, bpe8k, preamble, Spot infra

Recipe and launch procedure: **docs/runs/gen3-medium.md** (new
convention — run docs carry the config; this entry carries the story).

**Goal.** Skip gen-2.5 (its fixes ride along) and prepare generation-3
— kimi3-medium on a casual-chat-retargeted stack — to launch-ready,
without launching.

**Setup / work done.**
- **Chat product** (context for the data pivot): flora→lily site,
  saved chats, worker+UI shipped; a live chat battery against gen-2
  exposed the failure modes that set this session's agenda (no
  identity, no perspective flip, template-locked skills, register).
- **Data overhaul** (docs/DATASETS.md): SmolTalk casual-chat filter
  (−47%: 8% tool-calling, 16% summarize/rewrite pipelines, 24%
  oversized); OASST1 all-English-paths 3,670→20,505 convs; identity
  corpus (persona/no-clock honesty, varied names); recall task family
  (person-deixis flip); arith 0.25 format credit; 3–4 prompt
  paraphrases per family; RL headroom weights (arith 2.0…copy 0.5,
  eval uniform); +fineweb-edu 400 MB (HTTP range reads — 242 MB
  transferred of a 2.15 GB shard) and Cornell+DailyDialog dialogue
  corpora; math ×0.05. New tools: sample_data.py, filter_chat_data.py.
- **bpe8k** (8192, digit-isolated, specials pinned): 8–12% fewer
  tokens/byte than bpe4k (enwik8 2.95, fineweb 3.68 B/tok). User
  call: take the compression, attribution be damned.
- **Metrics**: every logged bpb is now bits-per-RAW-BYTE via exact
  per-id byte tables (init sanity: 4.45 bpb = uniform over 8192);
  train entropy_bits; per-domain pretrain val (enwik8/fineweb/
  dialogue/books); per-source SFT val (val_source/*).
- **Preamble**: plain text before the first marker (no new control
  byte — reuses pretrained language circuitry, no tokenizer re-pin);
  varied names in identity/task data so identity is read, not
  memorized; GRPO rollouts + serve --preamble carry the deployed
  string.
- **Infra**: VM recreated as Spot on its boot disk; b5 probe
  20.1/22.5 GiB @ 4,098 tok/s (b6 OOMs); boot-resume crontab;
  kimi3-spot-restart scheduler (paused); DONE_CMD self-cleanup;
  PT cadence 250/500; git bundle refreshed (was 16 commits stale).

**Results.** Recipe frozen: **72.0M measured** @ bpe8k, 145k steps ×
b5 ≈ 1.48B tokens (20.6 tok/param), mix-gen3-chat, SFT 20k + tail
1.5k, GRPO 600 weighted — est. ~5.3 days, ~$40 Spot. Pre-launch
validation: three toy full-pipeline iterations on the VM, each caught
a real bug — (1) the boot-resume crontab fires on EVERY boot and
resumed stale gen-2 state alongside the toy (install it only after
launch is stepping); (2) bare `--resume` fell back to small-recipe
defaults, which would have made an interrupted 145k-step pretrain
look finished and started SFT on an unfinished base (env now baked
into the crontab via printf %q); (3) SFT eval cadence wasn't
tunable and best.pt only lands at evals (SFT_EVAL_EVERY). Third
iteration fully green: every stage incl. tail and GRPO
(recall/weighted sampling/preamble live in logs), evals, DONE_CMD.
Peak 21.3/22.5 GiB (GRPO is the memory high-water mark, not
pretrain). bpe8k token cache pre-warmed.

**Learnings.**
- A 500-conversation sampling audit beats any amount of dataset-card
  reading — one random draw found the tool-calling subset.
- Compute-optimal sizing said d=384/74M for a ~4.5-day L4 budget;
  d=512 (~125M) needs ~7+ days to be worth it (declined for now).
- Fetch gotchas for the record: yanran.li is bot-walled (use the
  roskoN HF mirror); pyarrow probes .closed as an attribute; raw
  urllib needs certifi's SSL context on macOS; importlib-loaded
  modules must be registered in sys.modules before dataclasses work.

**Next steps.** Launch per the run doc (unpause restarter → DONE_CMD
pipeline command). During: watch val_domain/dialogue+fineweb,
val_source/identity, dead_frac; daily laptop eval_checkpoint on
best.pt. After: results into the run doc, chat battery vs gen-2,
model naming (Wren/Fern/Willa/Flora reserved), Cloud Run + Vercel
deploys.

## 2026-08-17 — Gen-3 (Melise) complete: −12% test bpb, eviction anatomy, probe suite

**Goal.** Land the full gen-3 run (145k pretrain → SFT → tail → GRPO →
evals), monitor it to completion, test the product locally, and turn
what testing revealed into a measurement suite for gen-4.

**Setup.** kimi3-medium 72.1M (d=384, 13 attn layers, 24 experts
top-4, 45.6M active/token), bpe8k, ctx 2048, `mix-gen3-chat`, 145k
steps × b5 ≈ 1.485B tokens (20.6 tok/param). L4 Spot at launch
(2026-08-11 00:10 UTC); converted Spot→on-demand IN PLACE at step
~37k after 12 preemptions in 38h (set-scheduling --no-preemptible
--provisioning-model=STANDARD --clear-instance-termination-action;
~10 min downtime, zero steps lost). 4.0–4.1k tok/s throughout. SFT
20k × b5, tail 1.5k, GRPO 600.

**Results.**
- **enwik8 test bpb 1.096** (best.pt @126.5k) vs gen-2's 1.247 —
  **−12.1%**, the only cross-gen-comparable number. Sampled val 1.19
  at 145k end.
- Domains: dialogue 1.55→1.37, fineweb 1.56→1.25, enwik8 →1.17 —
  and **war_and_peace floor 1.190 @52k → 1.532 @145k (+0.34)**, the
  climb ACCELERATING through LR decay. Annealing amplifies eviction:
  the mixture-optimal point actively trades a 2%-share domain away.
  SFT miniature: dolly rose 1.02→1.14 then recovered to ~1.07
  (register overlap saved it; Tolstoy had no such luck).
- SFT: assistant-byte val 0.784; identity source bpb floored at 0.06
  by step 14k. GRPO uniform eval: **arith 0.25 (gen-2: 0.00), count
  1.00 (0.75), recall 1.00 (new), copy/parity/words 1.00 held**;
  late dead_frac ~0.8 = all-correct saturation.
- Forgetting evals: enwik8 test 1.096 → 1.839 after chat-only SFT
  (+0.74), → 1.842 after GRPO (+0.003 — GRPO is essentially free).
- Serve test (local serve.py + web UI, MPS): pool-name identity
  perfect and consistent; **"You are Melise" → "Leo"** (novel name,
  fails 3/3; name drifts per reply: Leo→Ivan). Date honesty
  excellent ("no clock or calendar"). Basic facts ~0 ("Name an
  animal" → "An animal"). Sonnet 18 NOT memorized even by the base
  model given a 2-line verbatim key — pastiche in meter instead.
- **Probe suite built** (transformer/probes.py, configs/facts.json
  61 entries, scripts/probe_checkpoint.py; 12/12 tests). First
  cross-stage run (--quick): identity/novel **0.67 pretrain → 0.00
  rlvr** — pretrain HAS context-copying of unseen names and the
  24-name SFT corpus TRAINED IT AWAY. facts/instances cloze 0.67 →
  chat 0.00 (content exists weakly; on-demand access doesn't).
  Verbatim trained-vs-heldout gap ≈ 0 at both stages.
- Incidents: (1) after pretrain finished, stage_done() saw gen-2's
  sft/rlvr dirs as newest-per-stage and SKIPPED all gen-3
  post-training; DONE_CMD's scheduler-pause failed silently
  (ACCESS_TOKEN_SCOPE_INSUFFICIENT — VM scopes lack cloudscheduler)
  while its shutdown worked → 15h boot→evals→shutdown loop against
  the enabled restarter (~$9). Recovery: pause restarter
  laptop-side, `--post-only` relaunch (forces fresh chain — no
  deletions needed, gen-2 artifacts untouched). (2) evals.jsonl
  never syncs (BucketSync's last kick precedes the eval append) —
  retrieved manually through an L4-stockout window. (3) best.pt
  (val_loss draw noise) stale from 22.25k to 66k while SFT inits
  from it — self-resolved in the decay, but a run ending earlier
  would have fine-tuned a 15%-trained base. (4) Two zone-wide L4
  stockouts (one at the provisioning switch, one post-run). (5) The
  repo move broke .venv shebangs (fixed by sed).
- Wall-clock: launch→PIPELINE_DONE 6.25 days (incl. preemption era +
  16h incident); on-demand portion ran 4.5 days without a single
  interruption.

**Learnings.**
1. **Small-share domains don't plateau under pressure — they get
   evicted, and LR decay makes it worse.** The single best
   instrumentation decision of the run was giving one book a val
   slice; gen-4 groups corpora and instruments the group.
2. **72M stores form, not content.** Register everywhere (play vs
   Victorian novel competing for the same prompt), zero verbatim
   retention (heldout gap ≈ 0 despite ~28 effective epochs/book),
   zero basic facts. The gen-4 capacity bet (3× routed storage)
   now has a precise success metric: probe/facts/*/heldout off the
   floor.
3. **SFT narrows more than it teaches**: +0.74 bpb general-text
   forgetting AND destruction of pretrain's novel-name induction.
   Fixes queued: 2–5% pretrain replay in SFT batches; identity
   corpus v2 (hundreds of names — preservation, not enforcement);
   context_recall RLVR family (retrieve name/date when given,
   refuse honestly when not).
4. **bpb alone diagnoses slowly.** The eviction took days to
   understand; accuracy/entropy triples + behavior probes at every
   eval make the same stories legible in realtime. The whole probe
   suite costs ~2–3% overhead.
5. Ops: stage_done needs a recency guard (stale dirs from ANY
   earlier era are landmines); VM-side DONE_CMD can never pause the
   scheduler (scopes) — leave the restarter paused on-demand;
   offline evals must push their own results; keeper checkpoints
   every ~25k (pruning destroyed the floor-era checkpoint the
   eviction anatomy needed); pre-launch toy dirs joined gen-2 dirs
   as resume hazards.
6. On-demand was the right call: the Spot discount is thin for L4
   now, and the uninterrupted tail ran exactly as scheduled.

**Next steps.** Gen-4 proposal v1 frozen-in-draft
(`gen4-ideas.md`): d=512 + 40 experts (163.2M/78.3M active), ~2.2B
tokens (~13d on-demand, probe memory first), grouped corpora (books
merged at 15%, fineweb → ~2GB with seeded row-group shuffle across
shards — gen-3's slice was 3 crawl dumps, nothing post-Jan-2020),
probe suite completion per `gen4-eval-suite.md`. Gen-3 ritual
remaining: identity-tail rerun decision (Melise name fix, ~1.5h
GPU), run-doc Results fill, memory updates, formal chat battery vs
gen-2, meliseai.com swap (rlvr best.pt + bpe8k tokenizer in image +
preamble). Deferred still deferred: bpe16k, DPO.

## 2026-08-17 (evening) — Gen-4 build-out: instruments, corpora, recipe; gen-3 live on meliseai.com

**Goal.** Execute the full gen-4 preparation list in one pass — every
eval/probe/pipeline/data change the gen-3 post-mortem called for —
plus the user-approved decisions: WSD schedule, 2.2B budget, gen-3 to
production, SFT held at 20k, new public dialogue corpora, chat-facts
into both SFT and GRPO.

**Setup / work done.**
- **Preset**: `kimi3-medium-wide` (d=512, 3 blocks, 40 experts top-4)
  — 163.2M / 94.4M routed / 78.3M active, matching the proposal's
  counted numbers exactly; test-pinned.
- **Eval overhaul**: deterministic fixed-window per-domain evals
  (pinned seeded windows; zero draw noise, equal coverage) with
  accuracy/top-5/entropy triples; byte-weighted `val_group/*`;
  best.pt = min trailing-3 val_bpb (resume-safe via checkpoint
  `val_bpb_hist`); keeper checkpoints every 25k exempt from pruning;
  BPE corpora stored on-device as int16 (~2 GB freed at gen-4 scale).
- **WSD** in `lr_at` (hold + linear decay over final 15%): extending
  --steps mid-hold is surgery-free; decay's eviction cost is paid at
  a chosen time.
- **Probe suite completed + wired** into all three loops
  (`--probe-every`; TB `probe/*`, metrics probe/probe_dump events;
  rotating dumps greedy+t0.8 via a private CPU RNG — training
  streams untouched). facts.json 58 → 298 with reject-lists (the
  forced-choice echo slack — "hot or cold" — closed before a facts
  REWARD could farm it); `probe/task_formats`; chat-wrapped verbatim.
- **Identity v2**: `transformer/identity.py` — ~370-name train pool
  (Melise ~8%, never dominant), 16 held-out names + the novel-name
  space reserved by IMPORT-TIME asserts (they caught a real collision
  during authoring: Bram). gen_identity_sft v2 renders dated
  preambles + retrieval turns + refusal preservation + ask-twice.
- **RLVR**: `context_recall` (name/date/no-date variants; randomized
  per-task preambles now plumbed through rollouts AND eval) and
  `facts` (train split only) joined the weighted rotation.
- **SFT replay**: `--replay-frac 0.03` interleaves raw pretrain-mix
  batches (CPU-resident, zero GPU cost) — the +0.74 forgetting fix;
  `train/replay_bpb` logged every replay step.
- **Pipeline hardening**: `chain_ok` lineage guard (downstream dirs
  must name the current upstream run — kills the gen-2-dirs AND
  toy-dirs incident classes), DONE_CMD shutdown-only, evals.jsonl +
  probes.jsonl self-push, holdout-book evals (emma/great_expectations
  as 100%-test) + full probe battery in the post-stage evals,
  gen-4 defaults + env knobs. rebuild_tb/recover_metrics replay all
  new events (found 2 pre-existing recover_metrics regexes broken
  since gen-3 — ent=/dead= fields — fixed).
- **Data**: prep_fineweb multi-shard seeded row-group shuffle (9,673
  row groups / 14 shards indexed; smoke draw hit 2015+2021 dumps);
  2GB fetch running. PersonaChat (10.9k convs, 8.3 MB, detokenized)
  + BlendedSkillTalk (6.8k convs, 6.4 MB) → dialogue 7% (fineweb
  44%), dialogue_persona val canary. chat_identity/chat_tasks/
  chat_facts regenerated + uploaded.
- **Serving**: DEFAULT_PREAMBLE → Melise; serve.py renders `{date}`
  per request (gen-4+ only); SERVE_PREAMBLE env; Dockerfile +
  .dockerignore now ship ALL tokenizer configs (the bpe8k crash
  footgun was enforced in TWO files).

**Results.**
- 41/41 tests (15+16+10), including new coverage: preset params,
  fixed-window determinism + byte-weighted aggregation, WSD shape +
  extension semantics, keeper pruning, windowed best, group
  membership, context_recall scoring (copy/refuse/anti-hack), facts
  train-split-only.
- Smoke runs of all three loops on MPS: pretrain 20 steps (WSD +
  fixed windows + `grp: books=…` + 29 probe scalars + keeper), SFT 6
  steps (replay interleave + 46 chat-form scalars), GRPO 2 steps
  (context_recall in rollouts with its own preamble; eval clean).
- **Gen-3 is live on meliseai.com** (`melise-worker` rev 00002):
  stripped rlvr best.pt (276 MB), Melise preamble via SERVE_PREAMBLE
  (verified applied: 19 prompt tokens vs 9 bare). t=0 "What is your
  name?" → "I'm called Leo." — the accepted pool-name limitation,
  gen-4's to fix. CPU decode ≈ 4.5 tok/s (gen-4 at 78M active
  projects ~2.5–3 — swap-time decision: --cpu 4 / bf16 / accept).
- Probe-round cost on MPS: ~400 s raw / ~840 s chat on an UNTRAINED
  model (every generation runs to max_new). CUDA + trained-model
  early stopping should cut this ~10×; the hardware probe must time
  a real round and keep in-loop overhead ≤3% (adjust *_PROBE_EVERY).

**Learnings.**
- The import-time assert pattern (reserved name strata) earns its
  keep immediately — it refused my own authoring mistake minutes
  after being written.
- Closed-world scoring needs an explicit reject dimension the moment
  it becomes a REWARD: contains-matching alone gives full credit for
  echoing a forced choice, and GRPO farms any slack it finds.
- The disaster-recovery parsers (recover_metrics) had silently rotted
  as print formats evolved — schema-drift tests or shared format
  constants would have caught it; for now they're fixed and the
  rebuilders replay every event type the loops emit.

**Next steps.** (1) fineweb 2GB upload once the fetch lands →
`download_data.py` on the VM → token-cache pre-warm. (2) VM window:
hardware probe (b4/b5 tok/s + peak GiB + GRPO memory + CUDA probe
timing) → freeze PT_STEPS in `gen4-medium-wide.md`; cross-gen probe
baselines on gen-2/gen-3 bests; gen-3 homework (routing map, eviction
anatomy, lm_head SVD) on the idle GPU. (3) Toy full-pipeline
validation incl. planted-stale-dir refusal → DELETE toy dirs →
launch. Non-blocking gen-3 ritual: run-doc Results fill, chat battery
vs gen-2.

*Addendum (same day, after review):* fineweb 2.0 GB / 93 dumps
uploaded and registered. Recipe review closed two questions: (a)
**batch → b5/215k steps** as the plan (int16 headroom + ~2 days of
wall-clock; probe must confirm peak ≤ ~21.5 GiB and a clean
500–1,000-step LR-stability segment at 2.5e-4, else b4/268.5k —
batch and steps freeze together); (b) **ctx stays 2048** — no
measured failure implicates the window, memory is spent on batch,
and NoPE+KDA make context an SFT-time property (gen-1 measured
256→1024 extension clean), so the option stays open for ~a day of
GPU if serve traffic ever shows dropped turns. Probe gate also
gained: VM disk check (gen-3 run dirs archived off disk first),
probe-cost contingency order (raise SFT_PROBE_EVERY, then trim
verbatim), git-bundle refresh. Commits 23c9258 + e0251ec.

## 2026-08-17 (night) — Gen-4 probe session: recipe frozen at b4, toy pipeline green, launch-ready

**Goal.** Exercise every part of the gen-4 run on the VM before
committing 12 days to it: memory/throughput at the exact config, LR
stability, probe cost, GRPO memory, the full toy pipeline, and the
chain-guard incident test. Freeze the recipe on measured numbers.

**Setup.** VM up after a 19-min L4 stockout (retry loop); gen-3's
stale boot-resume crontab defused ~35 s after boot. Code
checksum-synced at 0d46706-era; 2 GB fineweb + all new corpora pulled;
disk 119 G free (gen-3 dirs kept for post-run analyses).

**Results.**
- **b5 pretrain OOM'd on step 1** (AttnRes forward, 21.6 GiB before
  optimizer state existed). **b4 OOM'd on step 2** — the moment Adam's
  1.3 GB materialized. Fix: pretrain corpora now CPU-resident (the
  GPU never needed them — ~130 KB per-batch copies; 2.1 GiB freed).
  After the fix **b4 passed 1,000 steps clean**: 2,500–2,830 tok/s,
  grad_norm 0.61–1.41 at 2.5e-4 peak (WSD), val_bpb 2.30→1.93,
  all domain/group curves descending, moe_max_load 0.06–0.07 at 40
  experts. **Recipe frozen: PT b4 × 268,500.**
- **SFT b5 OOM'd in its first backward** (no resident corpus — b5
  activations outweigh the 2 GB saved). **SFT frozen: b4 × 25,000**
  (gen-3's example exposure).
- **GRPO peak 10.9 GiB** at production rollout shape — pretrain is
  now the memory high-water mark, inverting gen-3's rule of thumb.
- **Probe round: 266 s on CUDA** (untrained model = worst case).
  Cadences retuned 2000→4000 / 800→2500 / 100→200 + in-loop verbatim
  1/file: ~2% of pretrain, ~4–5% of SFT.
- **Toy full pipeline: PIPELINE_DONE** — post-only reuse, SFT b4 with
  replay + chat probes, tail, GRPO with all 8 task families, enwik8
  evals, both holdout books, full offline probe battery ×3, artifacts
  self-pushed to the bucket. **chain_ok refusal test passed**: gen-3's
  sft dir planted as newest → --resume started a fresh SFT from the
  current pretrain instead of trusting it.
- Wall-clock from measured tok/s: **~12 days, ~$245** end-to-end.

**Learnings.**
1. "Probe memory at full shape" (gen-2's lesson) earned its keep
   twice in one evening: THREE separate OOMs (pretrain b5, pretrain
   b4-with-resident-corpus, SFT b5) that would each have been a
   launch-day incident, all caught in ~$5 of GPU time.
2. Resident-corpus-on-GPU was a silent architecture tax nobody had
   re-examined since the 17M era — at 163M it was the difference
   between fitting and not. Assumptions scale worse than models.
3. Toy validation caught two more real defects: short runs (< one
   eval interval) produced no best.pt and aborted the chain (sft.py
   now falls back to the final checkpoint), and stage exit codes were
   read from the `newest()` call, so a crashed SFT logged "exit 0".
   Every toy-validation round so far (gen-3's and both of gen-4's)
   has found real bugs — the ritual stays.
4. The pkill-over-ssh footgun bit AGAIN despite the bracket trick
   (the launch text "pipeline.sh" in the same command line matched) —
   keep kill commands in a separate ssh from anything naming the
   scripts.

**Next steps.** Cross-gen probe baselines finishing on the VM
(gen-2 + gen-3, full battery → probes.jsonl in the bucket). Then the
user's launch call: VM is RUNNING, warm, and clean — launch is the
env block in gen4-medium-wide.md + boot-resume install + verification.
Non-blocking afterwards: gen-3 run-doc Results fill, chat battery,
routing map / eviction anatomy / lm_head SVD on the idle GPU windows.
