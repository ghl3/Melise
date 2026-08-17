# Gen-4 — planning & idea collection (OPEN — not a recipe)

Opened 2026-08-14, mid-gen-3 (step ~103k of 145k). A place to collect
observations, options, and half-decisions while gen-3 is still
teaching us things. Nothing here is frozen; promote the winners into
`docs/runs/gen4-<preset>.md` when the recipe is set. Add freely.

## Where gen-3 stands (snapshot at opening)

Healthy, 71% through pretrain, ETA Aug 15 ~19:00 UTC → pipeline done
~Aug 16. Aggregate val ~1.20 (windowed), already at gen-2's 1.247
test bar with the decay tail still to come. Thesis domains (dialogue,
fineweb-edu) improving throughout. VM converted Spot → on-demand
mid-run at step ~37k (Aug 12) after 12 preemptions in 38h; zero
interruptions since. Full results will land in `gen3-medium.md`.

## Evidence gathered during gen-3

Each of these is a measured observation, not a hunch. Details in
NOTEBOOK.md entries for 2026-08-12..14.

1. **Small-domain capacity eviction (the war_and_peace canary).**
   Only book with a val slice. Hit floor 1.190 bpb ~step 52k, then
   regressed steadily to 1.289 by 103k (+0.10) — the climb did NOT
   decelerate as LR halved 1.56e-4 → 8.3e-5, which falsifies
   "high-LR churn" and confirms genuine reallocation toward big
   domains. Register survives (samples still fluently Victorian);
   book-specific structure is what's evicted (invented casts,
   cross-book character blending). Same dynamics presumably apply to
   every low-share corpus — uninstrumented, so unmeasured.
   PENDING: does the final-15k cosine plunge recover any of it; and
   the offline forgetting evals' end-state numbers.
2. **best.pt criterion is unsafe.** It tracks sampled val_loss,
   whose mixture-draw noise (±0.04 bpb per eval) let one lucky draw
   at step 22,250 hold "best" for 53k steps — while `pipeline.sh`
   inits SFT from `best.pt`. Self-resolved when the decay tail beat
   the outlier, but a run that ended earlier would have fine-tuned a
   15%-trained base. Fix ideas below.
3. **Per-domain eval coverage is wildly uneven.** Each domain eval
   scores ~20 random 2048-tok windows: war_and_peace's 33KB val
   slice gets ~4× oversampled (near-deterministic, smooth curve);
   fineweb's 2MB gets ~7% coverage (noisy). Curve smoothness is a
   measurement artifact, not a training property — cost us several
   head-scratches.
4. **The sample log is a keyhole.** One prompt (`ROMEO:\n`), greedy
   decoding, every 100 steps. Generation quality per register is
   invisible; register competition only showed up as argmax flips
   between checkpoints.
5. **Spot economics may have shifted.** 12 preemptions in the first
   38h (~67–88% uptime, ~1–2h/day redone work); one tracker prices
   L4 Spot at only ~17% under on-demand now. The mid-run conversion
   (stop → `set-scheduling --no-preemptible
   --provisioning-model=STANDARD --clear-instance-termination-action
   --restart-on-failure` → start) worked in place, ~10 min downtime.
   Verify the real Spot rate on the August invoice before deciding
   gen-4's provisioning.
6. **metrics.jsonl can contain NUL-filled torn lines** (3 found,
   from preemption-kill mid-writes). All parsers must skip
   non-JSON lines. Consider fsync-on-eval or a repair pass.
7. **Keep pile (worked as designed):** bpe8k tokenizer; fineweb-edu
   400MB expansion; per-domain val instrumentation (the canary was
   this run's best diagnostic); dialogue corpora (register visibly
   installed); pipeline self-healing (every one of ~14 recoveries
   clean); pre-launch toy-pipeline validation (caught 3 real bugs).

## Decision axes

### 1. Capacity / architecture — the headline question

The eviction result says 72M is saturated for this mix. Options:

| option | cost | what it buys | caveats |
|---|---|---|---|
| **d=512** (deferred gen-4 candidate) | ~1.8× params & FLOPs; slower wall-clock or fewer steps | relieves interference everywhere incl. shared pathway (attention + residual), where register competition lives | expensive; L4 batch/memory must be re-probed; helps eviction only diffusely |
| **more routed experts** 24→32 | ≈constant FLOPs; +params (optimizer mem only) | targets FFN storage — exactly where long-tail eviction bites; cheapest capacity per wall-clock | each expert's data diet thins (top-4 of 32 on ~1.5B tok); doesn't touch shared-pathway interference |
| **both, modestly** | compounding | attacks both interference sites | budget; two variables at once muddies attribution |
| **neither — fix the mix instead** | free | if eviction is confined to decorative domains, weights are the knob | concedes the capacity ceiling for future ambitions |

Ranking for eviction specifically (rationale in NOTEBOOK 2026-08-13):
experts (modest bump) > width > fatter experts > depth; raising top_k
is the wrong direction (more blending = more sharing). The clean
experiment if budget allows: **d=512 vs 24→32 experts at matched
param count, judged on the war_and_peace curve** — we now know
exactly what its failure mode looks like at 72M/d=384.

L4 constraints to re-probe for any change: gen-3 medium ran b5 @
20.1/22.5 GiB, 4.1k tok/s; GRPO is the memory high-water mark.

### 2. Data mix

- **Grouped corpora: scale registers, not files.** Today every ×1.0 book rides at its byte-size share —
  the ~30 books implicitly claim ~28% of the gradient at ~28
  effective epochs each, an allocation nobody chose. Gen-4: mix
  config gains *groups* (glob + one computed multiplier hitting a
  target share; keep files separate — do NOT concatenate, per-file
  val splits break). Proposed groups: **fiction** (~22 novels + 10–20
  new Gutenberg acquisitions incl. some 20th-c. public domain),
  **nonfiction** (Darwin/Smith/Descartes/…), **drama**
  (Shakespeare×2 — the play register; scale near dialogue), plus the
  existing majors (fineweb, wikitext, enwik8, dialogue, code, math).
  Adding books then dilutes per-book epochs at fixed share instead of
  silently growing fiction. Lean: fiction ~15% (→ ~14 epochs at 2.2B
  on a ~75MB group, vs 27 today); freeze the number with the
  forgetting evals in hand. Instrumentation: group-level val slices
  (one honest val_domain/fiction curve), war_and_peace keeps its
  individual slice for cross-gen continuity, 2 new books held out of
  training entirely, memorization gap per group. Needs the small
  load_data_mix group extension — pre-launch code.

  **STATUS 2026-08-14: implemented.** `load_data_mix` supports
  `groups` (glob + `share`; solved to per-file multipliers; tested in
  tests/test_transformer.py). 16 books downloaded + registered
  (dracula, jane-eyre, wuthering-heights, dorian-gray, time-machine,
  war-of-the-worlds, tom-sawyer, call-of-the-wild, heart-of-darkness,
  crime-and-punishment, madame-bovary, age-of-innocence, little-women,
  great-gatsby, + holdouts emma, great-expectations). Draft config:
  `configs/mix-gen4-chat.json` — validated, shares exact:

  | group | MB | gen-3 share | gen-4 target | epochs @2.2B |
  |---|---|---|---|---|
  | fineweb | 402→~2000 | 16.8% | **46%** | 1.6 (after expansion; 8.1 before) |
  | wikitext | 541 | 32.2% | **16%** | 2.1 |
  | enwik8 | 100 | 6.0% | **5%** | 3.5 |
  | dialogue | 24 | 4.3% | **5%** | 14.7 (→8% only if corpus grows) |
  | books (45 works: fiction+nonfiction+drama) | 54 | 27.2% | **15%** | 19.9 (was ~28) |
  | code | 100 | 8.9% | **8%** | 5.7 |
  | math | 101 | 3.0% | **4%** | 2.8 |
  | reference (webster) | 28 | 1.7% | **1%** | 2.5 |

  Fiction val canaries: war_and_peace (continuity), sherlock,
  moby_dick, jane_eyre (a NEW book). Shares are draft — freeze after
  the forgetting evals + fineweb expansion land.
- **Trim breadth?** For a chat product: does `code_python` (×0.15)
  earn its capacity? `math_openweb` (×0.05)? `webster_dictionary`?
  Every dropped MB is capacity handed to thesis domains — the
  eviction result says this trade is real, not theoretical.
- **Protect small domains** (if we keep them): minimum-share floors,
  or a late-run replay bump once LR is low (cheap to trial: the
  mixture is an env knob).
- **Instrument more books**: val slices on 2–3 books across the size
  range (alice 150KB / sherlock 580KB / moby_dick 1.2MB) → turns
  "forgetting ∝ share?" into a measured curve. Config-only change.
- **More dialogue?** It's the product register and it improved
  steadily without saturating — consider a bigger share.
- Keep fineweb-edu at 400MB scale; it was the engine of this run.
- **Fineweb prefix bias (measured 2026-08-14):** prep_fineweb.py takes
  a sequential prefix of sample-10BT shard 0, and the shard is
  dump-CLUSTERED (each 1000-doc row group = one CommonCrawl snapshot,
  a few dumps interleaved in rotation). Gen-3's 402MB therefore
  contains exactly 3 of ~95 snapshots: CC-MAIN-2013-20, 2017-26,
  2020-05 — balanced between them, nothing newer than Jan 2020 (note
  for the gen-3 results doc: her web knowledge is pre-pandemic).
  Register impact ~nil; content-recency ceiling real. Gen-4 fix (with
  the 2GB multi-shard expansion): draw row groups in seeded-random
  order across all 14 shards so every dump is represented.

### 3. Eval / metrics / tooling

- **best.pt: track val_bpb, not val_loss** — and consider requiring
  a windowed mean (e.g. best 3-eval average) so no single draw can
  hold the title. Small change in pretrain.py, big safety win.
- **Deterministic val**: score a fixed set of windows per domain
  every eval (like `slice_nll` over a pinned subset) instead of
  fresh random draws — kills draw noise, makes evals comparable
  step-to-step, equalizes coverage. Costs nothing at val sizes.
- **Multi-prompt sample battery**: one fixed prompt per register
  (play, novel, wiki, dialogue, code if kept) + a temperature-sampled
  variant alongside greedy. `--sample-prompt` → list.
- **Expert×domain routing map** ("do experts get dialects?"): push
  held-out text per domain through best.pt, log top-4 routing per
  layer, heat-map it. Run on the VM post-pipeline (gen-3 homework
  that informs the gen-4 architecture choice).
- **lm_head SVD across checkpoints**: singular-value spectrum of the
  8192×d unembedding over saved step_*.pt — is effective head rank
  flattening late in the run? (Softmax-bottleneck check, Godey et
  al.; see docs/papers/. Cheap CPU analysis.)
- **Fully-held-out books**: exclude 1–2 books from training entirely
  as pure test domains — uncontaminated register generalization,
  free of the book-familiarity flattery that within-book val slices
  carry. (Context: mixture math gives every ×1.0 book ~28 effective
  epochs/gen — partial verbatim memorization of small corpora is
  expected, ~3.6 bits/param ≈ ~30MB capacity at 72M.)
- **Memorization gap**: eval_checkpoint the same checkpoint on a
  book's train slice vs val slice; the gap is a per-domain
  memorization index. Track it gen-3 → gen-4 to see whether added
  capacity buys generalization or recitation. VM, nearly free.
- **Eviction anatomy** (post-gen-3, VM): per-token p_true/rank/entropy
  on the war val slice, bucketed by token class (rare/names vs
  common), for best.pt (126.5k) vs final — does the decay-phase climb
  come from rank loss or from sharpened-but-generic mass (confidence
  misallocation)? Prediction: mostly the latter.
  Supporting probe (2026-08-17, laptop): pretrain best.pt cannot
  recite Sonnet 18 even given a 2-line verbatim key at t=0 — emits
  metrically-correct invented verse instead. ~28 effective epochs
  stored register, not strings. Expect the memorization gap to be
  SMALL; "name an animal"-type instance retrieval fails in chat for
  the same form-over-content reason.
- **Keeper checkpoints**: gen-3's pruning left no floor-era (~52k)
  checkpoint, killing the clean before/after eviction diff. Gen-4:
  keep a sparse milestone checkpoint every ~25k steps exempt from
  pruning (few GB in the bucket buys post-hoc science).

### 4. Tokenizer

bpe16k — **deferred by user decision, do not start.** Trade: fewer
tokens/byte (throughput, effective context) vs embedding table's
param share at this scale. Revisit only with d=512-class capacity.

### 5. Post-training

- DPO — **deferred by user decision, do not start.**
- GRPO task set: recall was new in gen-3 (results pending); collect
  candidates for gen-4 once the per-task numbers land.
- Identity: gen-3 trained with varied-name preambles, serves with
  `--preamble "You are Melise..."`. DEFAULT_PREAMBLE Lily→Melise
  rename is a post-gen-3 decision. Gen-4 could bake Melise in as the
  dominant training name — decide after seeing gen-3's identity evals.
  **MEASURED 2026-08-17 (local serve test):** preamble-reading works
  for in-distribution names ("You are Lily" → "Lily", 3/3 at t=0.2)
  but does NOT generalize to the unseen name Melise ("You are
  Melise" → "Leo", 3/3). Name-copying is pool-limited at 72M. Gen-3
  fix options considered: regen identity data + SFT tail rerun, or
  gen-4 identity v2. DECIDED 2026-08-17: NO gen-3 tail rerun — jump
  straight to gen-4; identity corpus v2 (Melise in pool +
  preservation training + held-out name eval) is the fix of record.
  Whether meliseai.com swaps to gen-3 as-is (she self-names from the
  pool) or stays on gen-2 until gen-4 is an open user decision.

### 6. Infra / cost

- **Provisioning**: given measured preemption pain vs a possibly-thin
  Spot discount, consider launching gen-4 on-demand from the start
  (~$0.85/hr; gen-3's remaining-run premium was ~$10–60 for ~2 days
  saved). Confirm actual Spot rate from the invoice first.
- Quota: GPUS_ALL_REGIONS=3 + regional L4 requests were resubmitted
  2026-08-09 (reconciling) — recheck before gen-4; if granted,
  parallel runs (e.g. the matched-param A/B above) become possible.
- Keep: restarter + boot-resume machinery (works on-demand too, as
  the stockout recovery showed), bucket-backed TB, BucketSync.

## Gen-4 proposal v1 (2026-08-14)

**Width AND experts AND data.** Model: d=512, 3 blocks / 13 attn
layers (deliberately NOT the large preset's 4 — depth held to cap
FLOPs), n_heads=16 (head_dim 32), **40 routed experts** top-4, K3
shape rules for the rest (kv_lora 128, kda_decay_rank 32, latent 256,
expert_hidden 256, shared 1024, dense 2048), bpe8k, ctx 2048.

Parameter budget (counted, not estimated — CPU-instantiated):

| | gen-3 | gen-4 v1 | × |
|---|---|---|---|
| total | 72.1M | **163.2M** | 2.26 |
| active/token | 45.6M | **78.3M** | 1.72 |
| routed pool | 31.9M | 94.4M | 2.96 |
| active/total | 63% | 48% | sparser |

Width fattens each expert (1.33M→2.36M) while count grows 24→40: the
storage pool — where the eviction result lives — triples, at only
1.72× compute.

**Data**: expand fineweb-edu 400MB → ~2GB (→ corpus ~1.05B unique
tokens, ~2 effective epochs at the 2.2B budget); grow dialogue too if
a source exists (thesis domain, only 24MB). Mix weights recomputed at
freeze. Per-expert diet stays ≈ gen-3 (220M vs 247M tok/expert), so
40-way routing doesn't thin the slices.

**Token budget** (freeze AFTER the throughput probe, see below).
Chinchilla is ambiguous for MoE — 3.3B by total params, 1.6B by
active; recommendation 2.2B:

| tokens | tok/total | tok/active | est. wall-clock (end-to-end) | ~cost |
|---|---|---|---|---|
| 1.5B floor | 9.2 | 19.2 | ~9.5d | $195 |
| **2.2B rec.** | 13.5 | 28.1 | **~13d** | $270 |
| 3.3B max | 20.2 | 42.1 | ~19d | $390 |

Assumes ~2.2k tok/s (FLOPs-scaled from gen-3's measured 4.1k; method
retro-predicts the large preset's measured 1.8k within ~5%), batch
b5→b4, on-demand from launch (zero preemptions; Spot discount thin —
verify invoice). SFT grows to ~21h, GRPO ~10h. GRPO stays the memory
high-water mark; toy-pipeline validation mandatory as always.

**Kept, deliberately (reviewed 2026-08-14):**
- **KDA/MLA 3:1 hybrid + NoPE** — trains stably, and it is the
  *faster-training* option: vanilla full attention at 2048 ctx would
  cost +10–20% step time (+1–1.5d) for likely-neutral quality, and
  its costs grow with any future ctx increase. Decision rule: if
  gen-3's recall evals / chat battery come back weak, revisit — KDA's
  fixed 32×32/head state and 4-of-13 exact-attention layers are the
  suspect, and a vanilla or 2:1 variant becomes a capability trade,
  not a taste trade. (Vanilla footnote: it would actually *save*
  ~2.6M attn params; serving cache 4MB → ~110MB/session — fine at
  concurrency 1.)
- **AttnRes (full form)** — ~free (0.01M params), two generations
  stable, and mechanistically points the right way for our diagnosed
  interference problem (stages choose which prior outputs to read
  instead of sharing one overwrite-prone stream). Least-studied
  component — first suspect if a run behaves unexplainably. Ablation
  vs plain residuals possible via deepseek.py but loses the GPU-day
  priority fight.
- Depth, bpe8k, pipeline: unchanged. bpe16k now architecturally
  viable at d=512 (d/V 6.3→3.1%… still fine; embed+head ~10%) but
  stays deferred — gen-5 with the width already banked.

**Open at freeze** (BUILD DONE 2026-08-17 — recipe frozen-in-draft in
`gen4-medium-wide.md`; only the VM hardware probe remains):
- [x] **LR schedule: WSD** (user call 2026-08-17). Implemented in
      run_utils.lr_at (`--lr-schedule wsd --decay-frac 0.15`, tested):
      extend-mid-hold needs no surgery; the decay's eviction cost is
      paid at a chosen time, not as a side effect.
- [ ] **Throughput/memory probe** of the exact config on the VM
      (~1h, between gens): real tok/s + peak GiB + b4-vs-b5 → then
      freeze PT_STEPS (2.2B budget decided; corpus tensors now int16,
      ~2 GB freed on-device). Also: time one CUDA probe round (≤3%
      overhead budget) and a toy GRPO step at the new size.
- [x] **Probe suite** — COMPLETE and wired into all three loops
      (`--probe-every`; TB `probe/*` + metrics probe/probe_dump
      events; rotating dumps, greedy+t0.8). facts.json 58→298 with
      reject-lists (forced-choice echo slack closed); gen_fact_sft.py
      (train split, self-checked); identity corpus v2
      (transformer/identity.py: ~370-name pool with import-time
      asserts keeping heldout/novel strata disjoint; dated preambles;
      ask-twice); `context_recall` + `facts` RLVR families with
      per-task preamble plumbing through rollouts and eval;
      serve.py `{date}` preamble rendering; task_formats + chat
      verbatim probes.
- [x] Eval fixes landed: best.pt by trailing-3 windowed val_bpb
      (resume-safe; recover_best_from_metrics matches); deterministic
      fixed-window per-domain evals (pin_val_windows /
      fixed_window_eval); val_domain_acc/top5/ent + byte-weighted
      val_group/*; keeper checkpoints (--keep-every 25000);
      rebuild_tb.py + recover_metrics.py replay every new event type
      (and two PRE-EXISTING parser gaps got fixed along the way:
      `ent=` in pretrain step lines and `dead=/ent=` in grpo lines
      had silently broken log recovery since gen-3).
- [x] Mix updated: dialogue → 7% with PersonaChat + BlendedSkillTalk
      (+15 MB, `dialogue_persona` val canary added); fineweb 44%.
      Byte counts re-freeze when the 2GB fineweb lands. Small-domain
      policy: no floors — eviction accepted but now visible live
      (groups + acc/ent decomposition + keepers for post-hoc).
- [x] Identity: Melise IN the pool, NOT dominant (~8% of identity
      preambles) — preservation over enforcement, per the probe
      finding. chat.DEFAULT_PREAMBLE renamed Lily → Melise.
- [x] SFT/GRPO budgets (user call): SFT 20k × b4 with 3% pretrain
      replay (`--replay-frac`), tail 1.5k with chat_facts added,
      GRPO 600 with the two new families in the weighted rotation.
- [x] Serving latency, first data point: gen-3 (45.6M active) on
      Cloud Run 2 vCPU ≈ 4.5 tok/s. Gen-4 (78.3M) projects ~2.5–3 —
      decide --cpu 4 / bf16-at-load / accept at swap time.

**Inter-generation work queue** (keeps GPU idle time ~zero): gen-3
post-run ritual (results/NOTEBOOK/memories/chat battery/model swap +
expert-routing map — the routing map doubles as 40-expert due
diligence) → fineweb expansion via add_dataset.py → eval/sampling
diffs → new preset config → probe → toy pipeline → freeze recipe as
gen4-medium-wide.md → launch.

Gen-5 pile (not gen-4): bpe16k, MTP (data-efficiency lever — relevant
now that unique tokens are the binding constraint), attention-ratio
revisit per recall results, AttnRes ablation.

## Deferred by explicit user decision (need sign-off to start)

d=512 · bpe16k · DPO (standing since 2026-08-11 handoff; d=512 is
the presumptive gen-4 headline but stays here until called).

## Incident 2026-08-16: post-training skipped, shutdown loop

Pretrain finished cleanly (end @145k, 18:55 UTC Aug 15) but SFT never
started. Working theory (confirm via ~/pipeline.log): the Aug-10
pre-launch TOY validation run dirs were still the newest sft/rlvr
dirs on the VM disk, so --resume's stage_done() skipped all
post-training, ran offline evals, hit PIPELINE_DONE → DONE_CMD. Its
scheduler-pause half FAILS SILENTLY (VM scopes lack cloudscheduler)
but the shutdown half works → boot→evals→self-shutdown→restarter
loop, ~15h (~$9). Recovery: pause restarter (done, from laptop),
delete toy dirs local+bucket, relaunch `--post-only` with full env.
Gen-4 fixes (ALL LANDED 2026-08-17 except the manual checklist item):
- [ ] Pre-launch checklist: DELETE toy/validation run dirs for all
      stages before the real launch (manual — chain_ok would refuse
      them anyway, but belt AND suspenders; in gen4-medium-wide.md's
      gates).
- [x] pipeline.sh guard — implemented STRONGER than the staleness
      idea: `chain_ok` requires a downstream dir's run.json lineage
      to name the current upstream run. Kills both incident classes
      (other-generation dirs AND toy dirs) with no timestamp
      fragility. Toy validation must plant a stale dir and watch it
      refused.
- [x] DONE_CMD: scheduler-pause half dropped — shutdown-only
      (`DONE_CMD="sudo shutdown -h +2"`); restarter stays paused
      laptop-side on-demand. OPERATIONS.md updated.
- [x] evals.jsonl self-push: eval_checkpoint.py uploads to the run's
      bucket mirror after appending (run_utils.push_run_file);
      probe_checkpoint.py pushes probes.jsonl the same way.
- [x] SFT replay: sft.py `--replay-frac 0.03 --replay-mix <mix>` —
      raw pretrain-mix LM batches interleaved every ~33rd step,
      corpora resident on CPU (zero GPU cost), train/replay_bpb
      logged. Measurable via the same forgetting eval + the raw-form
      verbatim probes now running during SFT.

## Blocked on gen-3 completion — revisit each after

- [ ] Final war_and_peace number + decay-tail recovery verdict
- [ ] Offline forgetting evals (the eviction end-state, measured)
- [ ] Test-slice bpb vs gen-2's 1.247 (`eval_checkpoint.py`, the only
      cross-gen comparable)
- [ ] SFT `val_source/*` identity curves; GRPO per-task incl. recall
- [ ] Chat battery vs gen-2 (the product-level verdict)
- [ ] Expert×domain routing map (VM, post-pipeline)
- [ ] August invoice: true Spot vs on-demand L4 rates
