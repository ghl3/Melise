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

## Working draft — decided direction (user call, 2026-08-14)

**Do both: width AND experts, plus data.** d=512 (3 blocks / 13 attn
layers — NOT the large preset's 4), n_heads=16, 40 routed experts
top-4, K3 shape rules for the rest (latent 256, expert_hidden 256,
shared 1024, dense 2048), bpe8k unchanged.

- **163.2M total / 78.3M active** (×2.26 / ×1.72 over gen-3); routed
  pool triples to 94.4M — the eviction result's budget line.
- **Token budget ~2.2B** (13.5 tok/total-param, 28 tok/active —
  between the two Chinchilla readings for MoE). Per-expert diet stays
  ≈ gen-3 (220M vs 247M tok/expert).
- **Corpus: expand fineweb-edu 400MB → ~2GB** first (→ ~1.05B unique
  tokens, ~2 effective epochs); grow dialogue too if a source exists.
  Mix weights recomputed at recipe freeze.
- **Est. ~2.2k tok/s** (FLOPs-scaled; method retro-predicts large
  preset's measured 1.8k). Batch likely 4. **~11.6d pretrain + ~1.5d
  post ≈ 13 days, ~$270 on-demand.** Floor option: 1.5B tokens ≈
  9.5d. Probe memory before freeze; GRPO is the high-water mark.
- Serving: active ×1.72 → Cloud Run CPU latency check at swap time.

## Deferred by explicit user decision (need sign-off to start)

d=512 · bpe16k · DPO (standing since 2026-08-11 handoff; d=512 is
the presumptive gen-4 headline but stays here until called).

## Blocked on gen-3 completion — revisit each after

- [ ] Final war_and_peace number + decay-tail recovery verdict
- [ ] Offline forgetting evals (the eviction end-state, measured)
- [ ] Test-slice bpb vs gen-2's 1.247 (`eval_checkpoint.py`, the only
      cross-gen comparable)
- [ ] SFT `val_source/*` identity curves; GRPO per-task incl. recall
- [ ] Chat battery vs gen-2 (the product-level verdict)
- [ ] Expert×domain routing map (VM, post-pipeline)
- [ ] August invoice: true Spot vs on-demand L4 rates
