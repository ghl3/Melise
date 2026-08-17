# Gen-4 — medium-wide (IN PREP — freeze pending hardware probe)

The capacity-and-content generation: 2.26× the params (3× the routed
storage) on ~1.5× the tokens, aimed squarely at gen-3's three measured
gaps — content (facts ~0), capacity (small-domain eviction), and her
own name ("You are Melise" → "Leo"). Instruments-first: every failure
mode gen-3 exposed now has a live metric before the run starts.
Decisions and rationale: `gen4-ideas.md` (proposal v1 + evidence),
`gen4-eval-suite.md` (probe suite), NOTEBOOK.md 2026-08-17.

User calls locked 2026-08-17: WSD schedule; 2.2B token budget
(recommendation accepted); gen-3 serves meliseai.com meanwhile (done —
`melise-worker` rev 00002); SFT stays 20k steps at the new scale;
PersonaChat + BlendedSkillTalk added to dialogue; chat-facts into both
SFT and GRPO.

## Recipe (draft — freeze after the VM throughput/memory probe)

| knob | value | why |
|---|---|---|
| model | **kimi3-medium-wide** — d=512, 3 blocks (13 attn layers), 16 heads (head_dim 32), **40 experts** top-4, K3 shape rules | **163.2M total / 78.3M active** (counted; test-pinned). Routed pool 94.4M = 3× gen-3's — targets FFN storage, where eviction and facts live, at only 1.72× active compute |
| tokenizer | bpe8k (unchanged) | bpe16k stays deferred (gen-5, width now banked) |
| context | 2048 | unchanged |
| pretrain | **~215,000 steps × b5 × 2048 ≈ 2.2B tokens** (13.5 tok/total-param, 28.1 tok/active) | Chinchilla band for MoE is 1.6–3.3B. **b5 is the plan** (decided 2026-08-18): int16 corpora freed ~2 GB and b5 buys ~2 days of wall-clock — but the probe must CONFIRM it (peak ≤ ~21.5 GiB, the margin gen-3 ran at). Fallback: b4 + 268,500 steps, same 2.2B. Batch and steps freeze TOGETHER — b5 with the b4 step count would silently train 2.75B |
| LR | **2.5e-4 peak, WSD** (warmup 1%, hold, linear decay over final 15%) | gen-3 measured decay AMPLIFYING eviction — when to pay that cost is now an explicit choice; extending mid-hold needs no schedule surgery. Peak scaled mildly from 3e-4 at d=384 |
| data mix | `configs/mix-gen4-chat.json` — grouped: fineweb 44% (2GB, all-dump shuffle), wikitext 16%, books 15% (45 works), code 8%, **dialogue 7%** (4 corpora), enwik8 5%, math 4%, reference 1% | shares are the knob, not file sizes; books drop ~28 → ~20 effective epochs; dialogue = product register, grew via PersonaChat+BST (+15MB) |
| SFT | 20,000 × b4, **+3% pretrain replay** (`--replay-frac 0.03 --replay-mix <DATA_MIX>`) | replay anchors the base LM (gen-3: +0.74 bpb forgetting, novel-name induction destroyed). chat_facts.txt + identity v2 corpora join via the chat_* glob |
| SFT tail | 1,500 @ lr 3e-5 on tasks+identity+**facts** | generative access to the formats (gen-2 lesson), now incl. facts |
| GRPO | 600 steps, lr 1e-5, headroom-weighted; **+context_recall, +facts** families | context_recall carries per-task randomized preambles (name/date/refusal); facts scores the train split of the 298-entry table |
| eval | deterministic **fixed-window** per-domain eval (16 pinned windows/domain), acc/top-5/entropy triples, byte-weighted `val_group/*`, **best.pt = min trailing-3 val_bpb** | kills the ±0.04 draw noise that held best.pt stale 53k steps; decomposes bpb moves live |
| probes | `probe/*` every 2,000 pretrain steps (raw forms) / 800 SFT / 100 GRPO steps (chat forms) + rotating dump battery, greedy+t0.8 | gen-3's blind spots, instrumented; full battery offline on keepers |
| checkpoints | save 1,000, keep-last 5, **keeper every 25,000** (pruning-exempt) | ~2 GB each at 163M; keepers buy post-hoc science (gen-3's floor-era ckpt was pruned) |
| cadence | PT_EVAL_EVERY=500 (~15 min at 2.2k tok/s) | 250 would double evals on a 270k-step run for nothing |

Est. wall-clock at b5: pretrain ~9.5–10d + SFT ~21h + tail ~1.5h +
GRPO ~10h ≈ **~11 days on-demand** (~$225); the probe's measured tok/s
firms this. Fallback b4 ≈ ~13d (~$270). Zero preemptions expected;
restarter stays PAUSED (boot-resume crontab covers host errors via
automaticRestart).

## Pre-launch gates (ordered)

1. **VM hardware probe (~2h, decides the freeze):**
   - `df -h` FIRST; archive/delete gen-3's local run dirs (bucket has
     them) — the run needs ~35 GB+ (2 GB fineweb + ~2 GB token cache
     + keep-last 5 × 2 GB + ~9 keepers).
   - **Pretrain at b5** (the plan): 30 steps for peak GiB + tok/s,
     then a **500–1,000-step LR-stability segment** at 2.5e-4 —
     grad_norm/loss-spike check, first look at moe/* balance at
     top-4-of-40, and an early loss curve vs gen-3's at the same
     token count. **Adopt b5 + PT_STEPS=215000 only if peak ≤
     ~21.5 GiB and the segment is clean**; else b4 + 268500. Also
     measure b4 tok/s for the record.
   - Toy GRPO step at the new size (GRPO is the memory high-water
     mark — policy + frozen reference + rollout caches;
     `--update-microbatch` may need lowering).
   - **One probe round timed on CUDA** (budget ≤3% of wall-clock).
     If over: raise SFT_PROBE_EVERY first, trim verbatim_per_file
     second (battery sizes are hardcoded in the three loops — edit
     BEFORE launch, nothing changes mid-run).
   - Refresh the repo git bundle in the bucket while the VM is up.
2. **fineweb 2GB fetch** (`prep_fineweb.py`, laptop, hours) →
   `add_dataset.py` upload → VM `download_data.py` → pre-warm the bpe8k
   token cache (first mix load tokenizes ~2.9GB — let it run before
   launch day).
3. **Regenerate + upload chat corpora**: gen_identity_sft.py (v2),
   gen_task_sft.py (new families ride along), gen_fact_sft.py (new) —
   then `sample_data.py` audit (500-conversation rule).
4. **Cross-gen probe baselines**: `probe_checkpoint.py --out` on gen-2
   and gen-3 best.pts (all stages, full table) so gen-4's TB curves
   have reference lines. VM or laptop-light.
5. **Toy full-pipeline validation** on the VM (30/20/10/3 steps, real
   preset/tokenizer) — must exercise: WSD resume mid-hold, replay
   batches, probes in all three loops, context_recall rollouts (per-
   task preambles in logs), keeper exemption, `chain_ok` refusing a
   deliberately planted stale dir, evals.jsonl/probes.jsonl self-push.
   **Then DELETE the toy run dirs, local + bucket** (they are the
   gen-3 incident, twice).
6. Recheck GPU quota (resubmitted 2026-08-09) and the August invoice's
   real Spot rate — informational; launch is on-demand regardless.

## Launch sequence

Per OPERATIONS.md, with the gen-4 env (pipeline.sh defaults ARE this
recipe now, but pass everything explicitly anyway):

    DONE_CMD="sudo shutdown -h +2" \
    PRESET=kimi3-medium-wide TOKENIZER=bpe8k \
    DATA_MIX=configs/mix-gen4-chat.json \
    PT_STEPS=215000 PT_BATCH=5 PT_LR=2.5e-4 PT_LR_SCHEDULE=wsd \
    SFT_BATCH=5 SFT_STEPS=20000 \
    nohup bash scripts/pipeline.sh > ~/pipeline_nohup.log 2>&1 &

(batch/steps pair per the probe verdict: b5/215000 or b4/268500 —
never mix them.) WSD note: decay starts at 85% of PT_STEPS; review
the curves at ~80% — decaying early (relaunch-with-lower---steps +
--resume) is a legitimate ship-sooner choice, not an accident.

(NOTE: DONE_CMD is shutdown-only now — the VM-side scheduler pause
never worked, scopes; keep the restarter paused from the laptop.)
Verify stepping → `--install-boot-resume` with the identical env →
restarter stays paused (on-demand; resume it only during a stockout
recovery window).

## Success metrics (decided before the run, so nobody argues after)

- **Primary capacity bet**: `probe/facts/*/heldout` off the floor
  (gen-3: ~0 everywhere in chat form; base cloze 0.67 on instances
  only). This is what 3× routed storage is FOR.
- **Identity**: `probe/identity/novel` ≥ its pretrain value after SFT
  (gen-3: 0.67 → 0.00 — the corpus-v2/preservation test), and
  serve-time "You are Melise" → "Melise".
- **Forgetting**: enwik8 test bpb (pretrain → rlvr) gap well under
  gen-3's +0.74 (replay + probes watching it live);
  `probe/date/honesty` stays ~1.0 (don't regress the refusal).
- **Eviction**: war_and_peace val curve vs gen-3's +0.34 climb;
  val_domain_acc/ent split telling rank-loss from confidence-
  misallocation in realtime; `val_group/books` the honest aggregate.
- **Cross-gen bar**: enwik8 test bpb < 1.096 (gen-3's best). Not the
  headline — the mix shifted toward chat/facts — but it must not
  regress much while the above move.
- GRPO: arith > 0.25, everything at 1.00 holds, context_recall/facts
  learn (dead_frac falling on the new families).

## Serving note

Gen-3 on Cloud Run CPU (2 vCPU) decodes ~4.5 tok/s (45.6M active).
Gen-4 at 78.3M active projects ~2.5–3 tok/s — likely too slow.
Options at swap time: --cpu 4, bf16 cast at load, or accept. Decide
with a measured number, not now.

## Results (fill during/after)

- [ ] hardware probe numbers (tok/s, peak GiB, b4 vs b5, probe-round s)
- [ ] frozen PT_STEPS + final mix byte counts
- [ ] wall-clock per stage; incidents
- [ ] pretrain test bpb + per-domain/group curves; holdout books
- [ ] probe arcs (facts heldout, identity strata, verbatim gap) across
      pretrain → SFT → tail → GRPO
- [ ] SFT forgetting with replay (vs +0.74) ; replay_bpb curve
- [ ] GRPO per-task incl. context_recall + facts
- [ ] chat battery vs gen-3; serve-time identity/date checks
- [ ] meliseai.com swap decision + latency measurement
