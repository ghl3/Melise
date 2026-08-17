# Gen-4 probe & eval suite (design — 2026-08-17)

Motivated by gen-3's blind spots: the ROMEO keyhole (one prompt, one
register), the identity failure found only at serve time ("You are
Melise" → "Leo"; name drifts per reply), verbatim-vs-register storage
discovered only by hand-probing after the run, and eviction diagnosed
days late for want of accuracy/entropy alongside bpb.

Principles:

1. **Same capabilities probed at every stage.** One shared module
   (`transformer/probes.py`) defines prompts, keys, and scorers; the
   pretrain loop wraps probes as raw continuations, the post-train
   loops wrap the same probes in the chat template. TB tags align
   (`probe/*`) so a capability's arc is one dashboard across
   pretrain → SFT → tail → GRPO.
2. **Two output classes.** *Scored scalars* (deterministic scoring →
   TB + metrics.jsonl, trend-watchable) and *generation dumps* (fixed
   prompts, greedy + t=0.8, into metrics.jsonl sample events, for
   offline reading — the ROMEO pattern, pluralized).
3. **Cheap enough to run in-loop.** Scalar probes every 4th eval
   (~1 min of GPU per round, budgeted below); dump probes rotate so
   each tick emits 2–3, full battery every ~1k steps.
4. Loss-side metrics (fixed-window per-domain bpb, accuracy/top-5/
   entropy triples, best-by-windowed-val_bpb) are specified in
   gen4-ideas.md; this doc is the behavior side. Offline per-checkpoint
   runs (`scripts/probe_checkpoint.py`) reuse the same module against
   keeper checkpoints.

## A. Generation dump battery (the ROMEO expansion)

Fixed prompts, one per register/capability. Pretrain: raw
continuation. Post-train: chat-wrapped equivalent (right column).

| id | pretrain prompt | post-train wrapping |
|---|---|---|
| play | `ROMEO:\n` (kept for gen-3 continuity) | "Continue this play script: ROMEO:" |
| novel | `“I am glad to see you,” said the count,` | "Continue this story: …" |
| wiki | `France is a country in` | "Tell me about France." |
| expository | `Photosynthesis is the process by which` | "Explain photosynthesis." |
| dialogue | movie-dialogue snippet (2 turns) | "Let's chat!" free turn |
| code (if kept in mix) | `def fibonacci(n):` | "Write a fibonacci function." |
| sonnet-key | 2 exact lines of Sonnet 18 | "Finish this line: Shall I compare thee…" |
| instance-list | `Common pets include cats, dogs,` | "Name three animals." |

Rationale: register competition (play vs novel vs expository) becomes
directly visible instead of inferred from one prompt's argmax flips;
sonnet-key and instance-list make the form-vs-content tradeoff
readable at a glance at any checkpoint.

## B. Scored scalar probes (TB `probe/*`)

### B1. Verbatim recall / memorization (pretrain + post-train)

- Fixed set: ~24 passages, stratified across groups (fineweb,
  wikitext, enwik8, books, dialogue, code) **plus the fully-held-out
  books** (emma, great_expectations) as the no-leakage control.
- Score: give a 64-byte key, greedy-generate 128 bytes, compute
  normalized Levenshtein similarity to the true continuation.
- TB: `probe/verbatim/<group>` (mean similarity), plus
  `probe/verbatim/heldout` (should stay near chance; the
  trained-vs-heldout gap is the LIVE memorization index — the offline
  memorization-gap eval, streamed).
- Post-train: same passages, both raw-continuation form (forgetting
  of the base skill) and chat form ("Continue exactly: …",
  instruction-following version). Gen-3's +0.74 bpb forgetting would
  have been visible per-group, per-eval.

### B2. Name induction / identity (the headline addition)

Capability = copy an arbitrary name from context. Testable at BOTH
stages:

- **Pretrain form** (`probe/induction_name`): synthetic contexts
  `"You are {name}, a tiny language model. […] My name is"` →
  score exact continuation of {name} (match rate + mean logprob of
  the name's tokens). Names drawn fresh each eval from three strata:
  *pool* (will appear in SFT data), *held-out real* (never in any
  training data), *novel strings* (generated, e.g. "Vexima") — the
  last is the pure copying test. Baseline expectation: induction
  capability rises during pretrain; if `novel` stays at 0 into late
  pretrain, no amount of SFT will fix serve-time renames.
- **Post-train form** (`probe/identity/{pool,heldout,novel}`): R=24
  one-turn conversations per stratum: preamble "You are {name}, a
  tiny language model." + "What is your name?" (3 phrasings), greedy
  reply, score contains-{name}. Gen-3's serve-time bug = heldout/novel
  scoring ~0 while pool scores ~1; would have been on TB at SFT step
  800.
- **Consistency** (`probe/identity_consistency`): ask twice in one
  conversation → same-name rate (the Leo→Ivan drift, quantified).

### B3. Instance retrieval (content vs form)

- Pretrain form: list-continuation ("Animals: cat, dog, horse,") —
  count valid instances in the next ~20 tokens against a wordlist →
  `probe/instances`.
- Post-train form: "Name an animal / a color / a city." → valid-and-
  non-echo rate (gen-3's "An animal" echo scores 0). Doubles as the
  benchmark for whether gen-4's expert-pool capacity buys content.

### B4. Context-field retrieval: date/time (same family as B2)

The serve-time preamble gains dynamic date/time ("You are …. Today is
Sunday, August 17, 2026."; serve.py renders it per-request). Date is
then another retrieve-from-context field, trained and probed exactly
like the name:

- `probe/date_retrieval`: preamble with a RANDOM date → "What day is
  today?" → contains-date rate. Random dates = pure induction (no
  date is frequent enough to memorize). Pretrain form: "Today is
  {date}. […] The date today is" continuation match.
- `probe/date_honesty` (kept): NO date in context → refusal-shape
  match ("no clock/calendar/don't know") vs invented date. The pair
  enforces the right conditional: retrieve when given, refuse when
  not. Gen-3 has the refusal half already — don't regress it.
- `probe/task_formats`: 1-shot spot-checks of each RLVR task format
  during SFT (cheap preview of GRPO liveness; complements tasks_bpb).

## C. Training-side enforcement (not just measurement)

1. **Identity corpus v2** (gen_identity_sft.py): name pool expanded to
   hundreds (multi-token, rare, non-English, generated strings; no
   single dominant name — Melise included but not majority), varied
   preamble phrasings, and explicit ask-twice consistency examples.
   A held-out name list is reserved for eval and NEVER trained.
2. **New RLVR task family `context_recall`**: preamble carrying
   random fields (name, date, later maybe user location/pronouns) →
   asked to retrieve one → verifiable reward on exact retrieval;
   plus no-field variants where the honest refusal is rewarded.
   One task family reinforces the whole copy-vs-refuse mechanism;
   joins the headroom-weighted rotation. Identity corpus v2 likewise
   includes dated preambles (varied dates) with retrieval turns AND
   no-date examples keeping the refusal.
3. Optional counterweight to the copy task's echo-reflex hypothesis:
   an `instance` RLVR task (wordlist-verifiable "name an X").

## D. Cost budget (per probe round, L4, batch-1 generation)

~24 verbatim × 128B + ~72 identity × ~15 tok + instances + dumps
≈ 4–6k generated tokens ≈ 45–90 s. At every-4th-eval cadence
(1k steps ≈ 40 min training) ≈ 2–3% overhead. Dump rotation keeps the
sample cadence at gen-3's cost.

## Implementation order (inter-generation window)

1. `transformer/probes.py` + scorers (Levenshtein via difflib ratio —
   already used by rl/tasks) + tests.
2. Wire into pretrain.py (raw forms) and sft.py/grpo.py (chat forms);
   TB tags + metrics.jsonl `probe` events.
3. `scripts/probe_checkpoint.py` for keeper checkpoints & old gens
   (gen-2/gen-3 baselines for every probe, so gen-4 has comparisons).
4. Identity corpus v2 + `whoami` RLVR task.
5. Dry-run in the toy pipeline validation (which gen-3 proved earns
   its keep).
