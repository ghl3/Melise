# Datasets

Every corpus used to train the models in this repo, by pipeline stage.
Canonical storage is the GCS bucket — `gs://crow-391712-transformer-data/data/`
with sizes and sha256s in `manifest.json` alongside. The laptop and VM
are both just caches: `scripts/download_data.py` pulls, and
`scripts/add_dataset.py` uploads + registers (laptop only). Nothing in
`data/` is committed to git.

Sizes below are raw UTF-8 bytes (from the manifest, 2026-08-10).

## Pretrain

36 files, ~915 MB raw. The mix is `configs/mix-downweight-wiki.json`:
every `data/*.txt` except `chat_*`, sampled proportionally to
bytes × multiplier — the multipliers pull Wikipedia, the dictionary, and
enwik8 down and let the book collection breathe. **Share** below is the
resulting sampling probability (~142 MB effective per epoch-equivalent).
Generation-2 consumed ~819M tokens ≈ 2.6 epochs of the mix.

Corpus loading is byte-sliced first, then tokenized per the model
config's `tokenizer` field, with token caches in `data/.tokcache` —
byte-first slicing keeps split boundaries identical across tokenizers.

| Dataset | Size | Source | Weight | Share | Notes |
|---|---|---|---|---|---|
| wikitext-103 | 540.6 MB | smerity.com (Wikipedia) | ×0.1 | 38.2% | Downweighted, still the largest slice |
| math-openweb | 101.0 MB | HF OpenWebMath subset | ×0.15 | 10.7% | Added for gen-2 |
| code-python | 100.1 MB | HF codeparrot-clean subset | ×0.15 | 10.6% | Added for gen-2 |
| enwik8 | 100.0 MB | mattmahoney.net | ×0.1 | 6.4% | 90/5/5 train/val/test — see below |
| webster | 28.0 MB | Gutenberg #29765 | ×0.1 | 2.0% | Dictionary; downweighted |
| shakespeare-all | 5.4 MB | Gutenberg #100 | ×1 | 3.8% | |
| bible-kjv | 4.3 MB | Gutenberg #10 | ×1 | 3.1% | |
| les-miserables | 3.3 MB | Gutenberg #135 | ×1 | 2.3% | |
| war-and-peace | 3.3 MB | Gutenberg #2600 | ×1 | 2.3% | |
| monte-cristo | 2.7 MB | Gutenberg #1184 | ×1 | 1.9% | |
| wealth-of-nations | 2.4 MB | Gutenberg #3300 | ×1 | 1.7% | |
| don-quixote | 2.3 MB | Gutenberg #996 | ×1 | 1.6% | |
| anna-karenina | 2.0 MB | Gutenberg #1399 | ×1 | 1.4% | |
| brothers-karamazov | 2.0 MB | Gutenberg #28054 | ×1 | 1.4% | |
| david-copperfield | 2.0 MB | Gutenberg #766 | ×1 | 1.4% | |
| descent-of-man | 1.9 MB | Gutenberg #2300 | ×1 | 1.3% | |
| middlemarch | 1.8 MB | Gutenberg #145 | ×1 | 1.3% | |
| decline-fall-1 | 1.8 MB | Gutenberg #731 | ×1 | 1.3% | |
| moby-dick | 1.2 MB | Gutenberg #2701 | ×1 | 0.9% | |
| voyage-beagle | 1.2 MB | Gutenberg #944 | ×1 | 0.8% | |
| shakespeare | 1.1 MB | karpathy/char-rnn | ×1 | 0.8% | tinyshakespeare |
| origin-species | 0.9 MB | Gutenberg #1228 | ×1 | 0.7% | |
| tale-two-cities | 0.8 MB | Gutenberg #98 | ×1 | 0.5% | |
| pride-prejudice | 0.7 MB | Gutenberg #1342 | ×1 | 0.5% | |
| walden | 0.6 MB | Gutenberg #205 | ×1 | 0.5% | |
| huckleberry-finn | 0.6 MB | Gutenberg #76 | ×1 | 0.4% | |
| sherlock | 0.6 MB | Gutenberg #1661 | ×1 | 0.4% | |
| grimms | 0.5 MB | Gutenberg #2591 | ×1 | 0.4% | |
| frankenstein | 0.4 MB | Gutenberg #84 | ×1 | 0.3% | |
| meditations | 0.4 MB | Gutenberg #2680 | ×1 | 0.3% | |
| treasure-island | 0.4 MB | Gutenberg #120 | ×1 | 0.3% | |
| wizard-of-oz | 0.2 MB | Gutenberg #55 | ×1 | 0.2% | |
| treatise-light | 0.2 MB | Gutenberg #14725 | ×1 | 0.2% | |
| relativity | 0.2 MB | Gutenberg #30155 | ×1 | 0.1% | |
| alice | 0.2 MB | Gutenberg #11 | ×1 | 0.1% | |
| discourse-method | 0.1 MB | Gutenberg #59 | ×1 | 0.1% | |

### enwik8 is load-bearing

`configs/mix-downweight-wiki.json` splits enwik8 90/5/5: the 5% val
slice drives in-run eval, and the final 5 MB is the **virgin test
slice** used by `scripts/eval_checkpoint.py`. Test bpb there is
normalized by slice *bytes* (token NLL ÷ bytes), which makes it the one
number comparable across tokenizers, model sizes, and generations —
every headline result in `docs/NOTEBOOK.md` (scarlet-harbor 1.560,
golden-dell 1.247) is this metric. Never train on the test slice;
byte-first slicing guarantees its boundaries never move.

## SFT

5 files, ~897 MB, 283,659 conversations. All stored in the byte chat
template — `0x01` user / `0x02` assistant / `0x03` end-turn / `0x04`
end-conversation, control bytes verified absent from every pretrain
corpus, and `chat_*` files excluded from the pretrain mix so chat can
never leak into pretraining. Because the BPE tokenizer pins the same
control characters to ids 0–4, these files encode correctly under both
byte and BPE models with no re-parsing.

`scripts/sft.py` picks up every `data/chat_*.txt` and samples batches
**uniformly per conversation**, so a corpus's share of training is its
conversation *count*, not its bytes. Conversations over the token
budget are split at turn boundaries.

| Dataset | Size | Convs | Share | Source | Notes |
|---|---|---|---|---|---|
| chat-smoltalk | 876.9 MB | 231,978 | 81.8% | HF SmolTalk (train shards) | Multi-turn general chat |
| chat-dolly | 11.9 MB | 15,011 | 5.3% | HF databricks-dolly-15k | Instruction-following |
| chat-oasst1 | 6.3 MB | 3,670 | 1.3% | HF OASST1 | English, best-ranked conversation paths |
| chat-tasks | 1.7 MB | 30,000 | 10.6% | generated: `scripts/gen_task_sft.py` | RL cold start — see below |
| chat-identity | 0.3 MB | 3,000 | 1.1% | generated: `scripts/gen_identity_sft.py` | Persona — see below |

The HF corpora were fetched with `scripts/prep_chat_data.py` (plain
HTTPS, no `datasets` dependency) and are static. The two *generated*
corpora are code-derived and must be **regenerated and re-uploaded
whenever their generators or the task definitions change**:

- **chat-tasks** — single-turn worked examples from the RL task
  families (`transformer/rl/tasks.py`); each response is the task's
  canonical full-credit answer, self-checked against that task's own
  scorer at generation time, so the SFT target and the RL reward can
  never drift apart. Regenerated 2026-08-10 (30k examples) to add the
  `recall` family and prompt-template variety.
- **chat-identity** — persona exchanges (assistant name, what-am-I,
  honest "I have no clock/calendar" answers, intro→name-recall
  multi-turns). The assistant name is a CLI flag
  (`--name Lily`) — regenerate on the next rename.

The gen-2.5 **task tail** (a short low-LR SFT pass, `SFT_TAIL_*` in
`scripts/pipeline.sh`) trains on chat-tasks + chat-identity only, to
convert formats the main SFT pass merely models into what the model
actually generates.

## RL (GRPO)

No stored dataset. Prompts are generated at runtime by the six task
families in `transformer/rl/tasks.py` — copy, arith, parity, count,
words, recall — each drawing from 3–4 paraphrase templates and scored
by a deterministic function in [0, 1] (no reward model). Generators use
seeded RNGs, so rollout prompts are reproducible across resumes and
each run's eval set is fixed. A −0.2 penalty applies to completions
that never emit the end-turn id.

Scoring quirks that exist on purpose:

- **arith** reads the *last* integer (showing work is rewarded, not
  punished) and gives 0.25 for worked-steps evidence (`=`) on a wrong
  answer — without it, all-wrong rollout groups have zero advantage
  variance and RL can never bootstrap the format (the gen-2 failure).
- **recall** gives half credit for the first-person echo ("My name is
  Frank."), full credit only in second person, and zero to replies
  over 10 words (blocks the list-every-name hack).
- Template variety (added 2026-08-10) means seeded eval rewards are
  **not comparable across that change** — gen-2's 0.700 and gen-2.5's
  numbers are different eval sets.

## Housekeeping

- Eyeball any dataset: `scripts/sample_data.py {pretrain|sft|rl}
  [--name <canonical-name>] [-n N]` prints metadata + sampled examples;
  with no `--name` it picks a dataset with the stage's real sampling
  weights (mix share / conversation count / uniform families).
- Add a new dataset: `scripts/add_dataset.py --name <n> --url <u>`
  (or `--seed-from-local` to sync everything in `data/`); it uploads,
  hashes, and registers in the manifest. Pull with
  `scripts/download_data.py` (works on VMs via the default service
  account).
- Licensing: Gutenberg texts are public domain; wikitext/enwik8 derive
  from Wikipedia (CC BY-SA); SmolTalk, Dolly, OASST1, OpenWebMath, and
  codeparrot-clean are permissively licensed HF datasets. Fine for
  this project's research use.
- Token caches (`data/.tokcache`) are derived state — safe to delete,
  rebuilt on demand.
