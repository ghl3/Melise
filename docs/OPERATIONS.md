# Operations runbook — launching, monitoring, and recovering training runs

How to run the training pipeline on GCP: launch, verify, watch, survive
preemptions, restart, finish, and clean up. Written for whoever (human
or agent) is driving a run. Recipes live in `runs/<gen>.md`; data
inventory in `DATASETS.md`; narrative in `NOTEBOOK.md`. This file is
the *how*, generation-independent.

**Golden rule: verify, don't trust.** VM status, scheduler state, what's
running, and what code is on the VM all drift between sessions (Spot
preemptions happen at any time). Check each fact with the commands below
before acting on it — including facts from a handoff that was written
hours ago.

## The pieces

| thing | identity | role |
|---|---|---|
| VM | `kimi3-train`, zone `us-central1-a`, project `crow-391712` | g2-standard-8, 1× L4 (22.5 GiB usable), **Spot** (~1/3 cost). Repo at `~/transformer-learning`, venv at `.venv`. No git remote — code ships by tar-over-ssh. |
| Bucket | `gs://crow-391712-transformer-data` | Canonical store: `data/` + `manifest.json` (sha256-verified corpora), `runs/<stage>/<run>/` (live-mirrored run dirs), `repo/` (git bundles). |
| Restarter | Cloud Scheduler job `kimi3-spot-restart`, **location `us-central1`** | Every 15 min: `instances start` (harmless no-op when running). Paused when no run is active; `DONE_CMD` re-pauses it at pipeline end. |
| Pipeline | `scripts/pipeline.sh` | Resume-aware pretrain → SFT (+ task tail) → GRPO → offline evals. Every knob is an env var. |
| Runs | `checkpoints/<stage>/<run-name>/`, stage ∈ `pretrain\|sft\|rlvr` | `metrics.jsonl` is the authoritative metric record; `tb/` is a derived view; `run.json` carries lineage (`init`); `best.pt` saves at the eval that produced it. |

## Launching a run

### Pre-launch checklist (all on the VM unless noted)

1. VM `RUNNING`: `gcloud compute instances describe kimi3-train --zone=us-central1-a --format="value(status)"`.
   If `TERMINATED` and `instances start` fails with "not enough
   resources", that's a Spot capacity drought — resume the restarter
   (safe while no boot-resume crontab exists; a boot is inert) and wait.
2. Code at the intended commit. The VM has no `.git`, so verify by
   checksum, laptop → VM:
   `git ls-files -z | xargs -0 shasum -a 256 > /tmp/manifest.txt`,
   scp it over, then on the VM `sha256sum --quiet -c /tmp/manifest.txt`.
   Sync stale files with `tar czf - <files> | gcloud compute ssh kimi3-train --zone=us-central1-a --command='cd ~/transformer-learning && tar xzf -'`.
3. Corpora present (`python scripts/download_data.py` pulls anything
   missing, sha256-verified) and the tokenizer's token cache pre-warmed
   (`data/.tokcache` — populated by any prior load with the same
   tokenizer; without it the first minutes go to encoding, with it the
   run steps within ~2 min).
4. GPU idle (`nvidia-smi`), no pipeline processes
   (`pgrep -af "pipelin[e]"` — see the pkill footgun), **crontab empty**
   (`crontab -l` — a stale boot-resume entry will resume dead state on
   the next boot).
5. Scheduler state known: `gcloud scheduler jobs describe kimi3-spot-restart --location=us-central1 --format="value(state)"`.

### Launch sequence — three steps, strictly in this order

**1. Launch** with the full recipe env (see the generation's run doc for
the real values — the script's built-in defaults are a *previous*
generation's recipe, so always pass everything explicitly):

    DONE_CMD="gcloud scheduler jobs pause kimi3-spot-restart --location=us-central1; sudo shutdown -h +2" \
    PRESET=... TOKENIZER=... DATA_MIX=configs/... \
    PT_STEPS=... PT_BATCH=... SFT_BATCH=... SFT_STEPS=... \
    nohup bash scripts/pipeline.sh > ~/pipeline_nohup.log 2>&1 &

`DONE_CMD` runs only after `PIPELINE_DONE` (there is no trap — killing
the pipeline does *not* trigger it), so a finished run pauses its own
restarter and halts the VM.

**2. Verify it's stepping**: `grep "^step" ~/pretrain_run.log` shows
step lines with tok/s; the first val line (with per-domain bracket)
lands at `PT_EVAL_EVERY`. Check `nvidia-smi` memory against the recipe's
probed peak. Only proceed when stepping is confirmed.

**3. Install the boot-resume hook with the IDENTICAL env**, then make
sure the restarter is running:

    DONE_CMD=... PRESET=... [same vars] bash scripts/pipeline.sh --install-boot-resume
    gcloud scheduler jobs resume kimi3-spot-restart --location=us-central1

`--install-boot-resume` bakes the env into an `@reboot` crontab line
(`printf %q`-escaped). Two known failure modes, both hit during gen-3
prep: installing **before** launch (the hook fires on *every* boot and
resumes whatever half-state is on disk), and installing with
**different env** (a bare `--resume` falls back to the script defaults
— a smaller default `PT_STEPS` makes an interrupted long pretrain look
finished and starts SFT on an unfinished base).

Launching over `gcloud compute ssh`: the ssh session may hang after
backgrounding the pipeline (a remote child holds the channel). Harmless
— kill the local ssh, open a fresh session, and verify with
`pgrep -af "scripts/pretrai[n].py"` + the logs.

## Monitoring

### TensorBoard — bucket-backed, from the laptop (preferred)

    bash scripts/tb_bucket.sh                # everything → localhost:6006
    bash scripts/tb_bucket.sh runs/pretrain  # one stage, faster scan
    TB_PORT=6007 bash scripts/tb_bucket.sh   # alternate port

Reads `gs://…/runs` directly (needs `gcsfs` in the venv and a one-time
`gcloud auth application-default login`). Works regardless of VM state —
preemptions, IP changes, and shutdowns don't touch it. Run dirs mirror
continuously (BucketSync kicks a detached rsync at each eval/save), so
curves lag live training by minutes at most.

TensorBoard *on the VM* (`:6006 --bind_all`) also works — the firewall
rule `allow-tensorboard-tb` is tag-based and survives restarts — but the
VM's external IP changes on every stop/start, the rule is scoped to one
laptop IPv4 `/32` (`curl -4 ifconfig.me` to check yours), and the VM
halts itself when the run finishes. Use the bucket view.

### Logs (all in `~` on the VM, all survive the run dir)

| file | contents |
|---|---|
| `~/pipeline.log` | one `say` line per stage transition; `PIPELINE_DONE` at the end. Boot-resume appends here too — it's the preemption audit trail. |
| `~/pipeline_nohup.log` | stdout of the launch invocation (stage lines again) |
| `~/pretrain_run.log`, `~/sft_run.log`, `~/grpo_run.log`, `~/eval_run.log` | stdout per stage: step lines, val lines, sampled generations, **and crash tracebacks** — stage scripts restore stderr before exceptions propagate, so tracebacks are here, NOT in the run dir's `train.log`. **Truncated on every resume** (`>` redirect), so after preemptions these hold only the latest segment — the run dir's `metrics.jsonl` is the complete record (and the input `recover_metrics.py` actually needs is whatever segments you still have). |

### What to watch, per stage

- **Pretrain**: `val/bpb` (byte-true bits-per-byte, comparable across
  runs and tokenizers), `val_domain/*` (is each register being
  learned), `train/entropy_bits` (starts near log2(vocab), falls),
  `train/tok_per_sec`, `moe/*` (expert load balance), and the sampled
  generations in the log — register drift is visible there first.
- **SFT**: `val/bpb` is bits per *assistant byte* (not comparable to
  pretrain bpb), `val/tasks_bpb`, `val_source/*` (per-corpus, every 4th
  eval — identity especially).
- **GRPO**: eval reward (`best.pt` here = highest eval reward, unlike
  other stages), `rollout/dead_frac` (all-zero-advantage groups —
  should fall), `rollout/entropy`, per-task eval numbers in the log.
- **Cross-generation comparisons**: ONLY `eval_checkpoint.py` numbers
  on the reserved test slice are comparable. Mid-run honest number:
  pull the bucket-synced `best.pt` to the laptop and run

      gcloud storage cp gs://crow-391712-transformer-data/runs/pretrain/<run>/best.pt /tmp/<run>-best.pt
      .venv/bin/python scripts/eval_checkpoint.py /tmp/<run>-best.pt --device mps

## Preemption and self-healing

The full loop is automatic — **verify, don't intervene**:

1. Spot preemption stops the VM (mid-write checkpoints are safe:
   atomic tmp+rename; at most `PT_SAVE_EVERY` steps of work is lost).
2. The scheduler retries `instances start` every 15 min. During
   capacity droughts, start attempts fail harmlessly and keep retrying
   — droughts of tens of minutes are normal for this zone.
3. On boot, the crontab waits 60 s and runs `pipeline.sh --resume` with
   the baked env: each stage whose newest run dir has a `latest.pt` but
   no `"event": "end"` in `metrics.jsonl` resumes bit-exactly; finished
   stages are skipped; the next one starts fresh.

Verifying a recovery: `~/pipeline.log` shows a new `resuming <stage>`
line; step lines continue from the checkpoint step. The VM's external
IP will have changed (saved TB links die; `tb_bucket.sh` doesn't care).
The first bucket op after boot can fail with a stale service-account
token — it retries/succeeds on the next kick; only worry on repeated
failures.

Resume semantics to know before improvising: `--resume` picks the
**newest** (mtime) run dir per stage. Killed/aborted run dirs with a
`latest.pt` look resumable — delete them (local *and* bucket copy) if
they must never be picked up, or make sure a newer live run dir exists.

## Manual operations

### Stopping a run deliberately

    # kill pipeline.sh FIRST so it can't advance to the next stage,
    # then the stage process; bracket-trick both patterns (see footguns)
    pkill -f "scripts/pipelin[e].sh"; sleep 1; pkill -f "scripts/pretrai[n].py"
    pgrep -af "pipelin[e]" || echo all dead

Killing never triggers `DONE_CMD` (no trap). If the run is not coming
back: remove the crontab (`crontab -r`), pause the restarter, and
delete the aborted run dir + its `gs://…/runs/<stage>/<run>` mirror so
resume logic and TensorBoard never see it again.

### Restarting after a code fix (mid-generation)

Never edit or scp a file the running pipeline is executing. The safe
cycle: kill (as above) → sync files → verify checksums → relaunch the
identical env command → verify stepping. The crontab needs reinstalling
only if the env changed. A killed-and-relaunched pretrain restarts from
step 1 unless you keep the old run dir as the newest live dir and
relaunch with `--resume`.

### Post-training only (`--post-only`)

`bash scripts/pipeline.sh --post-only` (with recipe env) runs a fresh
SFT → tail → GRPO chain off the newest *finished* pretrain — for
iterating on post-training without touching the base model.

### Other ops, one-liners

| op | command |
|---|---|
| ssh | `gcloud compute ssh kimi3-train --zone=us-central1-a --command='…'` |
| VM start/stop | `gcloud compute instances start\|stop kimi3-train --zone=us-central1-a` |
| restarter pause/resume | `gcloud scheduler jobs pause\|resume kimi3-spot-restart --location=us-central1` |
| data → VM | on the VM: `python scripts/download_data.py` (manifest + sha256) |
| new dataset → bucket | laptop only: `scripts/add_dataset.py` (origins + processing live there) |
| sample a checkpoint | `scripts/sample.py --checkpoint <best.pt> --prompt … --temperature …` |
| serve for the chat UI | `scripts/serve.py` — add `--preamble "You are Lily, a tiny language model."` for gen-3+ models (trained with one); omit for older | 
| audit training data | `scripts/sample_data.py` (stage-aware; a 500-conversation random audit beats any dataset card) |
| rebuild TB from metrics | `scripts/rebuild_tb.py <run-dir>` (`tb/` is derived; `metrics.jsonl` is truth) |
| rebuild metrics from logs | `scripts/recover_metrics.py <stage log> <run-dir>` then `rebuild_tb.py` (disaster path — stdout logs carry the full metric history) |

## Completion

`DONE_CMD` fires after the offline forgetting evals. Confirm both
halves: scheduler `PAUSED` (`jobs describe`), VM `TERMINATED`
(`instances describe`). Then the post-run ritual: fill the Results
checklist in the generation's run doc, add a dated `NOTEBOOK.md` entry
(format in its header), update the project memories, pull the three
`best.pt`s (pretrain/sft/rlvr) to the laptop, and run the chat battery
against the previous generation.

## Footguns (each learned the hard way — details in NOTEBOOK.md)

- **`pkill -f` over ssh kills its own session** when any part of the
  remote command line matches the pattern — bracket-trick the pattern
  (`pipelin[e]`) *and* keep the plain string out of every other part of
  the command (an echo label saying "pipeline" is enough to match).
- **Never sync code into a running pipeline.** bash reads scripts
  incrementally from disk; replacing `pipeline.sh` mid-run corrupts the
  read. Sync only between runs (or after a deliberate kill).
- **Boot-resume crontab: only after launch, only with matching env.**
  It fires on every boot and trusts the newest run dirs + its baked env.
- **Never blanket-rsync bucket → local over a live run dir.** The
  lagged bucket copy replaces the growing event file via rename; the
  writer keeps appending to the unlinked inode and the visible file
  freezes. (VM-side `~/pull_runs.sh` was rewritten to only pull run
  dirs that don't exist locally.)
- **Crash tracebacks are in `~/<stage>_run.log`**, not the run dir.
- **First bucket op after boot** can hit a stale-token race — retry
  once before diagnosing.
- **Bucket writes fail silently** if the VM's service-account scope is
  read-only — after any scope change, verify with a small test write
  and `gcloud storage ls gs://…/runs/`.
- **Script defaults are last generation's recipe.** Any `pipeline.sh`
  invocation without the full env (launch, `--resume`, `--post-only`,
  `--install-boot-resume`) silently runs the wrong recipe.
- **Spot start failures are normal.** "Not enough resources" just means
  retry later; the restarter does this for you — resume it and wait.

## Rebuilding the VM from scratch (if it's ever lost)

Same-image recipe: image family `common-cu129-ubuntu-2404-nvidia-580`,
`apt install python3.12-venv python3.12-dev` (triton compiles need
`Python.h`), venv with `torch`, `flash-linear-attention`, `tokenizers`,
`gcsfs`; service account scopes must include `storage-rw` (verify with
a test write — see footguns); ship the repo by tar-over-ssh; then
`scripts/download_data.py`. Throughput/memory reference points (L4,
fp32+TF32, seq 2048): kimi3-small b12 ≈ 11.3k tok/s @ 21.2 GiB;
kimi3-medium b5 ≈ 4.1k tok/s @ 20.1 GiB (b6 OOMs); GRPO is the memory
high-water mark (policy + frozen reference + rollout caches).
