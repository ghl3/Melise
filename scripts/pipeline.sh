#!/bin/bash
# Run the full training pipeline: pretrain -> SFT -> GRPO, resume-aware.
#
#     bash scripts/pipeline.sh                # fresh generation, defaults below
#     bash scripts/pipeline.sh --resume       # continue after crash/preemption
#     bash scripts/pipeline.sh --post-only    # fresh SFT->GRPO from the newest
#                                             # FINISHED pretrain (skip stage 1)
#     PRESET=kimi3-medium TOKENIZER=bpe8k PT_STEPS=145000 bash scripts/pipeline.sh
#
# Every knob is an environment variable (defaults = the gen-4 draft
# recipe, docs/runs/gen4-medium-wide.md — but ALWAYS pass the full env
# explicitly at launch; defaults drift between generations). Each stage
# runs to completion before the next starts; a stage whose newest run
# dir lacks an "end" event in metrics.jsonl is resumed from its
# latest.pt. Downstream stages must also CHAIN to the current upstream
# run (lineage check) — a finished-looking dir from an earlier
# generation or a toy validation is never trusted (the gen-3
# skipped-post-training incident). Logs: ~/pipeline.log + ~/<stage>_run.log.
#
# Spot/preemptible VMs: install the boot hook once —
#     bash scripts/pipeline.sh --install-boot-resume
# which adds a @reboot crontab entry running `pipeline.sh --resume`, so a
# preempted multi-day run continues by itself when the VM restarts
# (checkpoints resume bit-exactly; see README).

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Recipe (override via env; gen-4 draft — freeze after the VM probe) ----
PRESET="${PRESET:-kimi3-medium-wide}"
TOKENIZER="${TOKENIZER:-bpe8k}"
DEVICE="${DEVICE:-cuda}"
PT_STEPS="${PT_STEPS:-268500}"      # 2.2B tokens at batch 4 x seq 2048. BATCH AND STEPS FREEZE TOGETHER.
PT_BATCH="${PT_BATCH:-4}"           # PROBED 2026-08-17: b5 OOMs on step 1 (AttnRes forward, 21.6 GiB
                                    # peak even before optimizer state settles) — b4 is the ceiling at
                                    # d=512/40exp. (b5 was gen-3's d=384 ceiling; width costs a notch.)
PT_SEQ="${PT_SEQ:-2048}"
PT_LR="${PT_LR:-2.5e-4}"            # mild width-aware reduction from gen-3's 3e-4 at d=384
PT_LR_SCHEDULE="${PT_LR_SCHEDULE:-wsd}"   # hold at peak, linear decay over the final PT_DECAY_FRAC
PT_DECAY_FRAC="${PT_DECAY_FRAC:-0.15}"
PT_EVAL_WINDOWS="${PT_EVAL_WINDOWS:-16}"  # pinned windows per val domain (deterministic evals)
PT_KEEP_EVERY="${PT_KEEP_EVERY:-25000}"   # milestone checkpoints exempt from pruning
PROBE_EVERY="${PROBE_EVERY:-4000}"        # pretrain probe cadence. MEASURED: ~4 min/round on the L4
                                          # (trimmed battery) -> 67 rounds ~= 2% of a ~10d pretrain
DATA_MIX="${DATA_MIX:-configs/mix-gen4-chat.json}"
SFT_STEPS="${SFT_STEPS:-25000}"     # 25k x b4 = gen-3's example exposure (20k x b5)
SFT_BATCH="${SFT_BATCH:-4}"         # PROBED 2026-08-17: b5 OOMs in SFT's first backward even without
                                    # a resident corpus — b5 activations outweigh the 2 GB saved. b4.
SFT_SEQ="${SFT_SEQ:-2048}"
SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-200}"  # best.pt only lands at evals — keep < SFT_STEPS
SFT_PROBE_EVERY="${SFT_PROBE_EVERY:-2500}"  # chat rounds ~2x raw; 8 rounds ~= 4-5% of SFT — the stage
                                            # where probes matter most (identity destruction happens here)
SFT_REPLAY_FRAC="${SFT_REPLAY_FRAC:-0.03}"   # raw pretrain-mix batches vs SFT forgetting
SFT_REPLAY_MIX="${SFT_REPLAY_MIX:-$DATA_MIX}"
# Task-focus tail: a short low-LR pass over task-format data only, so the
# formats SFT merely *models* become what it *generates* (gen-2 lesson:
# tasks_bpb 1.04 yet zero worked-steps rollouts). 0 disables.
SFT_TAIL_STEPS="${SFT_TAIL_STEPS:-1500}"
SFT_TAIL_LR="${SFT_TAIL_LR:-3e-5}"
SFT_TAIL_DATA="${SFT_TAIL_DATA:-data/chat_tasks.txt data/chat_identity.txt data/chat_facts.txt}"
RLVR_STEPS="${RLVR_STEPS:-600}"
RLVR_LR="${RLVR_LR:-1e-5}"
RLVR_PROBE_EVERY="${RLVR_PROBE_EVERY:-200}"
# Pretrain eval/checkpoint cadence — a ~270k-step run at ~30 min per 500
# steps; 250/500 (gen-3) would double the eval count for no insight.
PT_EVAL_EVERY="${PT_EVAL_EVERY:-500}"
PT_SAVE_EVERY="${PT_SAVE_EVERY:-1000}"
# Optional command run after the final evals. On-demand gen-4 default:
# shutdown only — the VM-side scheduler pause NEVER works (VM scopes
# lack cloudscheduler; it fails silently) — keep the restarter paused
# from the laptop instead.
DONE_CMD="${DONE_CMD:-}"

LOG=~/pipeline.log
say() { echo "[pipeline $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ "${1:-}" = "--install-boot-resume" ]; then
    # Bake the CURRENT recipe env into the crontab line — a bare
    # `--resume` would fall back to the defaults above, and e.g. a
    # smaller default PT_STEPS makes an interrupted long pretrain look
    # finished (resume would skip ahead to SFT on an unfinished base).
    # Invoke this with the SAME env as the launch command.
    ENV_STR=""
    for v in PRESET TOKENIZER DEVICE PT_STEPS PT_BATCH PT_SEQ PT_LR \
             PT_LR_SCHEDULE PT_DECAY_FRAC PT_EVAL_WINDOWS PT_KEEP_EVERY \
             PROBE_EVERY DATA_MIX \
             SFT_STEPS SFT_BATCH SFT_SEQ SFT_EVAL_EVERY SFT_PROBE_EVERY \
             SFT_REPLAY_FRAC SFT_REPLAY_MIX SFT_TAIL_STEPS SFT_TAIL_LR \
             SFT_TAIL_DATA RLVR_STEPS RLVR_LR RLVR_PROBE_EVERY \
             PT_EVAL_EVERY PT_SAVE_EVERY DONE_CMD; do
        ENV_STR="$ENV_STR $v=$(printf %q "${!v}")"
    done
    LINE="@reboot sleep 60 && cd $REPO &&$ENV_STR bash scripts/pipeline.sh --resume >> ~/pipeline.log 2>&1"
    (crontab -l 2>/dev/null | grep -vF "pipeline.sh --resume"; echo "$LINE") | crontab -
    say "boot-resume crontab installed: $LINE"
    exit 0
fi
RESUME_MODE=$([ "${1:-}" = "--resume" ] && echo 1 || echo 0)
POST_ONLY=$([ "${1:-}" = "--post-only" ] && echo 1 || echo "${POST_ONLY:-0}")

newest() { ls -td "$REPO/checkpoints/$1"/*/ 2>/dev/null | head -1; }
stage_done() {  # run dir finished cleanly?
    [ -n "$1" ] && grep -q '"event": "end"' "$1/metrics.jsonl" 2>/dev/null
}
stage_live() {  # resumable run dir?
    [ -n "$1" ] && [ -e "$1/latest.pt" ] && ! stage_done "$1"
}
chain_ok() {  # $1 = downstream run dir, $2 = upstream run-dir basename.
    # run.json's "lineage" lists run names from the pretrain root down,
    # so a downstream dir that doesn't mention the CURRENT upstream run
    # belongs to another generation (or a toy validation) — never trust
    # it. This is the guard the gen-3 incident was missing: stage_done()
    # alone saw gen-2's sft/rlvr dirs as newest-per-stage and skipped
    # all gen-3 post-training.
    [ -n "$1" ] && [ -n "$2" ] && grep -q "$2" "$1/run.json" 2>/dev/null
}

# ---- Stage 1: pretrain ----
PT_DIR=$(newest pretrain)
if [ "$POST_ONLY" = 1 ]; then
    stage_done "$PT_DIR" || { say "post-only: no finished pretrain"; exit 1; }
    say "post-only: reusing pretrain $PT_DIR"
elif [ "$RESUME_MODE" = 1 ] && stage_live "$PT_DIR"; then
    say "resuming pretrain: $PT_DIR"
    $PY scripts/pretrain.py --resume "$PT_DIR/latest.pt" --steps "$PT_STEPS" \
        --batch-size "$PT_BATCH" --data-mix "$DATA_MIX" \
        --lr "$PT_LR" --lr-schedule "$PT_LR_SCHEDULE" --decay-frac "$PT_DECAY_FRAC" \
        --eval-every "$PT_EVAL_EVERY" --save-every "$PT_SAVE_EVERY" \
        --eval-windows "$PT_EVAL_WINDOWS" --keep-every "$PT_KEEP_EVERY" \
        --probe-every "$PROBE_EVERY" \
        --device "$DEVICE" > ~/pretrain_run.log 2>&1
elif [ "$RESUME_MODE" = 1 ] && stage_done "$PT_DIR"; then
    say "pretrain already done: $PT_DIR"
else
    say "fresh pretrain: $PRESET/$TOKENIZER, $PT_STEPS steps @ seq $PT_SEQ ($PT_LR_SCHEDULE)"
    $PY scripts/pretrain.py --preset "$PRESET" --tokenizer "$TOKENIZER" \
        --steps "$PT_STEPS" --batch-size "$PT_BATCH" --seq-len "$PT_SEQ" \
        --data-mix "$DATA_MIX" \
        --lr "$PT_LR" --lr-schedule "$PT_LR_SCHEDULE" --decay-frac "$PT_DECAY_FRAC" \
        --eval-every "$PT_EVAL_EVERY" --save-every "$PT_SAVE_EVERY" \
        --eval-windows "$PT_EVAL_WINDOWS" --keep-every "$PT_KEEP_EVERY" \
        --probe-every "$PROBE_EVERY" \
        --device "$DEVICE" > ~/pretrain_run.log 2>&1
fi
RC=$?
PT_DIR=$(newest pretrain)
say "pretrain exit $RC ($PT_DIR)"
[ -e "$PT_DIR/best.pt" ] || { say "no pretrain best.pt — aborting"; exit 1; }
PT_RUN=$(basename "$PT_DIR")

# ---- Stage 2: SFT ----
SFT_DIR=$(newest sft)
if [ "$RESUME_MODE" = 1 ] && stage_live "$SFT_DIR" && chain_ok "$SFT_DIR" "$PT_RUN"; then
    say "resuming sft: $SFT_DIR"
    $PY scripts/sft.py --resume "$SFT_DIR/latest.pt" --steps "$SFT_STEPS" \
        --eval-every "$SFT_EVAL_EVERY" --batch-size "$SFT_BATCH" \
        --replay-frac "$SFT_REPLAY_FRAC" --replay-mix "$SFT_REPLAY_MIX" \
        --probe-every "$SFT_PROBE_EVERY" --device "$DEVICE" > ~/sft_run.log 2>&1
elif [ "$RESUME_MODE" = 1 ] && stage_done "$SFT_DIR" && chain_ok "$SFT_DIR" "$PT_RUN"; then
    say "sft already done: $SFT_DIR"
else
    say "fresh sft from $PT_DIR/best.pt"
    $PY scripts/sft.py --init "$PT_DIR/best.pt" --steps "$SFT_STEPS" \
        --eval-every "$SFT_EVAL_EVERY" --batch-size "$SFT_BATCH" --seq-len "$SFT_SEQ" \
        --replay-frac "$SFT_REPLAY_FRAC" --replay-mix "$SFT_REPLAY_MIX" \
        --probe-every "$SFT_PROBE_EVERY" --device "$DEVICE" > ~/sft_run.log 2>&1
fi
RC=$?
SFT_DIR=$(newest sft)
say "sft exit $RC ($SFT_DIR)"
[ -e "$SFT_DIR/best.pt" ] || { say "no sft best.pt — aborting"; exit 1; }
chain_ok "$SFT_DIR" "$PT_RUN" || { say "sft dir $SFT_DIR does not chain to $PT_RUN — aborting"; exit 1; }

# ---- Stage 2b: task-focus SFT tail (short, low LR, task data only) ----
# Tail run dirs are named <main-run>-tail, so resume logic can tell the
# two apart: a live tail resumes via stage 2's own resume branch; a done
# tail is skipped here; a done main run with no tail yet starts one.
if [ "$SFT_TAIL_STEPS" -gt 0 ]; then case "$(basename "$SFT_DIR")" in
    *-tail) say "sft tail already done: $SFT_DIR" ;;
    *)
        TAIL_NAME="$(basename "$SFT_DIR")-tail"
        TAIL_DIR="$REPO/checkpoints/sft/$TAIL_NAME"
        if stage_done "$TAIL_DIR/" && chain_ok "$TAIL_DIR" "$PT_RUN"; then
            say "sft tail already done: $TAIL_DIR"
        else
            TAIL_DATA_ARGS=""
            for f in $SFT_TAIL_DATA; do TAIL_DATA_ARGS="$TAIL_DATA_ARGS --data $f"; done
            say "sft task tail from $SFT_DIR/best.pt ($SFT_TAIL_STEPS steps @ lr $SFT_TAIL_LR)"
            $PY scripts/sft.py --init "$SFT_DIR/best.pt" --steps "$SFT_TAIL_STEPS" \
                --eval-every "$SFT_EVAL_EVERY" --batch-size "$SFT_BATCH" --seq-len "$SFT_SEQ" --lr "$SFT_TAIL_LR" \
                --replay-frac "$SFT_REPLAY_FRAC" --replay-mix "$SFT_REPLAY_MIX" \
                --probe-every "$SFT_PROBE_EVERY" \
                $TAIL_DATA_ARGS --run-name "$TAIL_NAME" \
                --device "$DEVICE" >> ~/sft_run.log 2>&1
            say "sft tail exit $?"
        fi
        [ -e "$TAIL_DIR/best.pt" ] || { say "no tail best.pt — aborting"; exit 1; }
        SFT_DIR="$TAIL_DIR"
        ;;
esac; fi
SFT_RUN=$(basename "$SFT_DIR")

# ---- Stage 3: GRPO ----
RLVR_DIR=$(newest rlvr)
if [ "$RESUME_MODE" = 1 ] && stage_live "$RLVR_DIR" && chain_ok "$RLVR_DIR" "$SFT_RUN"; then
    say "resuming grpo: $RLVR_DIR"
    $PY scripts/grpo.py --resume "$RLVR_DIR/latest.pt" --steps "$RLVR_STEPS" \
        --probe-every "$RLVR_PROBE_EVERY" --device "$DEVICE" > ~/grpo_run.log 2>&1
elif [ "$RESUME_MODE" = 1 ] && stage_done "$RLVR_DIR" && chain_ok "$RLVR_DIR" "$SFT_RUN"; then
    say "grpo already done: $RLVR_DIR"
else
    say "fresh grpo from $SFT_DIR/best.pt"
    $PY scripts/grpo.py --init "$SFT_DIR/best.pt" --steps "$RLVR_STEPS" \
        --lr "$RLVR_LR" --probe-every "$RLVR_PROBE_EVERY" \
        --device "$DEVICE" > ~/grpo_run.log 2>&1
fi
RC=$?
RLVR_DIR=$(newest rlvr)
say "grpo exit $RC ($RLVR_DIR)"

# ---- Post-stage evals: virgin test-slice bpb for every stage's best ----
# (catastrophic-forgetting check: did SFT/RLVR damage the base LM?)
# eval_checkpoint pushes evals.jsonl to the bucket itself — BucketSync's
# last kick fires before this stage appends (gen-3 incident #2).
say "offline evals on the enwik8 test slice"
$PY scripts/eval_checkpoint.py "$PT_DIR/best.pt" "$SFT_DIR/best.pt" \
    "$RLVR_DIR/best.pt" --data-mix "$DATA_MIX" --device "$DEVICE" > ~/eval_run.log 2>&1
say "evals exit $? — see ~/eval_run.log and <run>/evals.jsonl"
# Fully-held-out books (never trained): uncontaminated register
# generalization, free of the within-book-val familiarity flattery.
for book in data/emma.txt data/great_expectations.txt; do
    [ -e "$book" ] || continue
    say "holdout eval: $book"
    $PY scripts/eval_checkpoint.py "$PT_DIR/best.pt" "$SFT_DIR/best.pt" \
        "$RLVR_DIR/best.pt" --holdout --data "$book" \
        --device "$DEVICE" >> ~/eval_run.log 2>&1
done
# Full probe battery on each stage's best (full facts table — the
# in-loop rounds subsample); probes.jsonl pushes itself.
say "offline probe battery"
for ck in "$PT_DIR/best.pt" "$SFT_DIR/best.pt" "$RLVR_DIR/best.pt"; do
    [ -e "$ck" ] || continue
    $PY scripts/probe_checkpoint.py "$ck" --out --device "$DEVICE" >> ~/eval_run.log 2>&1
done
say "offline probes exit $?"
say "PIPELINE_DONE"
if [ -n "$DONE_CMD" ]; then
    say "running DONE_CMD: $DONE_CMD"
    eval "$DONE_CMD"
fi
