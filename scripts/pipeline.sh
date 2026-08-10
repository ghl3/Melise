#!/bin/bash
# Run the full training pipeline: pretrain -> SFT -> GRPO, resume-aware.
#
#     bash scripts/pipeline.sh                # fresh generation, defaults below
#     bash scripts/pipeline.sh --resume       # continue after crash/preemption
#     bash scripts/pipeline.sh --post-only    # fresh SFT->GRPO from the newest
#                                             # FINISHED pretrain (skip stage 1)
#     PRESET=kimi3 TOKENIZER=bpe4k PT_STEPS=25000 bash scripts/pipeline.sh
#
# Every knob is an environment variable (defaults = the next-gen recipe).
# Each stage runs to completion before the next starts; a stage whose
# newest run dir lacks an "end" event in metrics.jsonl is resumed from
# its latest.pt. Logs: ~/pipeline.log + ~/<stage>_run.log.
#
# Spot/preemptible VMs: install the boot hook once —
#     bash scripts/pipeline.sh --install-boot-resume
# which adds a @reboot crontab entry running `pipeline.sh --resume`, so a
# preempted multi-day run continues by itself when the VM restarts
# (checkpoints resume bit-exactly; see README). Combine with an L4 Spot
# instance for ~1/3 GPU cost on long runs.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Recipe (override via env) ----
PRESET="${PRESET:-kimi3}"
TOKENIZER="${TOKENIZER:-bpe4k}"
DEVICE="${DEVICE:-cuda}"
PT_STEPS="${PT_STEPS:-33333}"       # ~819M tokens at batch 12 x seq 2048
PT_BATCH="${PT_BATCH:-12}"          # batch 16 @ seq 2048 fp32 OOMs the 22 GiB L4
PT_SEQ="${PT_SEQ:-2048}"
DATA_MIX="${DATA_MIX:-configs/mix-downweight-wiki.json}"
SFT_STEPS="${SFT_STEPS:-12000}"     # 12000 x batch 12 = same examples as 9000 x 16
SFT_BATCH="${SFT_BATCH:-12}"        # sft.py's own default (16) doesn't fit either
SFT_SEQ="${SFT_SEQ:-2048}"
# Task-focus tail: a short low-LR pass over task-format data only, so the
# formats SFT merely *models* become what it *generates* (gen-2 lesson:
# tasks_bpb 1.04 yet zero worked-steps rollouts). 0 disables.
SFT_TAIL_STEPS="${SFT_TAIL_STEPS:-1500}"
SFT_TAIL_LR="${SFT_TAIL_LR:-3e-5}"
SFT_TAIL_DATA="${SFT_TAIL_DATA:-data/chat_tasks.txt data/chat_identity.txt}"
RLVR_STEPS="${RLVR_STEPS:-600}"
RLVR_LR="${RLVR_LR:-1e-5}"

LOG=~/pipeline.log
say() { echo "[pipeline $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ "${1:-}" = "--install-boot-resume" ]; then
    LINE="@reboot sleep 60 && bash $REPO/scripts/pipeline.sh --resume >> ~/pipeline.log 2>&1"
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

# ---- Stage 1: pretrain ----
PT_DIR=$(newest pretrain)
if [ "$POST_ONLY" = 1 ]; then
    stage_done "$PT_DIR" || { say "post-only: no finished pretrain"; exit 1; }
    say "post-only: reusing pretrain $PT_DIR"
elif [ "$RESUME_MODE" = 1 ] && stage_live "$PT_DIR"; then
    say "resuming pretrain: $PT_DIR"
    $PY scripts/pretrain.py --resume "$PT_DIR/latest.pt" --steps "$PT_STEPS" \
        --batch-size "$PT_BATCH" --data-mix "$DATA_MIX" \
        --device "$DEVICE" > ~/pretrain_run.log 2>&1
elif [ "$RESUME_MODE" = 1 ] && stage_done "$PT_DIR"; then
    say "pretrain already done: $PT_DIR"
else
    say "fresh pretrain: $PRESET/$TOKENIZER, $PT_STEPS steps @ seq $PT_SEQ"
    $PY scripts/pretrain.py --preset "$PRESET" --tokenizer "$TOKENIZER" \
        --steps "$PT_STEPS" --batch-size "$PT_BATCH" --seq-len "$PT_SEQ" \
        --data-mix "$DATA_MIX" --device "$DEVICE" > ~/pretrain_run.log 2>&1
fi
PT_DIR=$(newest pretrain)
say "pretrain exit $? ($PT_DIR)"
[ -e "$PT_DIR/best.pt" ] || { say "no pretrain best.pt — aborting"; exit 1; }

# ---- Stage 2: SFT ----
SFT_DIR=$(newest sft)
if [ "$RESUME_MODE" = 1 ] && stage_live "$SFT_DIR"; then
    say "resuming sft: $SFT_DIR"
    $PY scripts/sft.py --resume "$SFT_DIR/latest.pt" --steps "$SFT_STEPS" \
        --batch-size "$SFT_BATCH" --device "$DEVICE" > ~/sft_run.log 2>&1
elif [ "$RESUME_MODE" = 1 ] && stage_done "$SFT_DIR"; then
    say "sft already done: $SFT_DIR"
else
    say "fresh sft from $PT_DIR/best.pt"
    $PY scripts/sft.py --init "$PT_DIR/best.pt" --steps "$SFT_STEPS" \
        --batch-size "$SFT_BATCH" --seq-len "$SFT_SEQ" \
        --device "$DEVICE" > ~/sft_run.log 2>&1
fi
SFT_DIR=$(newest sft)
say "sft exit $? ($SFT_DIR)"
[ -e "$SFT_DIR/best.pt" ] || { say "no sft best.pt — aborting"; exit 1; }

# ---- Stage 2b: task-focus SFT tail (short, low LR, task data only) ----
# Tail run dirs are named <main-run>-tail, so resume logic can tell the
# two apart: a live tail resumes via stage 2's own resume branch; a done
# tail is skipped here; a done main run with no tail yet starts one.
if [ "$SFT_TAIL_STEPS" -gt 0 ]; then case "$(basename "$SFT_DIR")" in
    *-tail) say "sft tail already done: $SFT_DIR" ;;
    *)
        TAIL_NAME="$(basename "$SFT_DIR")-tail"
        TAIL_DIR="$REPO/checkpoints/sft/$TAIL_NAME"
        if stage_done "$TAIL_DIR/"; then
            say "sft tail already done: $TAIL_DIR"
        else
            TAIL_DATA_ARGS=""
            for f in $SFT_TAIL_DATA; do TAIL_DATA_ARGS="$TAIL_DATA_ARGS --data $f"; done
            say "sft task tail from $SFT_DIR/best.pt ($SFT_TAIL_STEPS steps @ lr $SFT_TAIL_LR)"
            $PY scripts/sft.py --init "$SFT_DIR/best.pt" --steps "$SFT_TAIL_STEPS" \
                --batch-size "$SFT_BATCH" --seq-len "$SFT_SEQ" --lr "$SFT_TAIL_LR" \
                $TAIL_DATA_ARGS --run-name "$TAIL_NAME" \
                --device "$DEVICE" >> ~/sft_run.log 2>&1
            say "sft tail exit $?"
        fi
        [ -e "$TAIL_DIR/best.pt" ] || { say "no tail best.pt — aborting"; exit 1; }
        SFT_DIR="$TAIL_DIR"
        ;;
esac; fi

# ---- Stage 3: GRPO ----
RLVR_DIR=$(newest rlvr)
if [ "$RESUME_MODE" = 1 ] && stage_live "$RLVR_DIR"; then
    say "resuming grpo: $RLVR_DIR"
    $PY scripts/grpo.py --resume "$RLVR_DIR/latest.pt" --steps "$RLVR_STEPS" \
        --device "$DEVICE" > ~/grpo_run.log 2>&1
elif [ "$RESUME_MODE" = 1 ] && stage_done "$RLVR_DIR"; then
    say "grpo already done: $RLVR_DIR"
else
    say "fresh grpo from $SFT_DIR/best.pt"
    $PY scripts/grpo.py --init "$SFT_DIR/best.pt" --steps "$RLVR_STEPS" \
        --lr "$RLVR_LR" --device "$DEVICE" > ~/grpo_run.log 2>&1
fi
RLVR_DIR=$(newest rlvr)
say "grpo exit $? ($RLVR_DIR)"

# ---- Post-stage evals: virgin test-slice bpb for every stage's best ----
# (catastrophic-forgetting check: did SFT/RLVR damage the base LM?)
say "offline evals on the enwik8 test slice"
$PY scripts/eval_checkpoint.py "$PT_DIR/best.pt" "$SFT_DIR/best.pt" \
    "$RLVR_DIR/best.pt" --device "$DEVICE" > ~/eval_run.log 2>&1
say "evals exit $? — see ~/eval_run.log and <run>/evals.jsonl"
say "PIPELINE_DONE"
