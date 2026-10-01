#!/bin/bash
#SBATCH --job-name=cobit_win
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# WINDOW temporal ordering, from scratch. One arm per job (or chain of jobs):
#
#   WIN_MODE=none   WIN_K=inf      control (uniform, through the same window path)
#   WIN_MODE=l2r|r2l|random  WIN_K=64|16|4
#   WIN_LOSSMATCH=64|16|4 (with WIN_MODE=none) supervision-matched control
#   WIN_SEED (42, 43, ...)  WIN_STEPS  WIN_TAG  WIN_CKPT_EVERY
#
# Submit through window_study.sh, which chains links; a link resumes from
# runs/<experiment>/checkpoints/last.pt automatically.
#
# COMPUTE SAFETY. A link never spends a node on a run that has already failed:
#   * runs/<exp>/HALT (or runs/tasks/tinygsm/WINDOW_GLOBAL_HALT) -> exit at once
#     and cancel this run's pending links;
#   * already at WIN_STEPS -> exit, cancel the rest;
#   * after training, every link records start/end step and rc in
#     runs/<exp>/links/<jobid>.txt and, unless the outcome is understood, writes
#     HALT and cancels its own pending successors:
#       rc=0, reached WIN_STEPS        DONE
#       rc=0, no progress               HALT
#       divergence guard                HALT (a result: never resumed)
#       NCCL hang after >=5k steps      next link resumes (the known deadlock)
#       anything else                   HALT
#   SIGTERM (wall clock, scancel) is trapped and is the normal end of a
#   non-final link: never a HALT. Any other exit before the epilogue HALTs,
#   and two earlier links that made no progress HALT before a node is used.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch

export WIN_MODE="${WIN_MODE:?set WIN_MODE=none|l2r|r2l|random}"
export WIN_K="${WIN_K:?set WIN_K=inf|64|16|4}"
export WIN_SEED="${WIN_SEED:-42}"
export WIN_STEPS="${WIN_STEPS:-500000}"
export WIN_TAG="${WIN_TAG:-win}"
export WIN_LOSSMATCH="${WIN_LOSSMATCH:-}"
CFG="configs/tasks/tinygsm_bits_window.py"

KT="k${WIN_K}"; [ -n "$WIN_LOSSMATCH" ] && KT="${KT}_lm${WIN_LOSSMATCH}"
RUN_NAME="${WIN_TAG}_${WIN_MODE}_${KT}_s${WIN_SEED}"
RUN_DIR="runs/tasks/tinygsm/$RUN_NAME"
LOG="$PROJECT_DIR/logs/arch/${SLURM_JOB_NAME:-cobit_win}_${SLURM_JOB_ID:-local}.log"
mkdir -p "$RUN_DIR/links"
LINKREC="$RUN_DIR/links/${SLURM_JOB_ID:-local}.txt"

ckpt_step () {
  [ -f "$RUN_DIR/checkpoints/last.pt" ] || { echo 0; return; }
  "$COBIT_PYTHON" - "$RUN_DIR/checkpoints/last.pt" <<'PYEOF'
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["global_step"]))
except Exception:
    print(-1)
PYEOF
}
cancel_rest () { scancel --name="${SLURM_JOB_NAME:-none}" --state=PENDING 2>/dev/null || true; }
halt () {
  echo "$(date -Is) job=${SLURM_JOB_ID:-local} $1" >> "$RUN_DIR/HALT"
  echo "[win] HALT: $1"
  cancel_rest
}

if [ -f "$RUN_DIR/HALT" ] || [ -f runs/tasks/tinygsm/WINDOW_GLOBAL_HALT ]; then
  echo "[win] refusing to start: $(cat "$RUN_DIR/HALT" runs/tasks/tinygsm/WINDOW_GLOBAL_HALT 2>/dev/null | tail -1)"
  cancel_rest; exit 0
fi
STEP0=$(ckpt_step)
if [ "$STEP0" -lt 0 ]; then halt "last.pt unreadable before start"; exit 0; fi
if [ "$STEP0" -ge "$WIN_STEPS" ]; then
  echo "[win] already at step $STEP0 >= $WIN_STEPS -- done, not starting"; cancel_rest; exit 0
fi
# Links killed without reaching the epilogue (node failure, OOM-kill of the
# batch shell) leave only a start record. Two earlier links that both started
# at today's step mean two links of zero progress: stop.
prev=$(for f in $(ls -t "$RUN_DIR"/links/*.txt 2>/dev/null | head -2 || true); do
         sed -n 's/^start_step=\([0-9-]*\).*/\1/p' "$f"; done | grep -cx "$STEP0" || true)
if [ "${prev:-0}" -ge 2 ]; then halt "no progress across the last 2 links (stuck at step $STEP0)"; exit 0; fi
echo "start_step=$STEP0 start=$(date -Is)" > "$LINKREC"

# SIGTERM = wall-clock limit or scancel: the normal end of a non-final link,
# never a reason to HALT. Any OTHER exit before the epilogue is unexplained.
TERMINATED=0; EPILOGUE=0
trap 'TERMINATED=1' TERM
on_exit () {
  stop_stall_watchdog 2>/dev/null || true
  if [ "$EPILOGUE" != 1 ] && [ "$TERMINATED" != 1 ]; then
    halt "launcher exited before the epilogue (line ${BASH_LINENO[0]:-?})"
  fi
}
trap on_exit EXIT

source "$PROJECT_DIR/scripts/hpc/arch/deadlock_debug.sh"
export DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-20}"
enable_nccl_flight_recorder "$PROJECT_DIR/logs/arch"
start_stall_watchdog "$LOG" "${STALL_MIN:-15}"

echo "[win] run=$RUN_NAME commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain | grep -c . || true)"
echo "[win] python $($COBIT_PYTHON -V 2>&1)"
echo "[win] mode=$WIN_MODE k=$WIN_K lossmatch=${WIN_LOSSMATCH:-none} seed=$WIN_SEED steps=$WIN_STEPS start_step=$STEP0"
echo "[win] noise table sha256 $(sha256sum assets/noise_tables/tinygsm_window_F_v1.pt | cut -c1-16)"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader || true
compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null || { EPILOGUE=1; halt "no corpus"; exit 0; }

set +e
srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${WIN_NPROC:-4}" \
     train.py --config "$CFG"
rc=$?
set -e
EPILOGUE=1
STEP1=$(ckpt_step)
echo "end_step=$STEP1 rc=$rc terminated=$TERMINATED end=$(date -Is)" >> "$LINKREC"
echo "[win] rc=$rc steps $STEP0 -> $STEP1 terminated=$TERMINATED"

if [ "$TERMINATED" = 1 ]; then
  echo "[win] terminated by SLURM (wall clock or scancel) -- the next link resumes"
elif [ $rc -eq 0 ]; then
  if [ "$STEP1" -ge "$WIN_STEPS" ]; then echo "[win] DONE"; cancel_rest
  elif [ "$STEP1" -le "$STEP0" ]; then halt "rc=0 but no progress ($STEP0 -> $STEP1)"; fi
elif grep -q "\[divergence-guard\]" "$LOG" 2>/dev/null; then
  halt "DIVERGED near step $STEP1 (divergence guard)"
elif grep -qE "collective operation timeout|WorkNCCL.*timing out|DistBackendError|NCCL.*[Tt]imeout" "$LOG" 2>/dev/null \
     && [ $((STEP1 - STEP0)) -ge 5000 ]; then
  echo "[win] NCCL hang after $((STEP1 - STEP0)) steps of progress; the next link resumes"
else
  halt "crash rc=$rc near step $STEP1: $(grep -m1 -E 'Error|Traceback|Killed' "$LOG" 2>/dev/null | cut -c1-160)"
fi
echo "[win] finished $(date)"
exit 0
