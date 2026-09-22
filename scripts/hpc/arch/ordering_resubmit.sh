#!/bin/bash
# Replace the queued ordering links with instrumented ones.
#
# WHY. SLURM copies a batch script at submission time, so the links queued
# before scripts/hpc/arch/deadlock_debug.sh existed carry none of it: no NCCL
# flight recorder, no SIGUSR1 stack dump, and DDP_TIMEOUT_MIN=120, which burns
# two hours of the reservation on every hang before the job even dies. Every job
# so far has hung after exactly five completed epochs, so that is a two-hour tax
# roughly every twenty hours of compute, per arm.
#
#   bash scripts/hpc/arch/ordering_resubmit.sh            # show what it would do
#   bash scripts/hpc/arch/ordering_resubmit.sh --go       # do it
#
# Only PENDING links are cancelled. A RUNNING job is left alone and the arm's
# new chain is made to depend on it, so no live compute is thrown away and no
# two jobs can ever write to the same run directory -- which has happened here
# before and cost four runs.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
cd "$PROJECT_DIR"

GO=0
[ "${1:-}" = "--go" ] && GO=1
WALL="${WALL:-36:00:00}"

# steps remaining as of 16 Sep: control/l2r 42,840; r2l/random 271,420.
# A link gets ~5 epochs (~114,000 steps) before the deadlock kills it, so the
# link counts are sized on that, not on the wall clock.
ARMS=("none:0:2" "l2r:0.25:2" "r2l:0.25:4" "random:0.25:4")

echo "== current cobit_ordtrain jobs =="
squeue -u "$USER" -n cobit_ordtrain -o "%.10i %.8T %.12l %R" || true
echo

PEND=$(squeue -u "$USER" -n cobit_ordtrain -h -t PENDING -o "%i" | tr '\n' ' ')
echo "pending to cancel: ${PEND:-<none>}"

for spec in "${ARMS[@]}"; do
  IFS=: read -r mode w links <<< "$spec"
  run="runs/tasks/tinygsm/ord_fs500k_$([ "$mode" = none ] && echo none || echo "${mode}_w${w}")_s42"
  echo "  $mode w=$w -> $links links  ($run)"
done

if [ $GO -eq 0 ]; then
  echo
  echo "dry run. Re-run with --go to cancel the pending links and resubmit."
  exit 0
fi

if [ -n "${PEND// /}" ]; then
  scancel $PEND || { echo "scancel failed -- STOPPING" >&2; exit 1; }
  sleep 5
  left=$(squeue -u "$USER" -n cobit_ordtrain -h -t PENDING -o "%i" | tr '\n' ' ')
  # A failed or empty squeue is "unknown", not "zero jobs": a timeout once
  # returned empty and a wait loop read it as "all done".
  if [ $? -ne 0 ]; then echo "squeue failed after scancel -- check by hand" >&2; exit 1; fi
  [ -n "${left// /}" ] && { echo "still pending after scancel: $left -- STOPPING" >&2; exit 1; }
fi

for spec in "${ARMS[@]}"; do
  IFS=: read -r mode w links <<< "$spec"
  # If this arm is still running, chain behind it rather than racing it.
  running=$(squeue -u "$USER" -n cobit_ordtrain -h -t RUNNING -o "%i" | while read -r j; do
    scontrol show job "$j" | tr ' ' '\n' | grep -q "ORD_MODE=$mode\$" && echo "$j"; done | head -1)
  prev="$running"
  [ -n "$prev" ] && echo "[resub] $mode: chaining behind running job $prev"

  EXPORTS="ALL,ORD_MODE=$mode,ORD_W=$w,ORD_INIT=none,ORD_LR=1e-4,ORD_STEPS=500000,ORD_TAG=fs500k"
  for i in $(seq 1 "$links"); do
    if [ -z "$prev" ]; then
      out=$(sbatch --parsable --time="$WALL" --export="$EXPORTS" scripts/hpc/arch/ordering_train.slurm)
    else
      out=$(sbatch --parsable --time="$WALL" --dependency=afterany:"$prev" \
                   --export="$EXPORTS" scripts/hpc/arch/ordering_train.slurm)
    fi
    rc=$?
    if [ $rc -ne 0 ] || [ -z "$out" ]; then
      echo "[resub] sbatch rc=$rc out='$out' for $mode -- STOPPING." >&2
      echo "[resub] rc!=0 means UNKNOWN, not 'not submitted'. Check squeue before retrying." >&2
      exit 1
    fi
    echo "[resub] $mode link $i: $out${prev:+ (afterany:$prev)}"
    prev="$out"
  done
done

echo
squeue -u "$USER" -n cobit_ordtrain -o "%.10i %.8T %.12l %R"
