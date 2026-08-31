#!/bin/bash
#SBATCH --job-name=cobit_guid
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/guidance/%x_%A_%a.log
#
# Generic array runner: one SLURM array task == one cell of one grid.
#
# The grid is the single source of truth (experiments/guidance/grids.py), so the
# array bounds and the cell definitions can never drift apart. Enumerate and
# review a grid BEFORE submitting:
#
#     python -m experiments.guidance.grids show cfg_coarse
#     python -m experiments.guidance.grids size cfg_coarse
#
# Submit (GRID is required; OUT_DIR defaults to runs/guidance/$GRID):
#
#     sbatch --array=0-11 --export=ALL,GRID=cfg_coarse \
#            scripts/hpc/guidance/array.slurm
#
# Failure-safe behaviour:
#   * a cell whose result JSON already exists is skipped, so a partially
#     completed array can be resubmitted wholesale without redoing work or
#     double-counting;
#   * the cell's argv is printed before it runs, so the log alone is enough to
#     reproduce the run by hand;
#   * an out-of-range index is a hard error rather than a silent no-op.
set -euo pipefail

source "$(dirname "$0")/env.sh"

GRID="${GRID:?set GRID=<name>, e.g. GRID=cfg_coarse}"
OUT_DIR="${OUT_DIR:-runs/guidance/$GRID}"
IDX="${SLURM_ARRAY_TASK_ID:?must be run as a SLURM array (--array=...)}"

mkdir -p "$OUT_DIR" logs/guidance

N=$("$COBIT_PYTHON" -m experiments.guidance.grids size "$GRID")
if [ "$IDX" -ge "$N" ]; then
  echo "[cell] index $IDX >= grid size $N for '$GRID' -- array bounds are wrong." >&2
  exit 2
fi

ARGS=$("$COBIT_PYTHON" -m experiments.guidance.grids cli "$GRID" "$IDX" --out_dir "$OUT_DIR")
echo "[cell] grid=$GRID index=$IDX/$N"
echo "[cell] argv: $ARGS"

# Idempotence: the cell name is embedded in the result filename via --tag.
TAG=$(sed -n 's/.*--tag \([^ ]*\).*/\1/p' <<<"$ARGS")
if compgen -G "$OUT_DIR/gsm8k_results_*_${TAG}.json" > /dev/null; then
  echo "[cell] SKIP: a result for tag '$TAG' already exists in $OUT_DIR"
  exit 0
fi

START=$(date +%s)
# shellcheck disable=SC2086
srun "$COBIT_PYTHON" -m evaluation.tasks.gsm8k_eval $ARGS
echo "[cell] done in $(( $(date +%s) - START ))s"
