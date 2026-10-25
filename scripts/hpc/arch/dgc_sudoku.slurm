#!/bin/bash
#SBATCH --job-name=dgc_sudoku
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Sudoku DGC A/B: one difficulty, every (arm, gamma, seed) cell in one job so the
# small model loads once per cell and the job overhead is paid once.
# Submit through scripts/dgc/tasks_ab.sh. Cells with a result file are skipped.
#
#   SU_DIFF     easy | medium | hard
#   [SU_ARMS="cobit dgc"] (also dpdgc = DGC table + finite-NFE DP allocation; cobit_dgcchurn, dgc_stdchurn)
#   [SU_GAMMAS="0 0.4"] [SU_SEEDS="42 43 44"] [SU_LIMIT=2000] [SU_STEPS=180]
#   [SU_OUT_DIR=results/dgc/sudoku] [SU_SIGMA_DATA=0.5]  (0.5 = what the stored baselines used)
#
# Protocol of the stored baselines: raw weights (EMA scores 0 at 20k steps), 180 steps,
# entropic grid, full-band churn; gamma 0 => deterministic.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"

SU_DIFF="${SU_DIFF:?}"
export SUDOKU_DIFFICULTY="$SU_DIFF"   # the config reads it at load time (dataset + experiment paths)
SU_ARMS="${SU_ARMS:-cobit dgc}"; SU_GAMMAS="${SU_GAMMAS:-0 0.4}"; SU_SEEDS="${SU_SEEDS:-42 43 44}"
SU_STEPS="${SU_STEPS:-180}"
SU_LIMIT="${SU_LIMIT:-2000}"; SU_OUT_DIR="${SU_OUT_DIR:-results/dgc/sudoku}"; SU_SIGMA_DATA="${SU_SIGMA_DATA:-0.5}"
SUDOKU_ROOT="${SUDOKU_ROOT:-/scratch/u6ya/raricu.u6ya/sudoku_drive}"
RUN_DIR="$SUDOKU_ROOT/$SU_DIFF/cobit_raw_binary_bits"
CK="$RUN_DIR/checkpoints/step=000020000.pt"
mkdir -p logs/arch "$SU_OUT_DIR"
[ -f "$CK" ] || { echo "[su] FATAL: missing $CK" >&2; exit 1; }

echo "[su] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader

for arm in $SU_ARMS; do
  DGC_DIR="assets/entropy_tables/tasks/sudoku_${SU_DIFF}_dgc"
  unset COBIT_CHURN_TABLE
  SCHED=entropic
  case "$arm" in   # *_dgcchurn / *_stdchurn: steps from one table, churn density from the other
    cobit) TABLE_DIR="$RUN_DIR" ;;
    dgc)   TABLE_DIR="$DGC_DIR" ;;
    dpdgc) TABLE_DIR="$DGC_DIR"; SCHED=dp_dgc ;;
    cobit_dgcchurn) TABLE_DIR="$RUN_DIR"; export COBIT_CHURN_TABLE="$DGC_DIR" ;;
    dgc_stdchurn)   TABLE_DIR="$DGC_DIR"; export COBIT_CHURN_TABLE="$RUN_DIR" ;;
    *) echo "[su] FATAL: arm $arm" >&2; exit 1 ;;
  esac
  for f in entropy_pdf.pt entropy_cdf.pt entropy_sigmas.pt; do
    [ -f "$TABLE_DIR/$f" ] || { echo "[su] FATAL: $TABLE_DIR/$f missing -> grid/churn would be wrong" >&2; exit 1; }
  done
  for g in $SU_GAMMAS; do
    g=$("$COBIT_PYTHON" -c "print(float('$g'))")   # the result file name spells gamma as python's float repr
    if awk "BEGIN{exit !($g > 0)}"; then SAMPLER=(--sampler stochastic --gamma "$g")
    else SAMPLER=(--sampler deterministic --gamma 0.0); fi
    for seed in $SU_SEEDS; do
      tag="${arm}_seed${seed}_n${SU_LIMIT}"
      if ls "$SU_OUT_DIR"/sudoku_results_"${SU_DIFF}"_*_g"${g}"_s"${SU_STEPS}"_*_"$tag".json >/dev/null 2>&1; then
        echo "[su] skip $SU_DIFF g=$g $tag"; continue
      fi
      echo "[su] === $SU_DIFF arm=$arm gamma=$g seed=$seed $(date +%T)"
      t0=$(date +%s)
      "$COBIT_PYTHON" evaluation/tasks/sudoku_eval.py \
        --config configs/tasks/sudoku_bits.py --checkpoint "$CK" --difficulty "$SU_DIFF" \
        --ema 0 --schedule "$SCHED" --entropy_dir "$TABLE_DIR" "${SAMPLER[@]}" \
        --steps "$SU_STEPS" --seed "$seed" --limit "$SU_LIMIT" --sigma_data "$SU_SIGMA_DATA" \
        --out_dir "$SU_OUT_DIR" --tag "$tag"
      echo "[su] cell seconds: $(( $(date +%s) - t0 ))"
    done
  done
done
echo "[su] finished $(date)"
