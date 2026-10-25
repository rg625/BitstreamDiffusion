#!/bin/bash
#SBATCH --job-name=cobit_ordscreen
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/arch/%x_%j.log
#
# TEMPORAL ORDERING -- STAGE 1 SCREEN
# 9 cells (7 at 256 steps + 2 at 512), 250 problems, 1 seed, ~1.35 samples/s
# on one A100 => ~3 min/cell at 256 steps, ~6 min at 512. ~35 min, ~0.6 GPU-h.
#
#   sbatch scripts/hpc/arch/ordering_screen.slurm
#
# Control (order_w=0) and every intervention differ by ONE number. Sampling-time
# only: the healthy 500k production checkpoint, no training.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch results/ordering/screen

echo "[ord] commit $(git rev-parse --short HEAD)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader

"$COBIT_PYTHON" - <<'PY' > /tmp/ord_cells.txt
import sys; sys.path.insert(0, "experiments/ordering")
import grid, shlex
for c in grid.screen() + grid.screen_steps():
    print(shlex.join(c.cli("results/ordering/screen")))
PY
n=$(wc -l < /tmp/ord_cells.txt)
echo "[ord] $n cells"
i=0
while IFS= read -r args; do
  i=$((i+1))
  echo "[ord] ---- cell $i/$n : $args"
  eval "\"$COBIT_PYTHON\" evaluation/tasks/gsm8k_eval.py $args" || echo "[ord] cell $i FAILED"
done < /tmp/ord_cells.txt
echo "[ord] finished at $(date)"
