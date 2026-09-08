#!/bin/bash
#SBATCH --job-name=cobit_ordconf
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# TEMPORAL ORDERING -- STAGE 2 CONFIRMATION
#
# The screen (n=250, 1 seed) found: w=0.1 no effect (+0.0000, CI +-0.020);
# w >= 0.25 collapses to 0.0000. So there is no w that helps, and the point of
# this stage is to put a TIGHT interval on the null at the only strength the
# model survives, rather than to hunt for a winner.
#
# Arms: control (w=0), l2r w=0.1, random w=0.1. 1319 problems x 3 seeds.
# Larger w is deliberately NOT confirmed -- it is already known to collapse and
# spending on it would buy nothing.
# R->L is NOT run: the brief says only if the first results justify it, and a
# flat null plus a collapse does not.
#
#   sbatch scripts/hpc/arch/ordering_confirm.slurm
#
# ~1319 problems at ~1.35 samples/s => ~16 min/cell, 9 cells => ~2.4 GPU-h.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch results/ordering/confirm results/ordering/screen

echo "[ord] commit $(git rev-parse --short HEAD)"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader

# First: the two random screen cells that died on a generator/device mismatch.
"$COBIT_PYTHON" - <<'PY' > /tmp/ord_rand.txt
import sys; sys.path.insert(0, "experiments/ordering")
import grid, shlex
for c in grid.screen():
    if c.order_mode == "random":
        print(shlex.join(c.cli("results/ordering/screen")))
PY
while IFS= read -r a; do
  echo "[ord] rerun screen random: $a"
  eval "\"$COBIT_PYTHON\" evaluation/tasks/gsm8k_eval.py $a" || echo "[ord] FAILED"
done < /tmp/ord_rand.txt

# Then the confirmation grid.
"$COBIT_PYTHON" - <<'PY' > /tmp/ord_conf.txt
import sys; sys.path.insert(0, "experiments/ordering")
import grid, shlex
for c in grid.confirm([(0.1, "l2r"), (0.1, "random")], limit=1319, steps=256,
                      seeds=(0, 1, 2)):
    print(shlex.join(c.cli("results/ordering/confirm")))
PY
n=$(wc -l < /tmp/ord_conf.txt); i=0
echo "[ord] $n confirmation cells"
while IFS= read -r a; do
  i=$((i+1)); echo "[ord] ---- $i/$n : $a"
  eval "\"$COBIT_PYTHON\" evaluation/tasks/gsm8k_eval.py $a" || echo "[ord] cell $i FAILED"
done < /tmp/ord_conf.txt
echo "[ord] finished at $(date)"
