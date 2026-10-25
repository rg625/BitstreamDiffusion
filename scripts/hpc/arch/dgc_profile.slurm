#!/bin/bash
#SBATCH --job-name=dgc_profile
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Forward-only calibration of one frozen checkpoint (scripts/dgc/profile_forward_increments.py).
#   PR_CONFIG  eval config    PR_DATA  clean held-out bit sequences (.pt)    PR_OUT  output prefix
#   [PR_N=2048] [PR_K=64] [PR_NESTED_K=128] [PR_CLOSURE=null] [PR_BATCH=64] [PR_LO] [PR_HI] [PR_EXTRA]
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch "$(dirname "${PR_OUT:?}")"
echo "[profile] commit $(git rev-parse --short HEAD) | $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"
"$COBIT_PYTHON" scripts/dgc/profile_forward_increments.py --config "${PR_CONFIG:?}" --data "${PR_DATA:?}" \
  --out "$PR_OUT" --n_seq "${PR_N:-2048}" --K "${PR_K:-64}" --nested_K "${PR_NESTED_K:-128}" \
  --closure "${PR_CLOSURE:-null}" --batch "${PR_BATCH:-64}" ${PR_LO:+--sigma_lo $PR_LO} ${PR_HI:+--sigma_hi $PR_HI} \
  ${PR_EXTRA:-}
echo "[profile] finished $(date)"
