#!/bin/bash
#SBATCH --job-name=ge_eval
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Gamma-dependent Euler study (configs/gamma_euler_eval.py): runs manifest lines GE_FIRST..GE_LAST (1-based) one
# after another. Each line is a set of env assignments (GE_DATA GE_ARM GE_GRID GE_SAMPLER GE_S|GE_GAMMA GE_NFES
# EVAL_SEED [DGC_NUM_SAMPLES]). A line whose output directory already holds a finished run (.ge_done) is skipped.
#
#   GE_MANIFEST  path   GE_FIRST  GE_LAST
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
: "${GE_MANIFEST:?}" "${GE_FIRST:?}" "${GE_LAST:?}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
CONFIG=configs/gamma_euler_eval.py
mkdir -p logs/arch
echo "[ge] commit $(git rev-parse --short HEAD) | $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader)"

fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "[ge] ==== $line"
  out=$(env $line "$COBIT_PYTHON" -c "import sys; sys.path.insert(0, '.')
from configs.gamma_euler_eval import get_config
c = get_config(); import os
assert os.path.isfile(c.evaluation.grid_file), c.evaluation.grid_file
assert os.path.isfile(c.evaluation.checkpoint_path), c.evaluation.checkpoint_path
print(c.evaluation.out_dir)" | tail -1) || { echo "[ge] bad line"; fail=1; continue; }
  if [ -f "$out/.ge_done" ]; then echo "[ge] done already: $out"; continue; fi
  t0=$(date +%s)
  if env $line "$COBIT_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       -m evaluation.run_eval --config "$CONFIG" --metrics external_ppl \
     && env $line "$COBIT_PYTHON" -m evaluation.compute_entropy_from_caches --config "$CONFIG"; then
    date > "$out/.ge_done"
    echo "[ge] ok $(( $(date +%s) - t0 ))s: $out"
  else
    echo "[ge] FAILED: $out"; fail=1
  fi
done < <(sed -n "${GE_FIRST},${GE_LAST}p" "$GE_MANIFEST")
echo "[ge] finished $(date)"
exit $fail
