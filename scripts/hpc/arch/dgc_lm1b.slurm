#!/bin/bash
#SBATCH --job-name=dgc_lm1b
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# LM1B DGC sweep: one grid x seed x list of cells (configs/lm1b/continuous/eval/dgc_sweep.py), GenPPL with
# GPT-2-large, then the token-unigram entropy pass over the cached texts. Submit through scripts/dgc/tasks_ab.sh.
#
#   SCHEDULE cobit|dgc   [EVAL_SEED=42] [CELLS="nfe:gamma/..."] [DGC_NUM_SAMPLES] (smoke only)
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
: "${SCHEDULE:?}"
export EVAL_SEED="${EVAL_SEED:-42}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
CONFIG=configs/lm1b/continuous/eval/dgc_sweep.py
mkdir -p logs/arch

echo "[lm1b] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader
echo "[lm1b] SCHEDULE=$SCHEDULE seed=$EVAL_SEED CELLS=${CELLS:-default} n=${DGC_NUM_SAMPLES:-1024}"

# The table must exist: a missing one fails the entropic grid, but the churn window
# would fall back to deterministic without an error.
"$COBIT_PYTHON" - <<'PYEOF'
import os, sys
sys.path.insert(0, ".")
from configs.lm1b.continuous.eval.dgc_sweep import get_config
cfg = get_config()
d = cfg.evaluation.entropy_run_dir
missing = [f for f in ("entropy_pdf.pt", "entropy_cdf.pt", "entropy_sigmas.pt") if not os.path.isfile(f"{d}/{f}")]
if missing or not os.path.isfile(cfg.evaluation.checkpoint_path):
    sys.exit(f"[lm1b] FATAL: table {d} missing {missing} or checkpoint {cfg.evaluation.checkpoint_path} absent")
print("[lm1b] table", d, "| ckpt", cfg.evaluation.checkpoint_path, "| out", cfg.evaluation.out_dir)
PYEOF

t0=$(date +%s)
"$COBIT_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  -m evaluation.run_eval --config "$CONFIG" --metrics external_ppl
echo "[lm1b] generation+GenPPL seconds: $(( $(date +%s) - t0 ))"
"$COBIT_PYTHON" -m evaluation.compute_entropy_from_caches --config "$CONFIG"
echo "[lm1b] finished $(date)"
