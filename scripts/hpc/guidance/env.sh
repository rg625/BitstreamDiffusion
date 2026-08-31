#!/bin/bash
# Shared CSD3 (Wilkes3 / ampere) environment for the guidance experiments.
# Sourced by every job script so a fix lands in one place.
#
# NOTE: this is deliberately NOT the environment used by the earlier
# scripts/tasks/*.slurm files. Those point at /home/gb511/miniconda3 and
# /rds/project/rds-LlrDsbHU5UM/..., neither of which is readable by this
# account; a job copied from them fails at `conda activate`.
set -euo pipefail

export PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd/bin/python}"

cd "$PROJECT_DIR"

if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
  module purge >/dev/null 2>&1 || true
  module load rhel8/default-amp >/dev/null 2>&1 || true
fi

# Everything offline: compute nodes have no outbound network, and a silent
# attempt to reach the Hub stalls the job until it times out.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/hf_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Determinism knobs. cuBLAS workspace config is required for reproducible
# reductions; without it, repeated runs of the same seed can differ slightly and
# paired comparisons lose their meaning.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "[env] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "[env] python=$COBIT_PYTHON"
"$COBIT_PYTHON" -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
