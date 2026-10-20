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

# Per-cluster parameters live in one place. On CSD3 this reproduces exactly what
# was hardcoded here before; on Isambard it supplies the aarch64 interpreter and
# that site's partition/account. Guarded so an older checkout without the file
# still works.
if [ -f "$PROJECT_DIR/scripts/site/site.sh" ]; then
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/scripts/site/site.sh"
  export PROJECT_DIR="${COBIT_PROJECT_DIR:-$PROJECT_DIR}"
fi
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd/bin/python}"

cd "$PROJECT_DIR"

if [ "${COBIT_SITE:-csd3}" = "csd3" ] && [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
  module purge >/dev/null 2>&1 || true
  module load rhel8/default-amp >/dev/null 2>&1 || true
fi
if [ "${COBIT_SITE:-}" = "isambard" ] && [ -n "${COBIT_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $COBIT_MODULES >/dev/null 2>&1 || true
fi

# Everything offline: compute nodes have no outbound network, and a silent
# attempt to reach the Hub stalls the job until it times out.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/hf_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# sbatch --export=ALL carries the LOGIN node's TMPDIR (/local/user/<uid> on
# Isambard) into the job. torch creates a temp dir at import, and all ranks died
# there before step 0 (jobs 6840804, 6841368). In 6841368 the directory passed an
# exists-and-writable check in the batch script and was still missing inside the
# srun step, so inside a job it is not trusted at all: node-local /tmp (TmpFS).
if [ -n "${SLURM_JOB_ID:-}" ] || [ -z "${TMPDIR:-}" ] || [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
  export TMPDIR=/tmp
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# srun has not inherited sbatch's --cpus-per-task since SLURM 22.05 (Isambard
# runs 24.11). Without this, `srun torchrun` gets ONE core for four ranks and
# all their dataloader workers.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  export SRUN_CPUS_PER_TASK="${SRUN_CPUS_PER_TASK:-$SLURM_CPUS_PER_TASK}"
fi

# Determinism knobs. cuBLAS workspace config is required for reproducible
# reductions; without it, repeated runs of the same seed can differ slightly and
# paired comparisons lose their meaning.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "[env] host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "[env] python=$COBIT_PYTHON"
"$COBIT_PYTHON" -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
