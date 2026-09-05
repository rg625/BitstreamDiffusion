#!/bin/bash
#SBATCH --job-name=cobit_probe
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Throughput probe for the representation branch.
#
# Every GPU-hour figure quoted for the representation and temporal-ordering
# branches is currently a GUESS: production trained 500k steps on 4xA100 but
# steps/sec was never recorded. This job measures it, for BOTH loss arms, so the
# pilot can be costed instead of estimated.
#
# It trains 200 steps and stops (optim.total_steps=200). It is not a pilot and
# its checkpoints are disposable.
#
# PREREQUISITE -- run on the LOGIN NODE first, because compute nodes are OFFLINE
# and the capped cache is built by streaming from HuggingFace:
#
#   export PYTHONPATH=$PWD HF_HOME=$PWD/hf_cache
#   PROBE_CAP=20000 $COBIT_PYTHON scripts/tasks/prebuild_task.py \
#       --config configs/tasks/tinygsm_bits_probe.py
#
# Submit (one arm per job):
#   sbatch --export=ALL,PROBE_LOSS=binary_sm scripts/hpc/arch/throughput_probe.slurm
#   sbatch --export=ALL,PROBE_LOSS=binary_ce scripts/hpc/arch/throughput_probe.slurm
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"
mkdir -p logs/arch

export PROBE_LOSS="${PROBE_LOSS:-binary_sm}"
export PROBE_CAP="${PROBE_CAP:-20000}"
export PROBE_STEPS="${PROBE_STEPS:-200}"
CFG="configs/tasks/tinygsm_bits_probe.py"

echo "[probe] loss=$PROBE_LOSS cap=$PROBE_CAP steps=$PROBE_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Refuse rather than silently build a cache on a compute node with no network.
if ! compgen -G "datasets/tinygsm/*cap${PROBE_CAP}*.meta.json" > /dev/null; then
  echo "[probe] FATAL: no capped cache for PROBE_CAP=$PROBE_CAP." >&2
  echo "[probe] Build it on the LOGIN node first (see header)." >&2
  exit 1
fi

# Start clean: a resumed run would report steady-state throughput for a
# different step count and silently invalidate the measurement.
rm -rf "runs/tasks/tinygsm/probe_${PROBE_LOSS}"

START=$(date +%s)
# `srun torchrun` fails with "execve(): torchrun: No such file or directory":
# the conda env is not on PATH inside the srun step (env.sh exports
# COBIT_PYTHON but never touches PATH, and the production launcher relies on a
# `conda activate` we do not do here). Invoke the module through the interpreter
# we already resolved, which needs no PATH at all.
srun "$COBIT_PYTHON" -m torch.distributed.run \
     --standalone --nproc_per_node=4 train.py --config "$CFG"
ELAPSED=$(( $(date +%s) - START ))

echo "[probe] loss=$PROBE_LOSS wall=${ELAPSED}s for ${PROBE_STEPS} steps on 4 GPUs"
"$COBIT_PYTHON" - <<PY
steps=int("${PROBE_STEPS}"); wall=${ELAPSED}; gpus=4
sps=steps/max(wall,1)
gpuh_per_1k = gpus*1000/max(sps,1e-9)/3600
print(f"[probe] {sps:.3f} steps/s   {gpuh_per_1k:.2f} GPU-h per 1k steps")
for k in (25,50,100,250,500):
    print(f"[probe]   {k}k-step arm -> {gpuh_per_1k*k:6.1f} GPU-h")
print("[probe] NOTE: includes startup/compile; subtract a fixed offset by")
print("[probe] re-running with PROBE_STEPS=400 and differencing.")
PY
