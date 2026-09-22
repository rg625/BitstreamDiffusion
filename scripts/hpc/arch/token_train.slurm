#!/bin/bash
#SBATCH --job-name=cobit_toktrain
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# TOKEN-SPACE CoBit: V-way softmax over the 49,153-token vocabulary per token
# position, replacing the per-bit binary head. This is the "predict logits
# instead of raw bits" experiment.
#
#   sbatch --export=ALL,TOK_LOSS=token_ce,TOK_BATCH=512,TOK_ACCUM=4 \
#          scripts/hpc/arch/token_train.slurm
#
# MEMORY, and why TOK_ACCUM exists. The logits tensor is [B,512,49153]: 26 GB in
# bf16 at global batch 512. Spread over 4 ranks that is 128 examples/GPU, which
# does NOT fit -- job 35223498 died asking for 12.00 GiB with 3.44 GiB free on an
# 80 GB A100. The first 500k V-way run therefore dropped to an effective batch of
# 128 (32/GPU) and scored 0.0255 against a binary control at the same 128 scoring
# 0.0493: a matched comparison, but made at a weak operating point, both far
# below production's 0.164.
#
# TOK_ACCUM restores the batch without more GPUs. TOK_BATCH stays the EFFECTIVE
# batch -- the examples behind one optimiser step, the number that has to match
# the binary control -- and the step is split into TOK_ACCUM forward passes of
# TOK_BATCH/(4 x TOK_ACCUM) each. TOK_BATCH=512 TOK_ACCUM=4 is 32 examples/GPU,
# exactly the per-GPU load the 128-batch run already proved fits, at 4x the
# effective batch. global_step stays an OPTIMISER-step count, so "500k steps"
# means the same thing here as in the binary arms.
#
# If it still OOMs: raise TOK_ACCUM (8 -> 16/GPU), never lower TOK_BATCH, or the
# comparison moves again.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch
export TOK_LOSS="${TOK_LOSS:-token_ce}" TOK_LR="${TOK_LR:-3e-5}" TOK_SEED="${TOK_SEED:-42}"
export TOK_STEPS="${TOK_STEPS:-50000}" TOK_TAG="${TOK_TAG:-fs}" TOK_BATCH="${TOK_BATCH:-512}"
export TOK_CKPT_EVERY="${TOK_CKPT_EVERY:-25000}"
export TOK_ACCUM="${TOK_ACCUM:-1}"

# Epoch-boundary deadlock instrumentation (see deadlock_debug.sh); the token run
# hit the same hang as the ordering arms. Free unless the run stalls.
source "$PROJECT_DIR/scripts/hpc/arch/deadlock_debug.sh"
export DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-20}"
enable_nccl_flight_recorder "$PROJECT_DIR/logs/arch"
start_stall_watchdog "$PROJECT_DIR/logs/arch/${SLURM_JOB_NAME:-cobit_toktrain}_${SLURM_JOB_ID:-local}.log" "${STALL_MIN:-15}"
trap stop_stall_watchdog EXIT
echo "[tok] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
echo "[tok] loss=$TOK_LOSS batch=$TOK_BATCH (accum=$TOK_ACCUM -> $((TOK_BATCH / (4 * TOK_ACCUM)))/GPU) lr=$TOK_LR steps=$TOK_STEPS seed=$TOK_SEED tag=$TOK_TAG"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader
compgen -G "datasets/tinygsm/*capall*.meta.json" >/dev/null || { echo "[tok] FATAL no corpus" >&2; exit 1; }
set +e
srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
     train.py --config configs/tasks/tinygsm_tokens.py
rc=$?; set -e
[ $rc -ne 0 ] && echo "[tok] rc=$rc -- check for OOM or [divergence-guard]"
echo "[tok] finished $(date)"; exit 0
