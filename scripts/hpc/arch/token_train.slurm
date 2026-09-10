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
#   sbatch --export=ALL,TOK_LOSS=token_ce,TOK_BATCH=512 scripts/hpc/arch/token_train.slurm
#
# MEMORY: the logits tensor is [B,512,49153] -- 26 GB in bf16 at global batch
# 512 (6.4 GB per GPU across 4 ranks), before activations. The loss is chunked
# (token_sm_chunk_size) rather than materialised at once. If this OOMs, drop
# TOK_BATCH and run the binary control at the SAME batch so the comparison
# stays matched.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch
export TOK_LOSS="${TOK_LOSS:-token_ce}" TOK_LR="${TOK_LR:-3e-5}" TOK_SEED="${TOK_SEED:-42}"
export TOK_STEPS="${TOK_STEPS:-50000}" TOK_TAG="${TOK_TAG:-fs}" TOK_BATCH="${TOK_BATCH:-512}"
export TOK_CKPT_EVERY="${TOK_CKPT_EVERY:-25000}"
echo "[tok] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
echo "[tok] loss=$TOK_LOSS batch=$TOK_BATCH lr=$TOK_LR steps=$TOK_STEPS seed=$TOK_SEED"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader
compgen -G "datasets/tinygsm/*capall*.meta.json" >/dev/null || { echo "[tok] FATAL no corpus" >&2; exit 1; }
set +e
srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
     train.py --config configs/tasks/tinygsm_tokens.py
rc=$?; set -e
[ $rc -ne 0 ] && echo "[tok] rc=$rc -- check for OOM or [divergence-guard]"
echo "[tok] finished $(date)"; exit 0
