#!/bin/bash
#SBATCH --job-name=cobit_obj50keval
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Definitive objective comparison, evaluated. binary_sm vs binary_ce, both
# trained FROM SCRATCH for 50k steps at lr 3e-5, differing in exactly one key.
# Karras schedule (analytic, identical for both arms), DDIM, gamma=0, EMA,
# 1319 problems, 3 seeds. Same settings under which production scores 0.164 --
# but per the recorded caveat, these arms are compared ONLY to each other,
# never to production's 3e-4 trajectory.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch results/objective/eval50k
ARM="${EV_ARM:?set EV_ARM=binary_sm|binary_ce}"
CK="runs/tasks/tinygsm/obj_${ARM}_fs50k_s42/checkpoints/${EV_CKPT:-step=000050000.pt}"
[ -f "$CK" ] || { echo "[ev] FATAL missing $CK" >&2; exit 1; }
echo "[ev] arm=$ARM ckpt=$CK python=$($COBIT_PYTHON -V 2>&1)"
for s in 0 1 2; do
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
    --config configs/tasks/tinygsm_bits_cfg.py --checkpoint "$CK" --ema 1 \
    --schedule karras --sampler_kind ddim --gamma 0.0 \
    --steps 256 --limit 1319 --seed "$s" \
    --out_dir results/objective/eval50k \
    --regime objective_fs50k --tag "${ARM}_50k_s${s}" || echo "[ev] FAILED $ARM seed $s"
done
echo "[ev] finished $(date)"
