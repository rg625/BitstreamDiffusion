#!/bin/bash
#SBATCH --job-name=cobit_ordeval
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Evaluate ONE trained ordering arm on the full 1319-problem TinyGSM/GSM8K set.
#
#   sbatch --export=ALL,EV_RUN=ord_lr3e5_l2r_w0.25_s42,EV_MODE=l2r,EV_W=0.25 \
#          scripts/hpc/arch/ordering_eval.slurm
#
# TWO decode settings per arm, because they answer different questions:
#   (a) MATCHED   -- decoded with the ordering it was TRAINED with. This is the
#       end-to-end arm: "a model trained and decoded with ordering" vs "a model
#       trained and decoded uniformly". The brief's consistency requirement.
#   (b) UNIFORM   -- the same weights decoded with w=0. Separates "training with
#       ordering improved the model" from "the ordering decode itself helps".
# Without (b) a win could not be attributed to training rather than decoding.
#
# 3 seeds each. Identical sampler, schedule, steps, EMA and problem set
# throughout, so every comparison is paired.
#
# SCHEDULE = KARRAS, NOT ENTROPIC, and this matters. The entropic schedule is
# fitted per run from that run's OWN entropy tables, and gsm8k_eval takes them
# from the checkpoint's directory. Each fine-tuned arm therefore refitted its
# own sampling schedule: measured max|cdf - production| was 0.024 (l2r), 0.175
# (control), 0.200 (random), 0.545 (r2l). Evaluating that way compares arms
# that differ in BOTH training ordering and sampling schedule, which is what
# produced an apparently catastrophic 0.0000 for the control. Karras is
# analytic and identical for every arm by construction.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch results/ordering/train_eval

EV_RUN="${EV_RUN:?set EV_RUN=<run dir name under runs/tasks/tinygsm>}"
EV_MODE="${EV_MODE:?set EV_MODE=none|l2r|r2l|random}"
EV_W="${EV_W:?set EV_W}"
EV_CKPT="${EV_CKPT:-last.pt}"
EV_STEPS="${EV_STEPS:-256}"
EV_LIMIT="${EV_LIMIT:-1319}"
CK="runs/tasks/tinygsm/${EV_RUN}/checkpoints/${EV_CKPT}"
[ -f "$CK" ] || { echo "[ev] FATAL: missing $CK" >&2; exit 1; }

echo "[ev] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
echo "[ev] run=$EV_RUN ckpt=$EV_CKPT train_mode=$EV_MODE w=$EV_W"

run_one () {   # $1=order_mode $2=order_w $3=seed $4=label
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
    --config configs/tasks/tinygsm_bits_cfg.py --checkpoint "$CK" --ema 1 \
    --schedule karras --sampler_kind ordered \
    --order_w "$2" --order_mode "$1" --order_seed $((1234 + $3)) \
    --steps "$EV_STEPS" --limit "$EV_LIMIT" --seed "$3" \
    --out_dir results/ordering/train_eval \
    --regime ordering_train --tag "${EV_RUN}__ck${EV_CKPT//[^0-9]/}__${4}__s${3}" \
    || echo "[ev] FAILED ${EV_RUN} ${4} seed $3"
}

for s in 0 1 2; do
  run_one "$EV_MODE" "$EV_W" "$s" "matched"      # (a)
  run_one none 0.0 "$s" "uniform"                # (b)
done
echo "[ev] finished $(date)"
