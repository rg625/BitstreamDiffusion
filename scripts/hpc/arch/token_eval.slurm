#!/bin/bash
#SBATCH --job-name=cobit_tokeval
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Evaluate the 500k LOGITS-REPLACEMENT pair on the full 1319-problem GSM8K set.
#   EV_WHICH=token  -> V-way softmax model (configs/tasks/tinygsm_tokens.py)
#   EV_WHICH=bits   -> matched binary_sm control, SAME batch 128, same lr/steps
# Karras schedule, DDIM, gamma=0, EMA, 256 steps, 3 seeds.
#
# EV_BATCH matters for the token arm: a sampling forward materialises
# [B,512,49153] logits -- 6 GB at B=64 in fp32 -- and the sampler holds several
# such tensors, so the default batch OOM'd on all three seeds. Sampling batch
# affects throughput only, never the sample distribution, so shrinking it for
# the token arm does not make the comparison unfair. The two arms are
# compared only to each other: both ran batch 128 / lr 1e-4, not production's
# 512 / 3e-4.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch results/tokens/eval
W="${EV_WHICH:?set EV_WHICH=token|bits}"
if [ "$W" = token ]; then
  CFG=configs/tasks/tinygsm_tokens.py
  CK=runs/tasks/tinygsm/tok_fs500k_token_ce_s42/checkpoints/step=000500000.pt
  export TOK_LOSS=token_ce TOK_BATCH=128
else
  CFG=configs/tasks/tinygsm_bits_cfg.py
  CK=runs/tasks/tinygsm/obj_binary_sm_fs500k_b128_s42/checkpoints/step=000500000.pt
fi
[ -f "$CK" ] || { echo "[ev] FATAL missing $CK" >&2; exit 1; }
echo "[ev] which=$W cfg=$CFG ckpt=$CK"
for s in ${EV_SEEDS:-0 1 2}; do
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
    --config "$CFG" --checkpoint "$CK" --ema 1 \
    --schedule karras --sampler_kind ddim --gamma 0.0 \
    --steps 256 --limit "${EV_LIMIT:-1319}" --seed "$s" \
    --batch_size "${EV_BATCH:-64}" \
    --out_dir results/tokens/eval --regime logits_500k --tag "${W}_500k_s${s}" \
    || echo "[ev] FAILED $W seed $s"
done
echo "[ev] finished $(date)"
