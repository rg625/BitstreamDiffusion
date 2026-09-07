#!/bin/bash
#SBATCH --job-name=cobit_proddiag
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# PRODUCTION-ERA DIAGNOSTIC
#
# Question: is the early binary_sm divergence caused by OUR post-June code
# changes, or is it intrinsic to the historical production recipe?
#
# Runs the production recipe at commit 036a2b5 ("Add classifier-free-guidance
# TinyGSM run (p_uncond=0.1)", 2026-06-18) -- the commit under which the
# production run was started -- from a git worktree at
#   /rds/user/rg625/hpc-work/prod_era_diag
# with datasets/ and hf_cache/ symlinked to the main checkout, so the corpus is
# byte-identical to the one our stability runs used.
#
#   sbatch scripts/hpc/arch/prod_era_diagnostic.slurm
#
# EXACTLY THREE PATCHES are applied to the historical tree, all numerics-neutral
# (`git -C /rds/user/rg625/hpc-work/prod_era_diag diff` shows the whole surface):
#   1. utils/callbacks/mauve.py  -- `import mauve` made lazy. Production's env
#      had the package; ours does not. The callback is DISABLED by config
#      (train.mauve.enabled=False) so it never executes either way.
#   2. diffusion/discrete/losses.py -- add `from __future__ import annotations`.
#      The file uses PEP 604 (`torch.Tensor | None`) in a signature, which cannot
#      be imported on Python 3.9. Affects annotation evaluation only.
#   3. trainers/trainer.py -- the divergence guard, read-only on the loss.
#      Required by the brief so a diverging run stops cheaply.
# NO stabilizing changes. No clamp, no LR change, no schedule change, no reseed.
#
# Config: configs/tasks/tinygsm_bits_cfg_diag.py loads the production config
# verbatim and adds only the guard plus a separate run dir. Verified to differ
# from the production config by those keys alone. seed=42 (production's value),
# p_uncond=0.1, clamp absent, total_steps left at its original 250000 -- the
# wall clock bounds the run instead.
#
# BUDGET: 2 h wall x 4 GPUs = 8 GPU-h max, ~13k steps at the measured ~2.0 it/s.
# That clears all three of our binary_sm break points (4,481 / 6,356 / 9,872)
# with margin. A diverging run aborts earlier and costs less.
#
# READING THE RESULT
#   diverges  -> our post-June code changes are exonerated; the cause is the
#                environment or the recipe is intrinsically marginal.
#   survives  -> something in our changes matters, despite fwd+loss+bwd being
#                bit-identical between 036a2b5 and HEAD in fp32.
#
# CAVEAT, and it is not small: production was run by a collaborator from
# /rds/project/rds-LlrDsbHU5UM/gb511/... which is NOT readable by us, on
# Python >= 3.10 (the PEP 604 line above predates the run and cannot import on
# our 3.9). So this reproduces the production COMMIT in OUR environment. It
# cannot reproduce their working tree, which may have carried uncommitted
# changes, nor their interpreter and its torch build.
set -euo pipefail

DIAG_DIR="${DIAG_DIR:-/rds/user/rg625/hpc-work/prod_era_diag}"
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"

if [ ! -d "$DIAG_DIR" ]; then
  echo "[diag] FATAL: worktree $DIAG_DIR missing." >&2
  exit 1
fi
cd "$DIAG_DIR"
export PYTHONPATH="$DIAG_DIR"
mkdir -p logs/arch

echo "[diag] worktree HEAD: $(git rev-parse --short HEAD) ($(git log -1 --pretty=%s))"
echo "[diag] patch surface vs that commit:"
git diff --stat | sed 's/^/[diag]   /'
echo "[diag] python: $($COBIT_PYTHON -V 2>&1)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader

if ! compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null; then
  echo "[diag] FATAL: full-corpus cache not visible through the symlink." >&2
  exit 1
fi

set +e
srun "$COBIT_PYTHON" -m torch.distributed.run \
     --standalone --nproc_per_node=4 \
     train.py --config configs/tasks/tinygsm_bits_cfg_diag.py
rc=$?
set -e
# A guard abort is a RESULT, not an infrastructure failure.
if [ $rc -ne 0 ]; then
  echo "[diag] exited rc=$rc"
  echo "[diag] if the log shows [divergence-guard], the HISTORICAL recipe DIVERGED."
fi
echo "[diag] finished at $(date)"
exit 0
