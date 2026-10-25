#!/bin/bash
# Submit the token-space V-way run at effective batch 512 as a chain of
# afterany-linked SLURM jobs (the QOS wall cap is well below what 500k steps
# need, so the run has to be chained and resumed from last.pt).
#
#   bash scripts/hpc/arch/token_chain.sh [n_links]
#
# SMOKE FIRST. 32 examples/GPU is the per-GPU load the 128-batch run already
# proved fits, but accumulation itself has never run in this repo, so spend one
# hour before spending a week:
#
#   sbatch --time=01:00:00 --export=ALL,TOK_LOSS=token_ce,TOK_BATCH=512,\
# TOK_ACCUM=4,TOK_LR=1e-4,TOK_STEPS=300,TOK_TAG=smoke512,TOK_CKPT_EVERY=1000000 \
#     scripts/hpc/arch/token_train.slurm
#
# and check the log for:
#   [accum] effective batch 512 = 32/GPU x 4 GPUs x 4 accumulation steps
#   no CUDA OOM, and a loss that falls rather than sitting flat.
#
# THE ARM THIS IS MATCHED TO is the ordering control, ord_fs500k_none_s42:
# binary bits, from scratch, 500k steps, batch 512, lr 1e-4, seed 42. Same
# budget, same batch, same LR, same seed -- the only difference is the V-way
# head. Do NOT compare it to production's 0.164, which was trained at lr 3e-4,
# a rate that diverges 8 times out of 8 in this environment.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
cd "$PROJECT_DIR"

LINKS="${1:-9}"
# The QOS wall cap is 36 h and token_train.slurm's own default is 8 h, so the
# link length is set here. The smoke run measured ~0.5 optimiser steps/s, i.e.
# ~275 h for 500k steps: nine 36 h links with margin for queue gaps and restarts.
WALL="${WALL:-36:00:00}"

# A disk-full event once killed four runs mid-checkpoint. This run writes
# 5 archival checkpoints at 3.4 GB plus last.pt: ~21 GB.
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "[chain] free disk: ${AVAIL_GB}G | links=$LINKS x $WALL"
if [ "$AVAIL_GB" -lt 40 ]; then
  echo "[chain] REFUSING: under 40G free, and this run needs ~21G of checkpoints." >&2
  exit 1
fi

EXPORTS="ALL,TOK_LOSS=token_ce,TOK_BATCH=512,TOK_ACCUM=4,TOK_LR=1e-4,TOK_SEED=42"
EXPORTS="$EXPORTS,TOK_STEPS=500000,TOK_TAG=b512,TOK_CKPT_EVERY=100000"

prev=""
for i in $(seq 1 "$LINKS"); do
  if [ -z "$prev" ]; then
    out=$(sbatch --parsable --time="$WALL" --export="$EXPORTS" scripts/hpc/arch/token_train.slurm)
  else
    out=$(sbatch --parsable --time="$WALL" --dependency=afterany:"$prev" \
                 --export="$EXPORTS" scripts/hpc/arch/token_train.slurm)
  fi
  rc=$?
  # A failed sbatch is "unknown", not "not submitted": rc=124 once meant the
  # reply was lost while the job had in fact queued, and two jobs then wrote to
  # the same run directory. Stop and check squeue by hand rather than retrying.
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "[chain] sbatch returned rc=$rc out='$out' -- STOPPING." >&2
    echo "[chain] Check 'squeue -u $USER' before resubmitting: the job may have queued anyway." >&2
    exit 1
  fi
  echo "[chain] link $i: job $out${prev:+ (afterany:$prev)}"
  prev="$out"
done

echo "[chain] submitted $LINKS links -> runs/tasks/tinygsm/tok_b512_token_ce_s42"
echo "[chain] progress: python -c \"import torch;print(torch.load('runs/tasks/tinygsm/tok_b512_token_ce_s42/checkpoints/last.pt',mmap=True,weights_only=False)['global_step'])\""
echo "[chain] NOT the progress bar -- it resets at every link."
