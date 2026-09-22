#!/bin/bash
#SBATCH --job-name=cobit_dlrepro
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=logs/arch/%x_%j.log
#
# FAST REPRODUCTION of the epoch-boundary deadlock.
#
# The hang is deterministic, not flaky: every arm (control, l2r, r2l, random)
# and the token run have stopped at the SAME collective index -- ranks 1-3 in
# WorkNCCL(SeqNum=2516971, ALLREDUCE, NumelIn=1) while rank 0 is three
# collectives ahead in a 1,572,864-element gradient bucket. Same numbers,
# different jobs, different arms, different nodes.
#
# Every job completes exactly FIVE epochs and hangs entering the sixth, wherever
# it resumed (epoch 0->5, 5->10, 15->20). Five epochs is 114,290 steps, and
# 2,516,971 / 114,290 = 22.0 collectives per step, which is DDP's default 25 MB
# bucketing of 133.76M parameters. Three readings of that fit the same data:
# 5 epochs, ~2.5M collectives, or ~20 h of process lifetime -- a resume always
# restarts on an epoch boundary and every arm runs at the same steps/s, so the
# logs cannot tell them apart.
#
# THIS JOB TELLS THEM APART. COBIT_STEPS_PER_EPOCH truncates the epoch to a
# batch count -- identical on every rank, so the ranks reach the boundary in
# lockstep exactly as they do naturally -- which buys eight epoch BOUNDARIES for
# ~9k collectives and ten minutes of runtime, three orders of magnitude below
# the other two candidates.
#
#   hangs     -> the boundary is the trigger, and the repro loop is now minutes
#   runs clean-> the boundary is innocent; the trigger is cumulative (a leak or
#                a counter), which is a different and far more specific bug.
#                Re-run with COBIT_STEPS_PER_EPOCH unset to confirm at full
#                epoch length before concluding anything.
#
# Either way the flight recorder and the SIGUSR1 stack dumps from
# deadlock_debug.sh should name the call site that one rank executes and the
# others do not.
#
#   sbatch scripts/hpc/arch/deadlock_repro.slurm
#
# Then:
#   python experiments/ordering/decode_nccl_trace.py logs/arch/nccl_trace_<jobid>_rank --tail 12
#   cat logs/arch/stack_<jobid>_rank*.txt
#
# Writes to runs/tasks/tinygsm/ord_dlrepro_none_s42 -- a throwaway directory,
# never one of the four real arms. No checkpoints are written (400 steps is
# below every save interval), so this costs no disk.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"; mkdir -p logs/arch

# Control arm, from scratch, tiny budget; the boundary is what is under test,
# not the model.
export ORD_MODE=none ORD_W=0 ORD_INIT=none ORD_SEED=42
export ORD_STEPS="${ORD_STEPS:-400}"
export ORD_TAG=dlrepro
export ORD_CKPT_EVERY=1000000
export COBIT_STEPS_PER_EPOCH="${COBIT_STEPS_PER_EPOCH:-50}"

source "$PROJECT_DIR/scripts/hpc/arch/deadlock_debug.sh"
# Fail fast: the dump is what we want, not a long wait. 6 min is far longer
# than any healthy boundary (validation is ~45 s).
export DDP_TIMEOUT_MIN="${DDP_TIMEOUT_MIN:-6}"
enable_nccl_flight_recorder "$PROJECT_DIR/logs/arch"
start_stall_watchdog "$PROJECT_DIR/logs/arch/${SLURM_JOB_NAME:-cobit_dlrepro}_${SLURM_JOB_ID:-local}.log" "${STALL_MIN:-3}"
trap stop_stall_watchdog EXIT

echo "[dlrepro] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
echo "[dlrepro] steps=$ORD_STEPS steps_per_epoch=$COBIT_STEPS_PER_EPOCH timeout=${DDP_TIMEOUT_MIN}min"
nvidia-smi -i 0 --query-gpu=name --format=csv,noheader
compgen -G "datasets/tinygsm/*capall*.meta.json" > /dev/null || { echo "[dlrepro] FATAL: no corpus" >&2; exit 1; }

set +e
srun "$COBIT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
     train.py --config configs/tasks/tinygsm_bits_ordering.py
rc=$?
set -e
echo "[dlrepro] rc=$rc"
if [ $rc -eq 0 ]; then
  echo "[dlrepro] NO HANG: $((ORD_STEPS / COBIT_STEPS_PER_EPOCH)) epoch boundaries passed cleanly."
  echo "[dlrepro] That is itself informative -- the boundary alone is not sufficient;"
  echo "[dlrepro] re-run with a larger COBIT_STEPS_PER_EPOCH or ORD_PROBE_EVERY unset."
else
  echo "[dlrepro] reproduced (or crashed). Next:"
  echo "  $COBIT_PYTHON experiments/ordering/decode_nccl_trace.py logs/arch/nccl_trace_${SLURM_JOB_ID}_rank --tail 12"
  echo "  cat logs/arch/stack_${SLURM_JOB_ID}_rank*.txt"
fi
echo "[dlrepro] finished $(date)"
exit 0
