#!/bin/bash
# Per-cluster parameters. Everything a launcher needs to know about WHERE it is
# runs through this file, so moving to a new machine edits one file rather than
# fourteen.
#
# Select with COBIT_SITE=csd3|isambard, or let it auto-detect from the hostname.
# Sourced by scripts/hpc/guidance/env.sh, so every existing launcher picks it up
# without being modified.
#
# NOTE ON #SBATCH DIRECTIVES. SLURM's precedence is command line > in-script
# #SBATCH > environment, so exporting SBATCH_ACCOUNT cannot override a
# hardcoded directive. scripts/site/submit.sh therefore passes these as command
# line flags, which do win. Submit through that wrapper, not bare sbatch.

COBIT_SITE="${COBIT_SITE:-}"
if [ -z "$COBIT_SITE" ]; then
  case "$(hostname -f 2>/dev/null || hostname)" in
    *hpc.cam.ac.uk|login-q*|gpu-q-*) COBIT_SITE=csd3 ;;
    *isambard*|*aip*|nid*)           COBIT_SITE=isambard ;;
    *)                               COBIT_SITE=csd3 ;;   # historical default
  esac
fi
export COBIT_SITE

case "$COBIT_SITE" in
  csd3)
    export COBIT_PARTITION="ampere"
    export COBIT_ACCOUNT="GIROLAMI-SL2-GPU"
    export COBIT_GPUS_PER_NODE=4
    export COBIT_CPUS_PER_TASK=32
    export COBIT_WALL_MAX="36:00:00"          # QOS cap
    export COBIT_GRES="gpu:4"
    export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
    export COBIT_PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
    export COBIT_ARCH="x86_64"
    export COBIT_GPU="A100-80GB"
    ;;
  isambard)
    # Isambard-AI: GH200 Grace-Hopper. aarch64 CPU, sm_90 GPU, 96 GB HBM3.
    # Fill PARTITION/ACCOUNT from your grant email -- they are not guessable,
    # and a wrong account is a submission that silently never schedules.
    export COBIT_PARTITION="${COBIT_PARTITION:-grace}"
    export COBIT_ACCOUNT="${COBIT_ACCOUNT:?set COBIT_ACCOUNT to the Isambard project code}"
    export COBIT_GPUS_PER_NODE="${COBIT_GPUS_PER_NODE:-4}"
    export COBIT_CPUS_PER_TASK="${COBIT_CPUS_PER_TASK:-72}"
    export COBIT_WALL_MAX="${COBIT_WALL_MAX:-24:00:00}"
    export COBIT_GRES="gpu:${COBIT_GPUS_PER_NODE}"
    export COBIT_PYTHON="${COBIT_PYTHON:?set COBIT_PYTHON to the aarch64 env built by build_env_isambard.sh}"
    export COBIT_PROJECT_DIR="${PROJECT_DIR:-$HOME/BitstreamDiffusion}"
    export COBIT_ARCH="aarch64"
    export COBIT_GPU="GH200-96GB"
    ;;
  *)
    echo "[site] unknown COBIT_SITE=$COBIT_SITE" >&2; return 1 2>/dev/null || exit 1 ;;
esac

echo "[site] $COBIT_SITE | $COBIT_GPU x$COBIT_GPUS_PER_NODE | $COBIT_ARCH | account=$COBIT_ACCOUNT"
