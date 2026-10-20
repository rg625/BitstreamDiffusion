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
# Architecture first: it is the difference that actually matters, and Isambard's
# login nodes are plain "login40" with no domain, which the hostname patterns
# below used to send to the csd3 fallback.
if [ -z "$COBIT_SITE" ] && [ "$(uname -m)" = "aarch64" ]; then
  COBIT_SITE=isambard
fi
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
    export COBIT_CPUS_PER_GPU=8             # 32 per 4-GPU node, as the launchers ask
    export COBIT_WALL_MAX="36:00:00"          # QOS cap
    export COBIT_GRES="gpu:4"
    export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
    export COBIT_PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
    export COBIT_ARCH="x86_64"
    export COBIT_GPU="A100-80GB"
    ;;
  isambard)
    # Isambard-AI: GH200 Grace-Hopper. aarch64 CPU, sm_90 GPU, 96 GB HBM3.
    # ACCOUNT is not guessable, and a wrong one is a submission that silently
    # never schedules: `sacctmgr -nP show assoc user=$USER format=account`
    # lists the ones you have. The partition is the only one sinfo shows
    # (workq); workq_qos caps wall time at 1-00:00:00.
    export COBIT_PARTITION="${COBIT_PARTITION:-workq}"
    # Taken from SLURM only when it lists exactly ONE association -- that is
    # SLURM's answer, not a guess. Zero or several and it must be set by hand.
    if [ -z "${COBIT_ACCOUNT:-}" ]; then
      _accts=$(timeout 30 sacctmgr -nP show assoc user="$USER" format=account 2>/dev/null | sort -u)
      [ "$(printf '%s\n' "$_accts" | grep -c .)" = 1 ] && COBIT_ACCOUNT="$_accts"
    fi
    export COBIT_ACCOUNT="${COBIT_ACCOUNT:?set COBIT_ACCOUNT to the Isambard project code}"
    export COBIT_GPUS_PER_NODE="${COBIT_GPUS_PER_NODE:-4}"
    export COBIT_CPUS_PER_TASK="${COBIT_CPUS_PER_TASK:-72}"
    export COBIT_CPUS_PER_GPU="${COBIT_CPUS_PER_GPU:-72}"   # one Grace socket per GPU
    export COBIT_WALL_MAX="${COBIT_WALL_MAX:-24:00:00}"
    export COBIT_GRES="gpu:${COBIT_GPUS_PER_NODE}"
    # The env build_env_isambard.sh makes, if it exists; never the x86_64 one.
    _gh_py="${PROJECTDIR:-/nonexistent}/$USER/miniforge3/envs/cobit-gh/bin/python"
    [ -z "${COBIT_PYTHON:-}" ] && [ -x "$_gh_py" ] && COBIT_PYTHON="$_gh_py"
    export COBIT_PYTHON="${COBIT_PYTHON:?set COBIT_PYTHON to the aarch64 env built by build_env_isambard.sh}"
    export COBIT_PROJECT_DIR="${PROJECT_DIR:-$HOME/BitstreamDiffusion}"
    export COBIT_ARCH="aarch64"
    export COBIT_GPU="GH200-96GB"
    ;;
  *)
    echo "[site] unknown COBIT_SITE=$COBIT_SITE" >&2; return 1 2>/dev/null || exit 1 ;;
esac

echo "[site] $COBIT_SITE | $COBIT_GPU x$COBIT_GPUS_PER_NODE | $COBIT_ARCH | account=$COBIT_ACCOUNT"
