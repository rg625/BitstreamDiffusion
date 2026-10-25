#!/bin/bash
# Rebuild the Python environment on Isambard-AI (GH200: aarch64 + sm_90).
#
# THE CONDA ENV HERE CANNOT BE COPIED. sedd310 is x86_64 with torch
# 2.8.0+cu128 built for sm_80; every compiled wheel in it is the wrong
# architecture. Copying it produces an env that imports and then fails at the
# first CUDA call, which is the slowest possible way to find out.
#
#   bash scripts/site/build_env_isambard.sh [env_name]
#
# Runs on a login node (it needs the network; compute nodes have none). The
# login node has no GPU driver, so this script can only verify the BUILD half;
# the device half is gate 1 proper:
#
#   bash scripts/site/submit.sh scripts/site/verify_gpu.slurm
#
# Where things go. Home is capped at 101 GB and already holds the repo, so
# neither conda nor the pip cache belongs there:
#   conda + env   $COBIT_CONDA_ROOT   (default $PROJECTDIR/$USER/miniforge3)
#   pip cache     $SCRATCHDIR/pip-cache
set -euo pipefail
ENV_NAME="${1:-cobit-gh}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ROOT="${COBIT_CONDA_ROOT:-${PROJECTDIR:?PROJECTDIR unset -- set COBIT_CONDA_ROOT}/$USER/miniforge3}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCHDIR:-$CONDA_ROOT/..}/pip-cache}"

# torch 2.8.0 is the version behind every result. download.pytorch.org has no
# aarch64 2.8.0+cu128 wheel -- only x86_64 and Windows -- so the aarch64 build of
# the SAME torch is +cu129. An unpinned install from the cu128 index would
# silently give the newest torch (2.11 at the time of writing): a different
# compiler stack and DDPOptimizer under every torch.compile'd run.
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
TORCH_PINS="torch==2.8.0 torchvision==0.23.0"

echo "[build] target: aarch64 + CUDA sm_90 (GH200)"
uname -m | grep -q aarch64 || { echo "[build] this host is $(uname -m), not aarch64" >&2; exit 1; }

# 1. Interpreter. 3.10 matches what every result in this repo was produced
#    with; the 3.9/3.10 divergence question is closed but the pinning habit is
#    cheap and keeps one variable out of any future disagreement.
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  echo "[build] no conda at $CONDA_ROOT. Install Miniforge for aarch64:"
  echo "  curl -L -o \$SCRATCHDIR/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  echo "  bash \$SCRATCHDIR/mf.sh -b -p $CONDA_ROOT"
  exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
# `conda create -y` over an existing env deletes it first. Refuse instead: a
# working env is worth more than the minute it saves.
if [ -d "$CONDA_ROOT/envs/$ENV_NAME" ]; then
  echo "[build] env $ENV_NAME already exists at $CONDA_ROOT/envs/$ENV_NAME." >&2
  echo "[build] Verify it:  $CONDA_ROOT/envs/$ENV_NAME/bin/python $REPO/scripts/site/verify_env.py" >&2
  echo "[build] Or remove it first:  conda env remove -n $ENV_NAME" >&2
  exit 1
fi
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

# 2. PyTorch, from the versioned index, NOT PyPI -- plain `pip install torch`
#    on aarch64 gives a CPU-only build that reports cuda False after everything
#    else is installed.
# shellcheck disable=SC2086
pip install --index-url "$TORCH_INDEX" $TORCH_PINS

# 3. The rest, with requirements.txt's tested pins (transformers<4.50 among
#    them). The constraint file stops any dependency from moving torch off the
#    pinned build.
CONSTRAINTS="$(mktemp)"; trap 'rm -f "$CONSTRAINTS"' EXIT
# shellcheck disable=SC2086
printf '%s\n' $TORCH_PINS > "$CONSTRAINTS"
pip install -c "$CONSTRAINTS" -r "$REPO/requirements.txt" pytest

PY="$(python -c 'import sys;print(sys.executable)')"
echo
echo "[build] env at: $PY"
echo "[build] export COBIT_PYTHON=$PY"
echo
echo "[build] ---- build-time verification ----"
"$PY" "$REPO/scripts/site/verify_env.py"
echo
echo "[build] next, gate 1 on a GPU node (a few minutes):"
echo "  COBIT_SITE=isambard COBIT_ACCOUNT=<project> COBIT_PYTHON=$PY bash scripts/site/submit.sh scripts/site/verify_gpu.slurm"
echo "[build] and gate 2, the cheapest port check there is:"
echo "  cd $REPO && PYTHONPATH=\$PWD $PY -m pytest tests/ -q"
