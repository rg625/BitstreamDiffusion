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
# Then set COBIT_PYTHON to the printed path and run the verification at the
# bottom BEFORE submitting anything that costs grant hours.
set -euo pipefail
ENV_NAME="${1:-cobit-gh}"
PREFIX="${CONDA_PREFIX_ROOT:-$HOME/miniforge3}"

echo "[build] target: aarch64 + CUDA sm_90 (GH200)"
uname -m | grep -q aarch64 || echo "[build] WARNING: this host is $(uname -m), not aarch64 -- are you on a login node?"

# 1. Interpreter. 3.10 matches what every result in this repo was produced
#    with; the 3.9/3.10 divergence question is closed but the pinning habit is
#    cheap and keeps one variable out of any future disagreement.
if ! command -v conda >/dev/null 2>&1; then
  echo "[build] no conda found. Install Miniforge for aarch64:"
  echo "  curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
  echo "  bash mf.sh -b -p $PREFIX && source $PREFIX/etc/profile.d/conda.sh"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

# 2. PyTorch. aarch64 + CUDA wheels come from the versioned index, NOT PyPI --
#    plain `pip install torch` on aarch64 gives a CPU-only build that will
#    report cuda False after everything else is installed.
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 3. The rest. No pinned CUDA-compiled packages here, so these are portable.
pip install ml_collections transformers datasets numpy scipy matplotlib \
            tqdm tensorboard pytest

echo
echo "[build] env at: $(python -c 'import sys;print(sys.executable)')"
echo "[build] export COBIT_PYTHON=$(python -c 'import sys;print(sys.executable)')"
echo
echo "[build] ---- verification (all four must pass) ----"
python - <<'PYEOF'
import sys
ok = True
import torch
print(f"torch {torch.__version__}  arch={torch.__config__.show().splitlines()[0][:40]}")

# (1) CUDA present at all. A CPU-only aarch64 wheel is the classic silent failure.
if not torch.cuda.is_available():
    print("FAIL  torch.cuda.is_available() is False -- CPU-only wheel"); ok = False
else:
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"ok    cuda {name} capability {cap}")
    # (2) Hopper. sm_90 is what GH200 reports; anything else means a different node type.
    if cap[0] != 9:
        print(f"WARN  capability {cap} is not sm_90 -- not a GH200?")
    # (3) bf16. Every run in this repo trains in bf16 autocast.
    if not torch.cuda.is_bf16_supported():
        print("FAIL  bf16 unsupported"); ok = False
    else:
        print("ok    bf16 supported")
    # (4) Memory. The V-way token model needs the headroom; 96 GB removes the
    #     gradient-accumulation workaround that four 80 GB A100s forced.
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"ok    {gb:.0f} GB per GPU"
          + ("  -> TOK_ACCUM can likely drop to 2 or 1; re-smoke before trusting it" if gb > 85 else ""))

# (5) Triton, used by the token_sm loss kernel. There is a documented eager
#     fallback, so this is a warning rather than a failure.
try:
    import triton  # noqa: F401
    print("ok    triton present (token_sm fast path available)")
except Exception as e:
    print(f"WARN  no triton ({e}); token_sm falls back to the chunked kernel")

sys.exit(0 if ok else 1)
PYEOF
echo
echo "[build] now run the full suite -- it is the cheapest port check there is:"
echo "  cd \$PROJECT_DIR && PYTHONPATH=\$PWD $(python -c 'import sys;print(sys.executable)') -m pytest tests/ -q"
