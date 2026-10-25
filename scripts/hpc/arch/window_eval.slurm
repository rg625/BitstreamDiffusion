#!/bin/bash
#SBATCH --job-name=cobit_winev
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# Evaluate ONE window-ordering checkpoint on the full 1319 GSM8K test set.
#
# TWO decoders, each at every NFE in EV_NFES, 3 seeds each:
#   own       --sampler_kind window, the arm's own (mode, k) and frozen noise
#             table; for the control that is k=inf on the F-quantile grid.
#   standard  --sampler_kind ordered, w=0, Karras: the uniform deterministic
#             Euler every earlier result and the production anchor use.
# Both spend exactly NFE network evaluations (NFE-1 Euler steps + final
# denoise), so "own vs own" and "standard vs standard" across arms, and
# "own vs standard" within an arm, are all compute-matched.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"

EV_RUN="${EV_RUN:?set EV_RUN=<run dir name under runs/tasks/tinygsm, or label with EV_CK_PATH>}"
EV_MODE="${EV_MODE:?set EV_MODE}"
EV_K="${EV_K:?set EV_K}"
EV_CKPT="${EV_CKPT:-last.pt}"
EV_NFES="${EV_NFES:-256}"
EV_DECODERS="${EV_DECODERS:-own standard}"
EV_SEEDS="${EV_SEEDS:-0 1 2}"
EV_LIMIT="${EV_LIMIT:-1319}"
EV_OUT_DIR="${EV_OUT_DIR:-results/ordering/window_eval_gh200}"
EV_REGIME="${EV_REGIME:-window__eval_gh200}"
TABLE="assets/noise_tables/tinygsm_window_F_v1.pt"
mkdir -p logs/arch "$EV_OUT_DIR"

CK="${EV_CK_PATH:-runs/tasks/tinygsm/${EV_RUN}/checkpoints/${EV_CKPT}}"
[ -f "$CK" ] || { echo "[wev] FATAL: missing $CK" >&2; exit 1; }
EV_STEP=$("$COBIT_PYTHON" - "$CK" <<'PYEOF'
import sys, torch
try:
    print(f"{int(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False).get('global_step', -1)):09d}")
except Exception:
    print("unknown")
PYEOF
)
[ "$EV_STEP" = "unknown" ] && { echo "[wev] FATAL: no global_step in $CK" >&2; exit 1; }
echo "[wev] commit $(git rev-parse --short HEAD) | run=$EV_RUN step=$EV_STEP mode=$EV_MODE k=$EV_K"
echo "[wev] nfes=$EV_NFES decoders=$EV_DECODERS seeds=$EV_SEEDS out=$EV_OUT_DIR"

cell () {   # $1=decoder $2=nfe $3=seed
  local tag="${EV_RUN}__ck${EV_STEP}__${1}__n${2}__s${3}"
  if compgen -G "$EV_OUT_DIR/*${tag}_*.json" > /dev/null; then
    echo "[wev] exists, skipping $tag"; return
  fi
  if [ "$1" = own ]; then
    set -- "$1" "$2" "$3" --sampler_kind window --order_mode "$EV_MODE" --order_k "$EV_K" \
           --noise_table "$TABLE" --order_seed $((1234 + $3))
  else
    set -- "$1" "$2" "$3" --sampler_kind ordered --schedule karras --order_mode none --order_w 0.0
  fi
  local nfe="$2" seed="$3"; shift 3
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
    --config configs/tasks/tinygsm_bits_cfg.py --checkpoint "$CK" --ema 1 \
    "$@" --steps "$nfe" --limit "$EV_LIMIT" --seed "$seed" \
    --out_dir "$EV_OUT_DIR" --regime "$EV_REGIME" --tag "$tag" \
    || echo "[wev] FAILED $tag"
}

for s in $EV_SEEDS; do
  for n in $EV_NFES; do
    for d in $EV_DECODERS; do cell "$d" "$n" "$s"; done
  done
done
echo "[wev] finished $(date)"
