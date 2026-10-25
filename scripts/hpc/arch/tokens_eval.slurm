#!/bin/bash
#SBATCH --job-name=cobit_tokev
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# ONE cell of the token-vs-bits GSM8K evaluation: one checkpoint, one seed, the
# full 1319 problems. One cell per job so every (arm, step, seed) runs on its
# own GPU in parallel; submit through tokens_eval_gh200.sh, which fills in the
# site flags and never submits a cell twice.
#
#   EV_ARM=token|bits   token -> configs/tasks/tinygsm_tokens.py (V-way softmax, full vocab)
#                       bits  -> configs/tasks/tinygsm_bits_cfg.py
#   EV_CK_PATH          checkpoint
#   EV_RUN              label recorded in the tag (the run directory's name)
#   EV_SEED EV_BATCH EV_OUT_DIR EV_REGIME [EV_STEPS=256] [EV_LIMIT=1319]
#
# PROTOCOL, identical for both arms: Karras schedule, DDIM, gamma=0, EMA, 256
# steps. Karras because the entropic schedule is refitted from each run's own
# entropy tables and would confound representation with sampling schedule.
# FULL VOCABULARY: --codeword_topk is never passed, so the token arm's softmax
# runs over all 49,153 tokens at every step.
#
# EV_BATCH changes throughput and peak memory only, never the sampling
# distribution, so the token arm runs at the largest batch the probe measured
# to fit and the bits arm at its own.
#
# EV_PROBE="16:32:..." (colon-separated) switches the job to a MEMORY PROBE: one short run per
# batch size (--limit B, EV_PROBE_STEPS steps), each in its own process so an
# OOM kills only that size. Peak memory and samples/s land in the JSONs under
# EV_OUT_DIR.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"

EV_ARM="${EV_ARM:?set EV_ARM=token|bits}"
CK="${EV_CK_PATH:?set EV_CK_PATH}"
EV_RUN="${EV_RUN:?set EV_RUN}"
EV_OUT_DIR="${EV_OUT_DIR:?set EV_OUT_DIR}"
EV_REGIME="${EV_REGIME:?set EV_REGIME}"
EV_STEPS="${EV_STEPS:-256}"
EV_LIMIT="${EV_LIMIT:-1319}"
mkdir -p logs/arch "$EV_OUT_DIR"
[ -f "$CK" ] || { echo "[tev] FATAL: missing $CK" >&2; exit 1; }

case "$EV_ARM" in
  token) CFG=configs/tasks/tinygsm_tokens.py; export TOK_LOSS=token_ce ;;
  bits)  CFG=configs/tasks/tinygsm_bits_cfg.py ;;
  *) echo "[tev] FATAL: EV_ARM must be token|bits, got '$EV_ARM'" >&2; exit 1 ;;
esac

# The step comes from the checkpoint, never the file name (last.pt has none).
EV_STEP=$("$COBIT_PYTHON" - "$CK" <<'PYEOF'
import sys, torch
try:
    d = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
    print(f"{int(d.get('global_step', -1)):09d}")
except Exception:
    print("unknown")
PYEOF
)
[ "$EV_STEP" = "unknown" ] && { echo "[tev] FATAL: could not read global_step from $CK" >&2; exit 1; }

echo "[tev] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader
echo "[tev] arm=$EV_ARM run=$EV_RUN step=$EV_STEP cfg=$CFG ckpt=$CK"

run_eval () {  # $1=seed $2=batch $3=limit $4=steps $5=tag
  "$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
    --config "$CFG" --checkpoint "$CK" --ema 1 \
    --schedule karras --sampler stochastic --sampler_kind ddim --gamma 0.0 \
    --steps "$4" --limit "$3" --seed "$1" --batch_size "$2" \
    --out_dir "$EV_OUT_DIR" --regime "$EV_REGIME" --tag "$5"
}

if [ -n "${EV_PROBE:-}" ]; then
  PS="${EV_PROBE_STEPS:-16}"
  echo "[tev] PROBE batches={$EV_PROBE} steps=$PS"
  for b in ${EV_PROBE//:/ }; do
    if run_eval 0 "$b" "$b" "$PS" "probe_${EV_RUN}__b${b}"; then
      echo "[tev] probe b=$b OK"
    else
      echo "[tev] probe b=$b FAILED (rc=$?; OOM if the log above says so)"
    fi
  done
  echo "[tev] finished $(date)"; exit 0
fi

EV_SEED="${EV_SEED:?set EV_SEED}"
EV_BATCH="${EV_BATCH:?set EV_BATCH}"
echo "[tev] seed=$EV_SEED batch=$EV_BATCH steps=$EV_STEPS limit=$EV_LIMIT out=$EV_OUT_DIR regime=$EV_REGIME"
set +e
run_eval "$EV_SEED" "$EV_BATCH" "$EV_LIMIT" "$EV_STEPS" "${EV_RUN}__ck${EV_STEP}__s${EV_SEED}"
rc=$?; set -e
[ $rc -ne 0 ] && { echo "[tev] FAILED rc=$rc"; exit $rc; }
echo "[tev] finished $(date)"
