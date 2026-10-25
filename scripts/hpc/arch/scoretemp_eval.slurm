#!/bin/bash
#SBATCH --job-name=cobit_stev
#SBATCH --partition=ampere
#SBATCH --account=GIROLAMI-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/arch/%x_%j.log
#
# ONE cell of the Sigma-style ("global") score-temperature study on GSM8K:
# one checkpoint, one (w, tau, gamma, steps), seed 0, the full 1319 problems.
# Submit through scoretemp_gh200.sh, which fills in the site flags, gates the
# sweep on a smoke job, and never submits a cell twice.
#
#   ST_ARM      label (s425 | cfg425 | m250 | ...), part of the result tag
#   ST_CFG      config file          ST_CK   checkpoint
#   ST_TAU      score temperature    ST_W    CFG weight (0 = conditional only)
#   ST_GAMMA    EDM churn (0 => deterministic DDIM)
#   ST_STEPS    sampler steps        ST_LABEL  unique cell label (result tag)
#   [ST_SIGMA_DATA]  override; unset => the run's sigma_data.json sidecar
#   [ST_DIAG=0]      1 => --collect_diagnostics (per-step cfg_dir_rms vs sigma)
#   [ST_LIMIT=1319] [ST_BATCH=64] [ST_SEED=0] [ST_OUT_DIR] [ST_GATE_MIN_ACC]
#   [ST_SAMPLER_KIND=ddim]  ddim | ancestral (ancestral needs ST_GAMMA=0) | em
#   [ST_ST_FINAL=sharpened] unsharpened = last step uses the untempered drift
#   [ST_SCHEDULE=karras] [ST_EMA=1] [ST_SIGMA_MIN] (unset = config 0.002)
#   [ST_ENTROPY_DIR]        entropic grid + churn window table (unset = the run dir), e.g. a DGC table
#
# PROTOCOL: Karras schedule (analytic, identical for every checkpoint; the
# entropic schedule is refitted per run and the unguided CoBit-S run has no
# entropy tables on this cluster), DDIM, EMA, --score_temp_mode global, i.e.
# the drift divided by tau at every sigma, after the CFG mix.
#
# ST_GATE_MIN_ACC set => gate mode: fail the job (and so every afterok sweep
# job) unless accuracy >= that value.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/rds/user/rg625/hpc-work/BitstreamDiffusion}"
export COBIT_PYTHON="${COBIT_PYTHON:-/home/rg625/miniforge3/envs/sedd310/bin/python}"
source "$PROJECT_DIR/scripts/hpc/guidance/env.sh"
cd "$PROJECT_DIR"

ST_ARM="${ST_ARM:?}"; ST_CFG="${ST_CFG:?}"; CK="${ST_CK:?}"
ST_TAU="${ST_TAU:?}"; ST_W="${ST_W:?}"; ST_GAMMA="${ST_GAMMA:?}"
ST_STEPS="${ST_STEPS:?}"; ST_LABEL="${ST_LABEL:?}"
ST_LIMIT="${ST_LIMIT:-1319}"; ST_BATCH="${ST_BATCH:-64}"; ST_SEED="${ST_SEED:-0}"
ST_OUT_DIR="${ST_OUT_DIR:-results/scoretemp/eval_gh200}"
mkdir -p logs/arch "$ST_OUT_DIR"
[ -f "$CK" ] || { echo "[st] FATAL: missing $CK" >&2; exit 1; }

# Churn's window is an entropy-CDF quantile band read from the run directory;
# without those tables resolve_stochastic_cfg falls back to DETERMINISTIC with
# no error (12 Oct: all 21 s425 "gamma=0.41" cells silently ran as gamma=0).
RUN_DIR="$(dirname "$(dirname "$(readlink -f "$CK")")")"   # gsm8k_eval resolves symlinks too
TABLE_DIR="${ST_ENTROPY_DIR:-$RUN_DIR}"
if awk "BEGIN{exit !($ST_GAMMA > 0)}" && [ ! -f "$TABLE_DIR/entropy_cdf.pt" ]; then
  echo "[st] FATAL: gamma=$ST_GAMMA but $TABLE_DIR has no entropy tables -> churn would silently be off" >&2
  exit 1
fi
if awk "BEGIN{exit !($ST_GAMMA > 0)}"; then SAMPLER=(--sampler stochastic --gamma "$ST_GAMMA")
else SAMPLER=(--sampler deterministic --gamma 0.0); fi
EXTRA=()
[ -n "${ST_SIGMA_DATA:-}" ] && EXTRA+=(--sigma_data "$ST_SIGMA_DATA")
[ -n "${ST_ENTROPY_DIR:-}" ] && EXTRA+=(--entropy_dir "$ST_ENTROPY_DIR")
[ -n "${ST_GRID_FILE:-}" ] && EXTRA+=(--grid_file "$ST_GRID_FILE")   # precomputed sigma grid (gamma-Euler study)
[ "${ST_DIAG:-0}" = 1 ] && EXTRA+=(--collect_diagnostics)
[ -n "${ST_SIGMA_MIN:-}" ] && EXTRA+=(--sigma_min "$ST_SIGMA_MIN")   # end the trajectory here (e.g. 0.1, as on OWT)
# DeepMind-style logit temperature (softmax(logits/T) over valid codewords, every step)
[ -n "${ST_POST_TEMP:-}" ] && EXTRA+=(--posterior_temp "$ST_POST_TEMP" --posterior_temp_space "${ST_POST_SPACE:-token}" \
                                      --posterior_temp_target "${ST_POST_TARGET:-full}")

echo "[st] commit $(git rev-parse --short HEAD) | python $($COBIT_PYTHON -V 2>&1)"
nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader
echo "[st] arm=$ST_ARM ck=$CK cfg=$ST_CFG w=$ST_W tau=$ST_TAU gamma=$ST_GAMMA steps=$ST_STEPS limit=$ST_LIMIT label=$ST_LABEL"

"$COBIT_PYTHON" evaluation/tasks/gsm8k_eval.py \
  --config "$ST_CFG" --checkpoint "$CK" --ema "${ST_EMA:-1}" \
  --schedule "${ST_SCHEDULE:-karras}" --sampler_kind "${ST_SAMPLER_KIND:-ddim}" "${SAMPLER[@]}" \
  --guidance_scale "$ST_W" \
  --score_temp_tau "$ST_TAU" --score_temp_mode global --score_temp_final "${ST_ST_FINAL:-sharpened}" \
  --steps "$ST_STEPS" --limit "$ST_LIMIT" --seed "$ST_SEED" --batch_size "$ST_BATCH" \
  --out_dir "$ST_OUT_DIR" --regime scoretemp_gh200 --tag "$ST_LABEL" \
  ${EXTRA[@]+"${EXTRA[@]}"}

if [ -n "${ST_GATE_MIN_ACC:-}" ]; then
  f=$(ls -t "$ST_OUT_DIR"/gsm8k_results_*"$ST_LABEL"*.json | head -1 || true)
  "$COBIT_PYTHON" - "$f" "$ST_GATE_MIN_ACC" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1])); acc = float(r["accuracy"]); lo = float(sys.argv[2])
print(f"[st] GATE {sys.argv[1]}: acc={acc:.3f} exec={r.get('executable_rate')} need>={lo}")
sys.exit(0 if acc >= lo else 3)
PYEOF
fi
echo "[st] finished $(date)"
