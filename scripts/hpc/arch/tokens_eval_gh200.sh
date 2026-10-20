#!/bin/bash
# GSM8K evaluation of the V-way TOKEN-softmax arm against its matched BITS
# control, ON ISAMBARD, at every archived checkpoint (a learning curve, not one
# endpoint).
#
#   bash scripts/hpc/arch/tokens_eval_gh200.sh probe  [--go]  # eval batch-size memory probe (1 GPU)
#   bash scripts/hpc/arch/tokens_eval_gh200.sh pick           # read the probe, fix the token batch
#   bash scripts/hpc/arch/tokens_eval_gh200.sh gate   [--go]  # both decode paths on known checkpoints
#   bash scripts/hpc/arch/tokens_eval_gh200.sh sweep  [--go] [--step N]
#   bash scripts/hpc/arch/tokens_eval_gh200.sh report         # table + paired deltas
#
# Every phase is idempotent: sweep submits only the cells that have neither a
# result JSON nor a queued/running job, so running it again after each new
# checkpoint lands evaluates exactly the new cells.
#
# ARMS (both trained on GH200, matched in everything but the head):
#   tok_gh500k_token_ce_s42   V-way softmax over the full 49,153-token vocabulary
#   ord_gh500k_none_s42       binary bits, binary_sm
# Checkpoints every 100k (step=000100000.pt ... step=000500000.pt); 3 seeds per
# checkpoint, one GPU per cell.
#
# GATE before any arm hour: both decode paths on checkpoints whose answers are
# known. A sampler bug once scored 0.000 against a true 0.164 and was caught
# only by an anchor.
#   bits  : production anchor, band 0.12-0.20 (GH200 ordered-kind eval: 0.127/0.135)
#   token : tok_fs500k_token_ce_s42 @500k, CSD3 scored 0.022/0.026/0.029, band 0.010-0.045
# The token anchor is the only token checkpoint with a known score; it is
# evaluated here on GH200, never pooled with its CSD3 numbers.
#
# Everything lands in results/tokens/eval_gh200 and is compared only within it.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../site/site.sh" >/dev/null   # COBIT_PYTHON / COBIT_ACCOUNT, or refuse
PY="$COBIT_PYTHON"

PHASE="${1:?usage: tokens_eval_gh200.sh probe|pick|gate|sweep|report [--go] [--step N]}"; shift
GO=0; ONLY_STEP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --step) ONLY_STEP="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

OUT="results/tokens/eval_gh200"
PROBE_OUT="$OUT/probe"
REGIME="tokens_vs_bits__eval_gh200"
LAUNCHER="scripts/hpc/arch/tokens_eval.slurm"
TOK_BATCH_FILE="$OUT/token_batch.txt"   # "<batch> <samples/s> <probe steps>", written by `pick`
BITS_BATCH=64                           # as every GH200 bits eval so far; 2.7 GB peak
SEEDS="0 1 2"

TOK_ARM="tok_gh500k_token_ce_s42"
CTL_ARM="ord_gh500k_none_s42"
TOK_ANCHOR="runs/tasks/tinygsm/tok_fs500k_token_ce_s42/checkpoints/last.pt"   # global_step 500000; only last.pt was transferred
BITS_ANCHOR="tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt"
PROBE_BATCHES="${PROBE_BATCHES:-16 24 32 40 48 56 64 80 96}"

mkdir -p "$OUT"

queued_names () {  # a failed squeue is UNKNOWN, never "nothing queued"
  timeout 60 squeue -h -u "$USER" -o "%j" || { echo "[tev] squeue failed -- STOPPING" >&2; exit 1; }
}

submit () {  # $1=job name $2=wall $3=exports
  local out rc
  set +e
  out=$(bash scripts/site/submit.sh "$LAUNCHER" --parsable --job-name="$1" --time="$2" --export="$3"); rc=$?
  set -e
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    echo "[tev] submit rc=$rc out='$out' -- STOPPING. rc!=0 is UNKNOWN, not 'not queued':" >&2
    echo "[tev] check squeue -u \$USER before re-running." >&2
    exit 1
  fi
  echo "[tev] job $out  $1"
}

tok_batch () { [ -f "$TOK_BATCH_FILE" ] && awk '{print $1}' "$TOK_BATCH_FILE"; }

# Wall for one full 1319-problem cell. The token arm's comes from the probe's
# measured throughput, rescaled from the probe's step count to the eval's (a
# 16-step probe runs 16x faster per sample than a 256-step eval), x2 margin,
# +20 min for load; the bits arm took 280 s on GH200.
tok_wall () {
  "$PY" - "$TOK_BATCH_FILE" "${EV_STEPS:-256}" <<'PYEOF'
import sys
_, sps, probe_steps = open(sys.argv[1]).read().split()
sps = float(sps) * int(probe_steps) / int(sys.argv[2])
s = int(2 * 1319 / sps + 1200)
s = min(s, 24 * 3600)
print(f"{s // 3600:02d}:{s % 3600 // 60:02d}:00")
PYEOF
}

has_result () { compgen -G "$OUT/gsm8k_results_*_$1.json" >/dev/null; }

# $1=arm(token|bits) $2=run label $3=ckpt $4=step(9 digits) -- one job per seed
submit_cells () {
  local arm="$1" run="$2" ck="$3" step="$4" names s tag batch wall
  names=$(queued_names)
  if [ "$arm" = token ]; then batch=$(tok_batch); wall=$(tok_wall); else batch=$BITS_BATCH; wall=01:00:00; fi
  for s in $SEEDS; do
    tag="${run}__ck${step}__s${s}"
    if has_result "$tag"; then echo "  done    $tag"; continue; fi
    if grep -qx "tev_$tag" <<<"$names"; then echo "  queued  $tag"; continue; fi
    echo "  submit  $tag  (batch $batch, wall $wall)"
    # An if, not `[ $GO -eq 1 ] && submit`: in a dry run that list returns 1,
    # becomes the function's status, and set -e ends the script mid-sweep.
    if [ $GO -eq 1 ]; then
      submit "tev_$tag" "$wall" \
        "ALL,EV_ARM=$arm,EV_RUN=$run,EV_CK_PATH=$ck,EV_SEED=$s,EV_BATCH=$batch,EV_OUT_DIR=$OUT,EV_REGIME=$REGIME"
    fi
  done
}

need_batch () {
  [ -f "$TOK_BATCH_FILE" ] || { echo "REFUSING: no $TOK_BATCH_FILE -- run probe, then pick" >&2; exit 1; }
}

case "$PHASE" in
  probe)
    [ -f "$TOK_ANCHOR" ] || { echo "MISSING $TOK_ANCHOR" >&2; exit 1; }
    echo "would submit 1 probe job: token model, batches {$PROBE_BATCHES}, 16 steps each -> $PROBE_OUT"
    [ $GO -eq 1 ] || { echo "dry run. Add --go."; exit 0; }
    submit "tev_probe" 02:00:00 \
      "ALL,EV_ARM=token,EV_RUN=tok_fs500k_token_ce_s42,EV_CK_PATH=$TOK_ANCHOR,EV_OUT_DIR=$PROBE_OUT,EV_REGIME=probe,EV_PROBE=${PROBE_BATCHES// /:}"
    echo "[tev] then: bash $0 pick"
    ;;
  pick)
    # Largest batch that ran to completion with peak <= 97% of the card. Every
    # batch has the same shape, [B, 512, 49153], so the probe's peak IS the
    # eval's peak; the margin covers the CUDA context, which max_memory_allocated
    # does not count. Measured 24 Sep: b=64 peaks at 91.3 of 95 GB, b=80 OOMs,
    # and samples/s is flat from b=16 up (the GPU is saturated), so batch buys
    # memory headroom, not speed.
    "$PY" - "$PROBE_OUT" "$TOK_BATCH_FILE" 16 <<'PYEOF'
import glob, json, re, sys
rows = []
for f in glob.glob(f"{sys.argv[1]}/*.json"):
    m = re.search(r"__b(\d+)\.json$", f)
    if not m:
        continue
    e = json.load(open(f)).get("efficiency") or {}
    rows.append((int(m.group(1)), e.get("peak_gpu_gb"), e.get("samples_per_sec")))
if not rows:
    sys.exit("no probe results yet")
cap = 95.0   # GH200 usable HBM3, GB
print(f"{'batch':>6} {'peak GB':>8} {'samples/s':>10}")
for b, p, s in sorted(rows):
    print(f"{b:>6} {p:>8.1f} {s:>10.3f}")
ok = [(b, p, s) for b, p, s in rows if p is not None and p <= 0.97 * cap]
if not ok:
    sys.exit("no probed batch fits")
b, p, s = max(ok)
open(sys.argv[2], "w").write(f"{b} {s} {sys.argv[3]}\n")
print(f"picked batch {b} (peak {p:.1f} GB of {cap:.0f}) -> {sys.argv[2]}")
PYEOF
    ;;
  gate)
    need_batch
    for f in "$TOK_ANCHOR" "$BITS_ANCHOR"; do [ -f "$f" ] || { echo "MISSING $f" >&2; exit 1; }; done
    echo "token anchor (tok_fs500k @500k):"
    submit_cells token tok_fs500k_token_ce_s42 "$TOK_ANCHOR" 000500000
    echo "bits anchor (production):"
    submit_cells bits prod_anchor "$BITS_ANCHOR" 000500000
    [ $GO -eq 1 ] || echo "dry run. Add --go."
    ;;
  sweep)
    need_batch
    "$PY" experiments/tokens/analyze.py --dir "$OUT" --gate-only \
      || { echo "[tev] gate has not passed in $OUT -- refusing to spend arm hours" >&2; exit 1; }
    for spec in "token:$TOK_ARM" "bits:$CTL_ARM"; do
      IFS=: read -r arm run <<<"$spec"
      shopt -s nullglob
      cks=(runs/tasks/tinygsm/"$run"/checkpoints/step=*.pt)
      shopt -u nullglob
      [ ${#cks[@]} -gt 0 ] || { echo "$run: no archived checkpoints yet"; continue; }
      for ck in "${cks[@]}"; do
        step=$(sed -E 's/.*step=([0-9]+)\.pt$/\1/' <<<"$ck")
        [ -n "$ONLY_STEP" ] && [ "$((10#$step))" != "$ONLY_STEP" ] && continue
        echo "$run @ $((10#$step)):"
        submit_cells "$arm" "$run" "$ck" "$step"
      done
    done
    [ $GO -eq 1 ] || echo "dry run. Add --go."
    ;;
  report)
    "$PY" experiments/tokens/analyze.py --dir "$OUT" --token "$TOK_ARM" --bits "$CTL_ARM" \
      --out results/tokens/summary_gh200.json
    ;;
  *) echo "unknown phase $PHASE (probe|pick|gate|sweep|report)" >&2; exit 1 ;;
esac
