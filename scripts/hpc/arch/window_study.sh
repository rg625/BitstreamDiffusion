#!/bin/bash
# Window-ordering study driver (Isambard GH200). Dry run unless --go.
#
#   bash scripts/hpc/arch/window_study.sh plan               # list arms + cost
#   bash scripts/hpc/arch/window_study.sh train --go         # chain every arm
#   bash scripts/hpc/arch/window_study.sh eval  --go         # idempotent evals
#   bash scripts/hpc/arch/window_study.sh status
#
# Arms: control (k=inf) + {l2r,r2l,random} x K_LIST, plus the supervision-
# matched controls (k=inf, loss subsampled like window-k) when LOSSMATCH=1.
# Every arm x SEEDS. Same recipe, data, tau/noise streams per seed.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_DIR"
source scripts/site/site.sh >/dev/null
PY="$COBIT_PYTHON"

PHASE="${1:?usage: window_study.sh plan|train|eval|status [--go]}"
GO=0; [ "${2:-}" = "--go" ] && GO=1
STEPS="${WIN_STEPS:-300000}"
SEEDS="${SEEDS:-42 43}"
K_LIST="${K_LIST:-64 16 4}"
LOSSMATCH="${LOSSMATCH:-1}"
LINKS="${LINKS:-2}"
TAG="${WIN_TAG:-win}"
EVAL_NFES_FINAL="${EVAL_NFES_FINAL:-256 1024}"

arms () {   # "mode:k:lossmatch" per line
  echo "none:inf:"
  for k in $K_LIST; do for m in l2r r2l random; do echo "$m:$k:"; done; done
  if [ "$LOSSMATCH" = 1 ]; then for k in $K_LIST; do echo "none:inf:$k"; done; fi
}
run_name () {  # mode k lm seed
  local kt="k$2"; [ -n "$3" ] && kt="${kt}_lm$3"; echo "${TAG}_$1_${kt}_s$4"
}
global_step () {
  "$PY" - "$1" <<'PYEOF'
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["global_step"]))
except Exception:
    print("none")
PYEOF
}
submit () { bash scripts/site/submit.sh "$@" --parsable 2>/dev/null; }

case "$PHASE" in
  plan|train)
    n=0
    for a in $(arms); do for s in $SEEDS; do n=$((n+1)); done; done
    gph=$(awk -v s="$STEPS" -v n="$n" 'BEGIN{printf "%.0f", n*s/3.4/3600*4}')
    echo "[win] $n runs x $STEPS steps @ ~3.4 steps/s on 4 GH200 => ~${gph} GPU-h training"
    echo "[win] arms: $(arms | tr '\n' ' ')  seeds: $SEEDS  links: $LINKS x 24h"
    [ "$PHASE" = plan ] && exit 0
    [ $GO -eq 1 ] || { echo "dry run. Add --go."; exit 0; }
    for a in $(arms); do
      IFS=: read -r m k lm <<<"$a"
      for s in $SEEDS; do
        r=$(run_name "$m" "$k" "$lm" "$s")
        if squeue -u "$USER" -h -n "win_$r" | grep -q .; then echo "  $r: already queued, skipping"; continue; fi
        prev=""
        for i in $(seq 1 "$LINKS"); do
          dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:"$prev")
          j=$(submit scripts/hpc/arch/window_train.slurm --job-name="win_$r" --time=24:00:00 \
                ${dep[@]+"${dep[@]}"} \
                --export="ALL,WIN_MODE=$m,WIN_K=$k,WIN_SEED=$s,WIN_STEPS=$STEPS,WIN_TAG=$TAG,WIN_LOSSMATCH=$lm")
          [ -n "$j" ] || { echo "submit failed for $r link $i -- STOPPING" >&2; exit 1; }
          prev="$j"
        done
        echo "  $r: chain ending in job $prev"
      done
    done
    ;;
  eval)
    # SAFETY: (1) nothing is evaluated until the production anchor has landed
    # in band IN THIS DIRECTORY; (2) one attempt per checkpoint -- a job that
    # ran and left cells missing is reported, never resubmitted automatically;
    # (3) no evals for HALTed runs or under WINDOW_GLOBAL_HALT.
    OUT=results/ordering/window_eval_gh200
    [ -f runs/tasks/tinygsm/WINDOW_GLOBAL_HALT ] && { echo "ALERT global halt present -- no evals submitted"; exit 0; }
    ANCHOR=tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt
    gate=$("$PY" - "$OUT" <<'PYEOF'
import glob, json, sys
accs = {}
for f in glob.glob(f"{sys.argv[1]}/*prod_anchor__*__standard__n256__*.json"):
    d = json.load(open(f)); accs[int(d.get("seed") or 0)] = float(d["accuracy"])
ok = set(accs) >= {0, 1, 2} and all(0.12 <= a <= 0.20 for a in accs.values())
bad = any(not (0.12 <= a <= 0.20) for a in accs.values())
print("pass" if ok else ("fail " + str(accs) if bad else "pending " + str(accs)))
PYEOF
)
    case "$gate" in
      pass) ;;
      fail*) echo "ALERT anchor OUT OF BAND ($gate) -- decode path suspect, no arm evals"; exit 0 ;;
      *)
        if ! squeue -u "$USER" -h -o "%j" | grep -qx "wev_prod_anchor" && \
           ! compgen -G "logs/arch/wev_prod_anchor_*.log" >/dev/null; then
          echo "  submit anchor gate"
          [ $GO -eq 1 ] && submit scripts/hpc/arch/window_eval.slurm --job-name=wev_prod_anchor --time=01:00:00 \
             --export="ALL,EV_RUN=prod_anchor,EV_CK_PATH=$ANCHOR,EV_MODE=none,EV_K=inf,EV_DECODERS=standard,EV_NFES=256" >/dev/null
        elif ! squeue -u "$USER" -h -o "%j" | grep -qx "wev_prod_anchor"; then
          echo "ALERT anchor job ran but the gate is not complete ($gate) -- inspect logs/arch/wev_prod_anchor_*.log"
        else echo "  anchor gate pending ($gate)"; fi
        exit 0 ;;
    esac
    queued=$(squeue -u "$USER" -h -o "%j")
    for a in $(arms); do
      IFS=: read -r m k lm <<<"$a"
      evmode=$m; [ -n "$lm" ] && evmode=none
      for s in $SEEDS; do
        r=$(run_name "$m" "$k" "$lm" "$s"); d="runs/tasks/tinygsm/$r/checkpoints"
        [ -d "$d" ] || continue
        [ -f "runs/tasks/tinygsm/$r/HALT" ] && continue
        for ck in "$d"/step=*.pt; do
          [ -f "$ck" ] || continue
          st=$(basename "$ck" .pt); st=$((10#${st#step=}))
          nfes=256; [ "$st" -ge "$STEPS" ] && nfes="$EVAL_NFES_FINAL"
          tagck=$(printf "%09d" "$st"); jn="wev_${r}_${tagck}"
          todo=0
          for n in $nfes; do for dec in own standard; do for es in 0 1 2; do
            compgen -G "$OUT/*${r}__ck${tagck}__${dec}__n${n}__s${es}_*.json" >/dev/null || todo=1
          done; done; done
          [ $todo = 1 ] || continue
          grep -qx "$jn" <<<"$queued" && continue
          if compgen -G "logs/arch/${jn}_*.log" >/dev/null; then
            echo "ALERT eval $r @ $st already attempted and cells are missing -- not resubmitting ($(ls logs/arch/${jn}_*.log | tail -1))"
            continue
          fi
          echo "  eval $r @ $st  nfes=[$nfes]"
          [ $GO -eq 1 ] && submit scripts/hpc/arch/window_eval.slurm --job-name="$jn" \
             --time=$([ "$nfes" = 256 ] && echo 01:00:00 || echo 03:30:00) \
             --export="ALL,EV_RUN=$r,EV_MODE=$evmode,EV_K=$k,EV_CKPT=$(basename "$ck"),EV_NFES=$nfes" >/dev/null
        done
      done
    done
    [ $GO -eq 1 ] || echo "dry run. Add --go."
    ;;
  status)
    for a in $(arms); do IFS=: read -r m k lm <<<"$a"; for s in $SEEDS; do
      r=$(run_name "$m" "$k" "$lm" "$s"); c="runs/tasks/tinygsm/$r/checkpoints/last.pt"
      printf "  %-32s step=%-8s %s\n" "$r" "$([ -f "$c" ] && global_step "$c" || echo -)" \
        "$(squeue -u "$USER" -h -n "win_$r" -o '%T' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
    done; done
    ;;
  *) echo "unknown phase $PHASE" >&2; exit 1 ;;
esac
