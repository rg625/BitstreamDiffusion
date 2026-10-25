#!/bin/bash
# DGC grid vs the entropy-rate grid on TinyGSM (GSM8K), Sudoku and OWT. One GPU per job.
# Only the sigma grid (and with it the churn window) changes; tables: scripts/dgc/build_dgc_tables.py.
#
#   bash scripts/dgc/tasks_ab.sh gate          # dry run
#   bash scripts/dgc/tasks_ab.sh gate --go     # submit (skips cells with a result or a queued job)
#   bash scripts/dgc/tasks_ab.sh main --go
#
# gate (~0.3 GPU-h): GSM8K s500 both arms on 128 problems (the cobit arm must reproduce the stored
#   per-problem answers of results/scoretemp/eval_gh200, so those baselines can be reused);
#   Sudoku medium gamma 0.4 seed 42 both arms (times a cell); OWT S and M both arms on 64 samples.
# main: GSM8K  s500 (CoBit-S unguided 500k EMA): DGC at tau 1 and at tau .3 unsharpened, 1024 steps
#              (paired with the stored cobit cells); both arms at 256 steps.
#              m250 (CoBit-M 250k): both arms at tau 1 and tau .3 unsharpened (its stored cells are Karras).
#       Sudoku 3 difficulties x {det, gamma .4} x seeds 42-44, both arms, raw weights.
#       OWT    CoBit-S 256/.13 and CoBit-M Table 2 (256/.21, 384/.24, 512/.26), both arms, seed 42.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"
WHAT="${1:?gate|gate-owt|main|main-tasks|main-owt|sweep-owt|tgsm-screen|tgsm-confirm|mechanism|seeds-owt|t1|lm1b-gate|lm1b-sweep|dp-smoke|dp-owt|dp-tasks}"; GO=0; [ "${2:-}" = "--go" ] && GO=1

GSM_OUT=results/dgc/gsm8k
DP_K="${DP_K:-8}"   # K blocks for the dp_dgc grid
DP_W="${DP_W:-increment}"   # dp_dgc block weight: increment (the notebook's G_k) | length_squared (Part 3 runs)
OWT_SEED=42  # OWT sampling seed; seeds-owt sets 43/44
GSM_G=0.41   # churn of the GSM8K cells; the tgsm-screen phase varies it
SWEEP_G="${SWEEP_G:-0.05 0.1 0.2 0.3 0.41}"   # sweep-owt gammas; 0.41 ~ the per-step churn cap sqrt(2)-1
S_CK=runs/tasks/tinygsm/cobit_raw_binary_bits/checkpoints/step=000500000.pt
M_CK=runs/tasks/tinygsm/cobit_m_bits_s42/checkpoints/step=000250000.pt
S_TAB=(runs/tasks/tinygsm/cobit_raw_binary_bits assets/entropy_tables/tasks/tinygsm_cobit_s_unguided_dgc)
M_TAB=(runs/tasks/tinygsm/cobit_m_bits_s42 assets/entropy_tables/tasks/tinygsm_cobit_m_dgc)
SD_S=0.399844765663147   # what the stored s500 cells used; m250 reads its sidecar
mkdir -p "$GSM_OUT" logs/arch
QUEUED="$(timeout 60 squeue -h -u "$USER" -o "%j")" || { echo "squeue failed -- STOPPING" >&2; exit 1; }
N=0

submit () {  # $1=job name $2=launcher $3=wall $4=exports
  if grep -qx "$1" <<<"$QUEUED"; then echo "skip $1 (queued)"; return; fi
  if [ $GO -eq 1 ]; then
    id=$(bash scripts/site/submit.sh "$2" --parsable --job-name="$1" --time="$3" --export="$4" ${DGC_EXCLUDE:+--exclude=$DGC_EXCLUDE} 2>/dev/null)
    echo "$id $1"
  else
    echo "DRY $1 ($3)"
  fi
  N=$((N + 1))
}

gsm () {  # $1=arm label (s500|m250) $2=grid (cobit|dgc|dpdgc) $3=tau $4=steps $5=limit $6=wall [$7 extra]; churn = $GSM_G
  local ck tab cfg sd=""
  if [ "$1" = s500 ]; then ck=$S_CK; cfg=configs/tasks/tinygsm_bits.py; sd=",ST_SIGMA_DATA=$SD_S"; [ "$2" = cobit ] && tab=${S_TAB[0]} || tab=${S_TAB[1]}
  else ck=$M_CK; cfg=configs/tasks/tinygsm_bits_cobit_m.py; [ "$2" = cobit ] && tab=${M_TAB[0]} || tab=${M_TAB[1]}; fi
  local sched=entropic; [ "$2" = dpdgc ] && sched=dp_dgc   # dpdgc: the DGC table, finite-NFE DP allocation
  local fin=""; [ "$3" != 1 ] && fin=",ST_ST_FINAL=unsharpened"
  local label="dgc_$1_$2_t$3_g${GSM_G}_n$4_l$5"
  if ls "$GSM_OUT"/gsm8k_results_*"__$label"*.json >/dev/null 2>&1; then echo "skip $label (done)"; return; fi
  submit "$label" scripts/hpc/arch/scoretemp_eval.slurm "$6" \
    "ALL,ST_ARM=$1,ST_CFG=$cfg,ST_CK=$ck,ST_W=0,ST_TAU=$3,ST_GAMMA=$GSM_G,ST_STEPS=$4,ST_LIMIT=$5,ST_SCHEDULE=$sched,ST_ENTROPY_DIR=$tab,ST_OUT_DIR=$GSM_OUT,ST_LABEL=__$label$fin$sd${7:+,$7}"
}

sudoku () {  # $1=difficulty $2=gammas $3=seeds $4=wall [$5 extra exports]
  submit "dgc_sudoku_$1_$(tr ' ' _ <<<"$2")_$(tr ' ' _ <<<"$3")${5:+_$(tr ',= ' '___' <<<"$5")}" scripts/hpc/arch/dgc_sudoku.slurm "$4" \
    "ALL,SU_DIFF=$1,SU_GAMMAS=$2,SU_SEEDS=$3${5:+,$5}"
}

lm1b () {  # $1=arm $2=wall $3=cells ("nfe:gamma/...") [$4 samples]
  local dir="runs/paper/unconditional_text/lm1b/continuous_rate_raw_binary_bits_1M_edm_weighting/dgc_sweep/$1/seed42_c$(sed 's/:/g/g; s/\//_/g' <<<"$3")${4:+_smoke$4}"
  if grep -qs gen_full_token_unigram_entropy "$dir/results.jsonl"; then echo "skip lm1b $1 $3 (done)"; return; fi
  submit "dgc_lm1b_$1_$(tr '/' '_' <<<"$3")${4:+_n$4}" scripts/hpc/arch/dgc_lm1b.slurm "$2" \
    "ALL,SCHEDULE=$1,EVAL_SEED=42,CELLS=$3${4:+,DGC_NUM_SAMPLES=$4}"
}

owt () {  # $1=model $2=arm $3=wall [$4 samples] [$5 cells]
  local exp; [ "$1" = s ] && exp=continuous_rate_raw_binary_bits_1M || exp=continuous_rate_raw_binary_bits_medium_24x1024
  local armdir="$2"; [ "$2" = dp_dgc ] && armdir="dp_dgc_K$DP_K$([ "$DP_W" = increment ] || echo _lensq)"
  local dir="runs/paper/unconditional_text/owt/$exp/dgc_ab/$armdir/seed${OWT_SEED}${5:+_c$(sed 's/:/g/g; s/[,/]/_/g' <<<"$5")}${4:+_smoke$4}"
  if grep -qs gen_full_token_unigram_entropy "$dir/results.jsonl"; then echo "skip $1 $2 $5 (done)"; return; fi
  submit "dgc_owt_$1_$2${4:+_n$4}${5:+_${5//\//_}}_s$OWT_SEED" scripts/hpc/arch/dgc_owt.slurm "$3" \
    "ALL,MODEL=$1,SCHEDULE=$2,EVAL_SEED=$OWT_SEED,DP_DGC_BLOCKS=$DP_K,DP_DGC_WEIGHT=$DP_W${4:+,DGC_NUM_SAMPLES=$4}${5:+,CELLS=$5}"
}

case "$WHAT" in
  gate)
    for arm in cobit dgc; do gsm s500 "$arm" 1 1024 128 00:30:00; done
    sudoku medium "0.4" "42" 00:45:00
    for m in s m; do for arm in cobit dgc; do owt "$m" "$arm" 00:45:00 64 "256:0.13"; done; done
    ;;
  gate-owt)
    for m in s m; do for arm in cobit dgc; do owt "$m" "$arm" 00:45:00 64 "256:0.13"; done; done
    ;;
  main|main-tasks)
    for t in 1 0.3; do gsm s500 dgc "$t" 1024 1319 01:00:00; done
    for arm in cobit dgc; do gsm s500 "$arm" 1 256 1319 00:30:00; done
    for arm in cobit dgc; do for t in 1 0.3; do gsm m250 "$arm" "$t" 1024 1319 02:00:00; done; done
    for d in easy medium hard; do sudoku "$d" "0 0.4" "42 43 44" "${SU_WALL:-01:00:00}"; done
    ;;&
  main|main-owt)
    # one job per CoBit-M cell so the slowest (512 NFE) sets the wall-clock, not their sum
    for arm in cobit dgc; do
      owt s "$arm" 01:00:00 "" "256:0.13"
      for c in 256:0.21 384:0.24 512:0.26; do owt m "$arm" "${OWT_M_WALL:-02:30:00}" "" "$c"; done
    done
    ;;
  tgsm-screen)
    # DGC x score temperature on CoBit-S unguided 500k, 1024 steps, unsharpened last step. The standard-grid
    # partners of the gamma 0.2/0.3 and tau 0.2/0.5 cells are stored (results/scoretemp/blocks_gh200, reproduced
    # exactly by the gate); tau 0.7 has no unsharpened partner, so both grids run there.
    for g in 0.2 0.3; do GSM_G=$g gsm s500 dgc 0.3 1024 1319 01:00:00; done
    for t in 0.2 0.5; do gsm s500 dgc "$t" 1024 1319 01:00:00; done
    for arm in cobit dgc; do gsm s500 "$arm" 0.7 1024 1319 01:00:00; done
    ;;
  tgsm-confirm)
    # the screen's best DGC setting on CoBit-M, both grids: TGSM_TAU / TGSM_G
    for arm in cobit dgc; do GSM_G=${TGSM_G:?} gsm m250 "$arm" "${TGSM_TAU:?}" 1024 1319 02:00:00; done
    ;;
  mechanism)
    # Where the churn acts vs where the steps are: swapped arms (steps from one table, churn density from
    # the other). Gammas chosen so the per-step cap barely binds (CPU check, 21 Oct).
    submit dgc_sudoku_hard_mech_0.15 scripts/hpc/arch/dgc_sudoku.slurm 00:45:00 \
      "ALL,SU_DIFF=hard,SU_GAMMAS=0.15,SU_SEEDS=42 43 44,SU_ARMS=cobit dgc cobit_dgcchurn dgc_stdchurn"
    for arm in cobit dgc; do
      submit "dgc_owt_s_${arm}_swapchurn" scripts/hpc/arch/dgc_owt.slurm 00:45:00 \
        "ALL,MODEL=s,SCHEDULE=$arm,EVAL_SEED=42,CELLS=256:0.05/256:0.13,CHURN_SWAP=1"
    done
    ;;
  seeds-owt)
    # Seeds 43/44 on the points that decide the comparison (picked from the seed-42 sweep, 21 Oct):
    # CoBit-S: the gammas bracketing where the two curves cross; CoBit-M: each grid's lowest-GenPPL and
    # highest-entropy point at each NFE.
    PTS="s:cobit:256:0.2 s:cobit:256:0.3 s:dgc:256:0.13 s:dgc:256:0.2
         s:cobit:384:0.3 s:cobit:384:0.41 s:dgc:384:0.2 s:dgc:384:0.3
         s:cobit:512:0.3 s:cobit:512:0.41 s:dgc:512:0.2 s:dgc:512:0.3
         m:cobit:256:0.1 m:cobit:256:0.3 m:dgc:256:0.2 m:dgc:256:0.41
         m:cobit:384:0.2 m:cobit:384:0.3 m:dgc:384:0.24 m:dgc:384:0.41
         m:cobit:512:0.26 m:cobit:512:0.41 m:dgc:512:0.3 m:dgc:512:0.41"
    for OWT_SEED in 43 44; do
      for pt in $PTS; do
        IFS=: read -r m arm nfe g <<<"$pt"
        owt "$m" "$arm" "$([ "$m" = s ] && echo 00:45:00 || echo 01:00:00)" "" "$nfe:$g"
      done
    done
    OWT_SEED=42
    ;;
  t1)
    # Everything at T = 1 (no score temperature), best-vs-best churn per grid (21 Oct, approved):
    # TinyGSM churn re-tuned for DGC (the CoBit-S standard-grid cells at churn .1/.2/.3 are stored),
    # Sudoku gamma swept on both grids, LM1B fronts on both grids in this pipeline (after lm1b-gate).
    for g in 0.1 0.2 0.3; do GSM_G=$g gsm s500 dgc 1 1024 1319 01:00:00; done
    for arm in cobit dgc; do for g in 0.1 0.2 0.3; do GSM_G=$g gsm m250 "$arm" 1 1024 1319 02:00:00; done; done
    [ -n "${T1_NO_SUDOKU:-}" ] || for d in easy medium hard; do sudoku "$d" "0.1 0.2 0.3 0.5" "42 43 44" 01:00:00; done
    ;;
  lm1b-gate)
    for arm in cobit dgc; do lm1b "$arm" 00:30:00 "256:0.2" 64; done
    ;;
  lm1b-sweep)
    for arm in cobit dgc; do for nfe in 128 256 384 512; do
      cells=""; for g in $SWEEP_G; do cells+="$nfe:$g/"; done
      lm1b "$arm" 01:00:00 "${cells%/}"
    done; done
    ;;
  dp-tasks)
    # pass@1 at low NFE: standard vs continuous DGC vs finite-NFE DP-DGC (K=8, notebook G_k), T = 1.
    # GSM8K: CoBit-S unguided 500k and CoBit-M 250k, churn 0.41, 1319 problems, seed 0 (s500 256-step standard and
    # continuous cells exist). Sudoku hard: gamma 0.5 (both grids' best), seeds 42-44, one job per step count
    # (the 180-step standard/continuous cells exist and are skipped inside the job).
    for st in 64 128 256; do for arm in cobit dgc dpdgc; do
      gsm s500 "$arm" 1 "$st" 1319 00:45:00
      gsm m250 "$arm" 1 "$st" 1319 01:00:00
    done; done
    for st in 32 64 90 180; do sudoku hard "0.5" "42 43 44" 00:45:00 "SU_ARMS=cobit dgc dpdgc,SU_STEPS=$st"; done
    ;;
  dp-smoke)
    # DP-DGC smoke: CoBit-M OWT, 64 samples, NFE 32 and 256, one job
    owt m dp_dgc 00:30:00 64 "32:0.2/256:0.2"
    ;;
  dp-owt)
    # Finite-NFE DP-DGC vs continuous DGC vs standard grid, CoBit-M OWT, seed 42, fixed churn gamma 0.2 for
    # every grid and NFE (the existing 256/384/512 standard and DGC cells at gamma 0.2 are reused).
    for nfe in 32 64 128 256 384 512; do for arm in cobit dgc dp_dgc; do   # DP_W=length_squared: Part 3
      owt m "$arm" 01:00:00 "" "$nfe:0.2"
    done; done
    ;;
  sweep-owt)
    # Pareto fronts: the same gamma grid on BOTH grids, same pipeline and seed (42), NFE 256/384/512.
    # One job per cell (a comma-separated CELLS list does not survive sbatch --export).
    for arm in cobit dgc; do
      for nfe in 256 384 512; do
        for g in $SWEEP_G; do
          owt s "$arm" 00:45:00 "" "$nfe:$g"
          owt m "$arm" 01:00:00 "" "$nfe:$g"
        done
      done
    done
    ;;
  gate|gate-owt|main|main-tasks|main-owt|tgsm-*|mechanism|seeds-owt|t1|lm1b-*|dp-*) ;;
  *) echo "usage: $0 gate|gate-owt|main|main-tasks|main-owt|sweep-owt [--go]" >&2; exit 1 ;;
esac
echo "$N jobs $([ $GO -eq 1 ] && echo submitted || echo '(dry run)')"
