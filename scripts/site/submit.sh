#!/bin/bash
# Submit a launcher with this site's partition/account/GPU flags applied.
#
#   bash scripts/site/submit.sh scripts/hpc/arch/ordering_train.slurm \
#        --time=24:00:00 --export=ALL,ORD_MODE=r2l,ORD_W=0.25
#
# Every launcher in scripts/hpc/ carries CSD3 #SBATCH directives. SLURM's
# precedence is command line > in-script directive > environment, so the flags
# this wrapper passes win without any launcher being edited. Submitting a
# CSD3 launcher bare on another cluster asks for partition "ampere" and account
# "GIROLAMI-SL2-GPU", which on Isambard is a job that never schedules.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PROJECT_DIR/scripts/site/site.sh"
cd "$PROJECT_DIR"

SCRIPT="${1:?usage: submit.sh <launcher.slurm> [extra sbatch args...]}"
shift
[ -f "$SCRIPT" ] || { echo "[submit] no such launcher: $SCRIPT" >&2; exit 1; }

# A wall request over the site cap is rejected at submit time on some clusters
# and silently truncated on others; refuse it here where the message is useful.
for a in "$@"; do
  case "$a" in
    --time=*)
      want="${a#--time=}"
      to_s () { awk -F: '{n=NF; s=0; for(i=1;i<=n;i++) s=s*60+$i; if (n==4) s=s; print s}' <<<"${1//-/:}"; }
      if [ "$(to_s "$want")" -gt "$(to_s "$COBIT_WALL_MAX")" ]; then
        echo "[submit] --time=$want exceeds this site's cap $COBIT_WALL_MAX" >&2; exit 1
      fi ;;
  esac
done

set -x
sbatch --partition="$COBIT_PARTITION" --account="$COBIT_ACCOUNT" \
       --gres="$COBIT_GRES" --cpus-per-task="$COBIT_CPUS_PER_TASK" \
       "$@" "$SCRIPT"
