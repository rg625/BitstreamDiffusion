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
# Exported so the job sees it: every launcher falls back to the CSD3 path when
# PROJECT_DIR is unset, and sbatch's --export=ALL carries only exported vars.
export PROJECT_DIR
# Banner to stderr: callers capture stdout as the job id (sbatch --parsable), and
# the banner in front of it broke every --dependency=afterany:<id> in a chain.
source "$PROJECT_DIR/scripts/site/site.sh" >&2
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

# GPUs: the launcher's own `#SBATCH --gres=gpu:N`, not a site-wide constant.
# Forcing gpu:4 on the 1-GPU eval launchers reserved (and billed) four GPUs
# per eval. CPUs scale with it: Isambard allocates one 72-core socket per GPU
# (DefCpuPerGPU=72), so a flat 72 gave a 4-GPU job a quarter of its cores.
# Flags passed on this command line win over both.
NGPU=$(sed -nE 's/^#SBATCH[[:space:]]+--gres=gpu:([0-9]+).*/\1/p' "$SCRIPT" | head -1)
NGPU="${NGPU:-$COBIT_GPUS_PER_NODE}"
RES=(--gres="gpu:$NGPU" --cpus-per-task="$((NGPU * COBIT_CPUS_PER_GPU))")
for a in "$@"; do
  case "$a" in --gres=*|--gpus*|--cpus-per-task=*) RES=() ;; esac
done

# SLURM does not create the --output directory; a missing one fails the job
# before it writes a single line, so there is no log to say why. The transfer
# does not carry logs/, so on a fresh site every launcher hit this.
OUTDIR=$(sed -nE 's/^#SBATCH[[:space:]]+--output=(.*)\/[^/]*$/\1/p' "$SCRIPT" | head -1)
for a in "$@"; do case "$a" in --output=*/*|-o*/*) OUTDIR="$(dirname "${a#*=}")" ;; esac; done
[ -n "$OUTDIR" ] && mkdir -p "$OUTDIR"

set -x
sbatch --partition="$COBIT_PARTITION" --account="$COBIT_ACCOUNT" \
       ${RES[@]+"${RES[@]}"} "$@" "$SCRIPT"
