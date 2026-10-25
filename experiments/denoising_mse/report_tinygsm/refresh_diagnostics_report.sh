#!/usr/bin/env bash
# Rebuild the diagnostics report when (1) CoBit-M's staggered-noise runs and (2) the whole diagnostics chain finish.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PDFPY=/tmp/claude-1003/-home-rg625-mnt-BitstreamDiffusion/b096fb3c-3ad7-4cf3-80c8-0d6a7fd0193d/scratchpad/pdfenv/bin/python
L="$REPO/experiments/denoising_mse/logs/tg_diag.log"
R="$REPO/experiments/denoising_mse/logs/tg_diag_report.log"
RAW="$REPO/experiments/denoising_mse/results_tinygsm/diagnostics/raw"
source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate sedd
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
rebuild() {
  echo "=== rebuild ($1) $(date -Is)" >> "$R"
  (cd "$REPO/experiments/denoising_mse/tinygsm" && CUDA_VISIBLE_DEVICES="" python -W ignore analyze_diagnostics.py >> "$R" 2>&1)
  (cd "$REPO/experiments/denoising_mse/report_tinygsm" && "$PDFPY" build_diagnostics_pdf.py >> "$R" 2>&1)
  echo "=== rebuild ($1) exit $? $(date -Is)" >> "$R"
}
until [ -f "$RAW/M_stagger0.2_seed1.json" ] || grep -q "=== diag done" "$L"; do sleep 120; done
rebuild "CoBit-M staggered noise done"
for N in 8 16; do
  until [ -f "$RAW/M_base_seed$((N - 1)).json" ] || grep -q "=== diag done" "$L"; do sleep 300; done
  rebuild "CoBit-M Maj@$N available"
done
until grep -q "=== diag done" "$L"; do sleep 300; done
rebuild "all diagnostics done"
echo "=== report refresh finished $(date -Is)" >> "$R"
