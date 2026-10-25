#!/bin/bash
# Wait for the CoBit-M TinyGSM run to reach 250k on Isambard, pull the
# checkpoint and its run-directory files to this machine, push them to cambox,
# verify sha256 at every hop, then drop UPLOAD_COMPLETE(.md) next to them.
#
# Run on YOUR machine -- the one where both `ssh cambox` and ssh to Isambard
# work (Isambard cannot reach cambox):
#
#   ISAMBARD=<your ssh alias for Isambard> bash fetch_cobit_m_250k.sh
#
# Re-runnable: rsync -P resumes partial copies, and finished hops are re-checked
# by hash rather than re-sent. Polls Isambard every POLL seconds (default 600)
# until the training log says the run stopped at 250k; exits non-zero if the
# final job ends any other way.
#
#   source  Isambard  /home/u6ya/raricu.u6ya/BitstreamDiffusion/runs/tasks/tinygsm/cobit_m_bits_s42
#   ckpt -> cambox    ~/mnt/BitstreamDiffusion/assets/tinyGSM/medium/step=000250000.pt
#   run  -> cambox    ~/mnt/BitstreamDiffusion/runs/tasks/tinygsm/cobit_m_raw_binary_bits/
set -euo pipefail
ISAMBARD="${ISAMBARD:?set ISAMBARD to the ssh host you use for Isambard}"
CAMBOX="${CAMBOX:-cambox}"
POLL="${POLL:-600}"
STAGE="${STAGE:-$HOME/cobit_m_250k_stage}"

SRC=/home/u6ya/raricu.u6ya/BitstreamDiffusion
RUN=$SRC/runs/tasks/tinygsm/cobit_m_bits_s42
JOB=7048128                                   # last job of the 3-job chain
LOG=$SRC/logs/arch/cobit_m_tinygsm_$JOB.log
CKPT=step=000250000.pt
FILES=(config.json original_config.py entropy_cdf.pt entropy_edges.pt entropy_pdf.pt
       entropy_sigmas.pt sigma_data.json)
DEST_CKPT=mnt/BitstreamDiffusion/assets/tinyGSM/medium          # relative to cambox ~
DEST_RUN=mnt/BitstreamDiffusion/runs/tasks/tinygsm/cobit_m_raw_binary_bits

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
sha () { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
mkdir -p "$STAGE"; cd "$STAGE"

# 1. Wait for the run to stop at 250k. The trainer prints this line only after
#    the final validation and last.pt save, so every file below is final.
say "waiting for $JOB to reach 250k (poll ${POLL}s)"
done_250k () {
  ssh -o BatchMode=yes "$ISAMBARD" \
    "grep -aq 'Reached target total_steps=250000' $LOG 2>/dev/null && test -f $RUN/checkpoints/$CKPT"
}
until done_250k; do
  state=$(ssh -o BatchMode=yes "$ISAMBARD" "sacct -j $JOB -X -n -o State" 2>/dev/null | tr -d ' ' || true)
  case "$state" in
    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|COMPLETED*)
      # The job may have finished between the two ssh calls: look once more.
      # COMPLETED without the line means the trainer exited early (rc is
      # swallowed by the launcher); either way 250k is not there.
      done_250k && break
      say "job $JOB ended '$state' without reaching 250k -- check $LOG"; exit 1 ;;
  esac
  say "not yet (job state: ${state:-unknown})"; sleep "$POLL"
done
say "250k reached"

# 2. Hashes at the source (sha256 of 7.4 GB: under a minute on the login node).
ssh -o BatchMode=yes "$ISAMBARD" "cd $RUN && sha256sum checkpoints/$CKPT ${FILES[*]}" \
  | sed 's#checkpoints/##' > SHA256SUMS
say "source hashes:"; cat SHA256SUMS

# 3. Isambard -> here.
rsync -aP "$ISAMBARD:$RUN/checkpoints/$CKPT" .
rsync -aP "${FILES[@]/#/$ISAMBARD:$RUN/}" .
sha -c SHA256SUMS
say "local copy verified"

# 4. Handoff note for whoever evaluates it on cambox.
cat > UPLOAD_COMPLETE.md <<EOF
# CoBit-M TinyGSM, step 250,000 -- ready

Uploaded $(date -u '+%Y-%m-%d %H:%M UTC') from Isambard run tasks/tinygsm/cobit_m_bits_s42.

- Checkpoint: ~/$DEST_CKPT/$CKPT (full training checkpoint: model + EMA + optimiser,
  7.4 GB). Evaluate with EMA (the default apply_ema=True), as for CoBit-S.
- Run files (this directory): ${FILES[*]}. sha256 for everything in SHA256SUMS.
- sigma_data (sidecar): $(tr -d '\n' < sigma_data.json). The CoBit-S sidecar
  value is 0.399844765663147; they agree to 7 significant figures.

Model: CoBit-M, 462.1M params. 24 blocks x 1024 wide, 16 heads, dim_ff 4096.
Build it from THIS config.json: a CoBit-S (12 x 768) config will not load
the weights.

Training recipe:
- Data, conditioning, validation, diffusion, loss, entropy schedule, EMA 0.9999,
  batch 512, seed 42: identical to the unguided CoBit-S TinyGSM config
  (configs/tasks/tinygsm_bits.py, p_uncond = 0).
- Optimiser, from CoBit-M OWT: AdamW lr 2e-4, cosine over 500k with 5k warmup,
  weight decay 0.01, betas (0.9, 0.99).
- Stopped at 250k, the middle of the cosine (LR ~1.0e-4, not annealed).
- Data-order seed 0 (the June CoBit-S order). 4 x GH200, bf16.

Comparing with CoBit-S: this is a RECIPE comparison, not a pure scale one.
- CoBit-S used lr 3e-4 constant, weight decay 0, beta2 0.999.
- This run also has the T1/T3 trainer fixes (every GPU uses the same
  sigma_data; noise differs per GPU), which CoBit-S trained without.
- The matched step is 250k (CoBit-S unguided step=000250000.pt).

Paths inside config.json still name the Isambard run (experiment
tasks/tinygsm/cobit_m_bits_s42, evaluation.checkpoint_path under runs/...).
Point the evaluator at the checkpoint path above explicitly.
- original_config.py loads its parent configs by relative path, so on its own
  it is not self-contained; config.json is the authoritative flat config.
- Keys stop_at_step, data_order_seed and checkpointing.interval.at_steps are
  training-only.
EOF

# 5. Here -> cambox, then verify there. The marker goes last so its presence
#    means everything else is complete and checked.
ssh "$CAMBOX" "mkdir -p ~/$DEST_CKPT ~/$DEST_RUN && rm -f ~/$DEST_RUN/UPLOAD_COMPLETE"
rsync -aP "$CKPT" "$CAMBOX:$DEST_CKPT/"
rsync -aP "${FILES[@]}" SHA256SUMS UPLOAD_COMPLETE.md "$CAMBOX:$DEST_RUN/"
ssh "$CAMBOX" "cd ~/$DEST_RUN && grep -v ' $CKPT\$' SHA256SUMS | sha256sum -c - \
  && grep ' $CKPT\$' SHA256SUMS | (cd ~/$DEST_CKPT && sha256sum -c -) \
  && touch UPLOAD_COMPLETE"
say "cambox copy verified; marker ~/$DEST_RUN/UPLOAD_COMPLETE written"
say "local staging copy kept in $STAGE (delete when done)"
