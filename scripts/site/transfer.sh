#!/bin/bash
# Move the non-git payload to another cluster: the tokenised corpus and the
# checkpoints worth keeping. Code travels through git, not this script.
#
#   bash scripts/site/transfer.sh                      # dry run, prints sizes
#   bash scripts/site/transfer.sh --dest user@isambard:/projects/xxx/BitstreamDiffusion
#   bash scripts/site/transfer.sh --dest ... --go
#
# rsync is resumable (-P) and re-runnable: interrupt it and run the same command
# again. It verifies by size+mtime; --checksum is available but reads 25 GB
# twice, so the script instead records sha256 for the files where a silent
# truncation would be expensive, and the far side checks them.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"

DEST=""; GO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --go)   GO=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

# WHAT MOVES, and why each item earns its bytes.
#
#   datasets/tinygsm/*capall*   the tokenised corpus. Rebuilding it needs the
#                               HuggingFace download and ~2 h of tokenising, and
#                               compute nodes are usually offline.
#   tinigsm_gsm8k/.../last.pt   the production anchor. This is the ONLY way to
#                               prove the port decodes correctly before the
#                               grant is spent -- a sampler bug once scored
#                               0.000 against a true 0.164 and was caught only
#                               because an anchor was in the batch.
#   entropy_*.pt, sigma_data    the anchor's own schedule tables; without
#                               sigma_data.json the evaluator falls back to a
#                               config default of 0.5 against a true 0.3998.
#   ord_*/checkpoints/last.pt   the four ordering arms, so r2l/random can
#                               continue rather than restart.
#   tok_b512_*/last.pt          the V-way run in flight.
#   results/                    the measured numbers, so analysis runs there.
ITEMS=(
  "datasets/tinygsm/tinygsm_bs512_answeronly_trainonpad_filtered_HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42.meta.json"
  "datasets/tinygsm/tinygsm_bs512_answeronly_trainonpad_filtered_HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42_train_ids.uint16"
  "datasets/tinygsm/tinygsm_bs512_answeronly_trainonpad_filtered_HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42_train_plen.int32"
  "datasets/tinygsm/tinygsm_bs512_answeronly_trainonpad_filtered_HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42_validation_ids.uint16"
  "datasets/tinygsm/tinygsm_bs512_answeronly_trainonpad_filtered_HuggingFaceTB__SmolLM-135M_capall_vr0.01_vs42_validation_plen.int32"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/checkpoints/last.pt"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/sigma_data.json"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/config.json"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/entropy_cdf.pt"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/entropy_pdf.pt"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/entropy_edges.pt"
  "tinigsm_gsm8k/runs/cobit_raw_binary_bits_cfg/entropy_sigmas.pt"
  "runs/tasks/tinygsm/ord_fs500k_none_s42/checkpoints/last.pt"
  "runs/tasks/tinygsm/ord_fs500k_none_s42/sigma_data.json"
  "runs/tasks/tinygsm/ord_fs500k_l2r_w0.25_s42/checkpoints/last.pt"
  "runs/tasks/tinygsm/ord_fs500k_l2r_w0.25_s42/sigma_data.json"
  "runs/tasks/tinygsm/ord_fs500k_r2l_w0.25_s42/checkpoints/last.pt"
  "runs/tasks/tinygsm/ord_fs500k_r2l_w0.25_s42/sigma_data.json"
  "runs/tasks/tinygsm/ord_fs500k_random_w0.25_s42/checkpoints/last.pt"
  "runs/tasks/tinygsm/ord_fs500k_random_w0.25_s42/sigma_data.json"
  "runs/tasks/tinygsm/tok_b512_token_ce_s42/checkpoints/last.pt"
  "runs/tasks/tinygsm/tok_b512_token_ce_s42/sigma_data.json"
  "results"
)

MANIFEST="scripts/site/transfer_manifest.txt"
: > "$MANIFEST"
total=0; missing=0
printf "%-68s %10s\n" "PATH" "SIZE"
shorten () {  # keep the tail, which is the distinguishing part of these paths
  local p="$1"
  if [ ${#p} -gt 68 ]; then printf '...%s' "${p:$(( ${#p} - 65 ))}"; else printf '%s' "$p"; fi
}
for it in "${ITEMS[@]}"; do
  if [ ! -e "$it" ]; then
    printf "%-68s %10s\n" "$(shorten "$it")" "MISSING"; missing=$((missing+1)); continue
  fi
  b=$(du -sb "$it" | cut -f1); total=$((total+b))
  printf "%-68s %9.2fG\n" "$(shorten "$it")" "$(echo "$b" | awk '{print $1/1073741824}')"
  echo "$it" >> "$MANIFEST"
done
echo
printf "total: %.1f GB in %d items (%d missing)\n" "$(echo $total | awk '{print $1/1073741824}')" "${#ITEMS[@]}" "$missing"

# PREFLIGHT. Reach the far side BEFORE hashing 24 GB: the first attempt spent
# minutes on sha256 and then died on "Could not resolve hostname login45".
#
# Try SSH FIRST, not DNS. A destination may legitimately be an alias defined in
# ~/.ssh/config with its own HostName, ProxyJump or ProxyCommand -- `getent
# hosts` cannot see any of that, so a DNS-first check rejects a setup that
# works. DNS is used only to explain a failure, never to gate one.
if [ -n "$DEST" ]; then
  sshtarget="${DEST%%:*}"                       # user@host or an alias
  host="${sshtarget##*@}"
  if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=15 "$sshtarget" true >/dev/null 2>&1; then
    echo "[transfer] preflight ok: $sshtarget reachable"
  else
    echo "[transfer] cannot reach: $sshtarget" >&2
    if ssh -G "$host" 2>/dev/null | grep -qiE "^hostname ${host}$" && \
       ! getent hosts "$host" >/dev/null 2>&1; then
      cat >&2 <<MSG

  "$host" is not in DNS and has no ~/.ssh/config entry, so ssh has nothing to
  connect to. On Isambard-AI the portal gives you a config stanza to paste --
  names like "login45" are ALIASES it defines, not public hostnames. Create
  ~/.ssh/config (it does not exist on this node) with what the portal gives
  you, then re-run this command unchanged:

    Host login45
        HostName   <what the portal says>
        User       <your user.project.system.isambard name>
        ProxyJump  <if the portal specifies one>
        IdentityFile ~/.ssh/<the key you registered>

MSG
    else
      cat >&2 <<MSG

  The name resolves or is configured, but SSH did not connect. Usual causes:
  the key is not registered with the site, the certificate has expired, or the
  site does not accept inbound connections from here.

MSG
    fi
    cat >&2 <<MSG
  Either way, the PULL direction avoids all of it -- most sites allow outbound
  but not inbound. Clone the repo on Isambard, then run THERE:

    rsync -avhP --files-from=scripts/site/transfer_manifest.txt \\
          ${USER}@login.hpc.cam.ac.uk:${PWD}/ ./

MSG
    exit 1
  fi
fi

# Checksums for the files where a truncated copy would cost a GPU-hour to
# discover. The corpus is excluded: 11 GB of hashing to protect a file rsync
# already size-checks, and a corrupt corpus shows up immediately as garbage loss.
SUMS="scripts/site/transfer_checksums.txt"
stale=0
if [ -f "$SUMS" ]; then
  for it in "${ITEMS[@]}"; do
    case "$it" in *.pt) [ -f "$it" ] && [ "$it" -nt "$SUMS" ] && stale=1 ;; esac
  done
else
  stale=1
fi
if { [ $GO -eq 1 ] || [ -n "$DEST" ]; } && [ $stale -eq 0 ]; then
  echo "[transfer] reusing $SUMS (checkpoints unchanged since it was written)"
elif [ $GO -eq 1 ] || [ -n "$DEST" ]; then
  echo "[transfer] hashing checkpoints (a few minutes)..."
  : > "$SUMS"
  for it in "${ITEMS[@]}"; do
    case "$it" in *.pt) [ -f "$it" ] && sha256sum "$it" >> "$SUMS" ;; esac
  done
  echo "[transfer] wrote $SUMS ($(wc -l < "$SUMS") files)"
fi

if [ -z "$DEST" ]; then
  echo; echo "dry run. Re-run with --dest user@host:/path [--go]"; exit 0
fi
if [ $GO -eq 0 ]; then
  echo; echo "would run:"
  echo "  rsync -avhP --files-from=$MANIFEST ./ $DEST/"
  echo "Re-run with --go to transfer."; exit 0
fi

rsync -avhP --files-from="$MANIFEST" ./ "$DEST/"
rsync -avhP "$SUMS" "$DEST/scripts/site/"
echo
echo "[transfer] done. On the far side, verify before using any of it:"
echo "  cd <project> && sha256sum -c scripts/site/transfer_checksums.txt"
