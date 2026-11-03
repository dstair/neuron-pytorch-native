#!/usr/bin/env bash
# Tensor parallel vs plain FSDP2 for the full fine-tune. Runs ON THE BOX, serially.
#
#   labs/run_lab05_tp.sh [layers] [seq] [steps]
#   ONLY=tp labs/run_lab05_tp.sh ...     # just the configs whose tag contains "tp"
#
# Full-FT only: TP shards the base weights, which only matters when they are trainable.
# lr 1e-5 because full-FT at the LoRA rate of 1e-4 diverges (NOTES.md).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"

LAYERS="${1:-32}"
SEQ="${2:-512}"
STEPS="${3:-6}"
GDN="${GDN:-nki-v15-split16}"
STACK="${STACK:-public}"
WORLD=4
LNC=2

run_one() {
  local tag="$1"; shift
  local log="$LOGS/$tag.log"
  # ONLY=<substring> re-runs a subset, e.g. ONLY=tp after a fix that cannot affect plain FSDP2.
  [[ -n "${ONLY:-}" && "$tag" != *"$ONLY"* ]] && return 0
  # Every config here is FSDP-based, so all need the host collective path (NOTES.md).
  echo "=== $tag (stack=$STACK world=$WORLD LNC=$LNC) -> $log"
  STACK="$STACK" LNC="$LNC" WORLD="$WORLD" HOST_CC=1 "$SRC/env/container.sh" exec \
    torchrun --nproc_per_node "$WORLD" --rdzv_backend c10d --rdzv_endpoint "localhost:2955$WORLD" \
    labs/lab04_ddp_vs_fsdp.py --layers "$LAYERS" --seq "$SEQ" --steps "$STEPS" \
    --gdn "$GDN" --full-ft --lr 1e-5 --tag "$tag" "$@" > "$log" 2>&1
  # Gate on the marker, not the exit code: these containers SIGABRT at teardown after
  # printing perfectly good results.
  if grep -q "LAB04 OK" "$log"; then
    echo "    OK  $(grep -E 'agg tok/s' "$log" | tail -1)"
  else
    echo "    FAILED -- last lines:"
    tail -6 "$log" | sed 's/^/      /'
  fi
}

run_one "lab05_${STACK}_fsdp2_fullft_w4"      --strategy fsdp2
run_one "lab05_${STACK}_fsdp2_tp2_fullft_w4"  --strategy fsdp2_tp --tp 2
run_one "lab05_${STACK}_fsdp2_tp4_fullft_w4"  --strategy fsdp2_tp --tp 4

echo "=== lab05 done"
