#!/usr/bin/env bash
# The DDP-vs-FSDP regime matrix. Runs ON THE BOX, serially (one device).
#
#   labs/run_lab04_matrix.sh [world] [layers] [seq] [steps]
#
# Serial on purpose: all configs share one Trainium chip, and running two at once fights
# over cores AND over host RAM during neuronx-cc.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"

WORLD="${1:-4}"
LAYERS="${2:-8}"
SEQ="${3:-512}"
STEPS="${4:-6}"
# nki-v15-split16 is correct AND the fastest build; see NOTES.md. Override with
# GDN=torch to re-measure on the reference path.
GDN="${GDN:-nki-v15-split16}"
# world<=4 fits at LNC=2 (24 GB/core, the proven regime); 8 ranks require LNC=1 (12 GB).
LNC=2; [ "$WORLD" -gt 4 ] && LNC=1

run_one() {
  local tag="$1"; shift
  local log="$LOGS/$tag.log"
  # Any FSDP needs the host collective-communication path on beta6. Without it BOTH FSDP1
  # and FSDP2 die with "Failed to build and load collectives resources for module.neff"
  # even though bare eager all_gather/reduce_scatter pass. DDP does not need it, and it is
  # left off there so the DDP number is not carrying extra machinery.
  local hostcc=0
  case " $* " in *" fsdp"*) hostcc=1 ;; esac
  echo "=== $tag (world=$WORLD LNC=$LNC HOST_CC=$hostcc) -> $log"
  LNC="$LNC" WORLD="$WORLD" HOST_CC="$hostcc" "$SRC/env/container.sh" exec \
    torchrun --nproc_per_node "$WORLD" --rdzv_backend c10d --rdzv_endpoint "localhost:2953$WORLD" \
    labs/lab04_ddp_vs_fsdp.py --layers "$LAYERS" --seq "$SEQ" --steps "$STEPS" \
    --gdn "$GDN" --tag "$tag" "$@" > "$log" 2>&1
  # Gate on the marker, not the exit code: these containers SIGABRT at teardown after
  # printing perfectly good results.
  if grep -q "LAB04 OK" "$log"; then
    echo "    OK  $(grep -E 'agg tok/s' "$log" | tail -1)"
  else
    echo "    FAILED -- last lines:"
    tail -6 "$log" | sed 's/^/      /'
  fi
}

# LoRA: DDP should WIN. FULL_SHARD shards a frozen base it must re-gather every forward,
# while the optimizer state it exists to shard is ~3M params.
run_one "lab04_ddp_lora_w$WORLD"        --strategy ddp   --lora
run_one "lab04_fsdp2_lora_w$WORLD"      --strategy fsdp2 --lora

# Full fine-tune: FSDP is the ENABLER. 47 GiB of state vs 24 GiB/core means single-core
# cannot run this at all; sharding the AdamW moments is what makes it fit.
run_one "lab04_fsdp2_fullft_w$WORLD"    --strategy fsdp2 --full-ft

echo "=== matrix done"
