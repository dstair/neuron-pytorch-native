#!/usr/bin/env bash
# Triage: 32-layer full-FT under FSDP2 produces a NaN FORWARD on every rank except rank 0.
#
#   [rank0] step 0 forward loss = 1.0299      <- finite
#   [rank1..3] step 0 forward loss = nan
#   then 426/426 parameters have non-finite grads and grad-norm is nan
#
# The NaN is in the forward, so it is not an optimizer or clipping problem. Three things
# changed at once between the working 8-layer run and this one, so separate them:
#
#   A  32L, LoRA, single core      -> is DEPTH alone survivable? no collectives involved.
#   B  32L, LoRA, DDP w4           -> is it MULTI-RANK? DDP replicates, does not shard.
#   C  32L, LoRA, FSDP2 w4         -> is it SHARDING? same depth, same ranks, params sharded.
#
# If A is finite and C is NaN, the parameter all-gather is the suspect. If B is also NaN,
# it is rank-dependent DATA rather than sharding (each rank seeds its loader by rank).
#
#   labs/run_nan_triage.sh [seq] [steps]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"
SEQ="${1:-512}"; STEPS="${2:-2}"

report() {
  local tag="$1" log="$2"
  local nan_ranks
  nan_ranks=$(grep -cE "forward loss = nan" "$log" 2>/dev/null || echo 0)
  echo "    ranks with NaN forward: $nan_ranks"
  grep -E "forward loss =" "$log" 2>/dev/null | head -4 | sed 's/^/      /'
  if grep -q "LAB04 OK\|TRAIN OK" "$log"; then echo "      => OK"; else
    echo "      => FAILED: $(grep -oE 'RuntimeError.*|Failed to allocate DEVICE memory.*' "$log" | head -1)"
  fi
}

echo "=== A  32L LoRA, single core (depth alone)"
LOG="$LOGS/nan_A_32L_lora_1core.log"
LNC=2 WORLD=1 "$SRC/env/container.sh" exec \
  python3 -m sqlft.train --layers 32 --seq "$SEQ" --bs 1 --steps "$STEPS" --lora \
  --tag nan_A_32L_lora_1core > "$LOG" 2>&1
grep -E "^step |TRAIN OK|RuntimeError" "$LOG" | head -4 | sed 's/^/      /'

echo "=== B  32L LoRA, DDP w4 (multi-rank, replicated)"
LOG="$LOGS/nan_B_32L_lora_ddp.log"
LNC=2 WORLD=4 HOST_CC=0 "$SRC/env/container.sh" exec \
  torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29561 \
  labs/lab04_ddp_vs_fsdp.py --strategy ddp --lora --layers 32 --seq "$SEQ" \
  --steps "$STEPS" --check-finite --tag nan_B_32L_lora_ddp > "$LOG" 2>&1
report nan_B "$LOG"

echo "=== C  32L LoRA, FSDP2 w4 (sharded)"
LOG="$LOGS/nan_C_32L_lora_fsdp2.log"
LNC=2 WORLD=4 HOST_CC=1 "$SRC/env/container.sh" exec \
  torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29562 \
  labs/lab04_ddp_vs_fsdp.py --strategy fsdp2 --lora --layers 32 --seq "$SEQ" \
  --steps "$STEPS" --check-finite --tag nan_C_32L_lora_fsdp2 > "$LOG" 2>&1
report nan_C "$LOG"

echo "=== triage done"
