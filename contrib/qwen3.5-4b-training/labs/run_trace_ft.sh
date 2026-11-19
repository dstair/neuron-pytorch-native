#!/usr/bin/env bash
# Reasoning-trace fine-tune of Qwen3.5-9B on Tier A (lab15: gpt-oss-120b traces, unedited,
# no hints, SQL result == verified gold), chat format with <think>, then thinking-mode
# evals. Same LoRA recipe as bird9b_r16 (r16 a32 all-targets, lr 1e-4, FSDP2, grad ckpt);
# only the data/format and seq differ. Runs ON THE BOX. Gates on markers.
#
#   nohup labs/run_trace_ft.sh > ../runs/trace9b.status 2>&1 &
#
# seq 2048 keeps 4,429 / 5,068 tier-A rows (87.4%; lab15). 4,429 / 4 ranks / accum 4 =
# 276 optimizer steps = 1 epoch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
C="$SRC/env/container.sh"
RUNS="$(cd "$SRC/.." && pwd)/runs"
TORCHRUN="torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29581"
export SQLFT_MODEL=/work/models/Qwen3.5-9B
TAG=trace9b_tierA DATA=/work/bird/bird_train_traces_tierA.jsonl SEQ=2048 STEPS="${STEPS:-276}"

# 0. wait for the thinking-mode baseline queue to release the cores
# (independent of its outcome: a failed baseline must not block training)
until grep -q "THINK BASELINE END" "$RUNS/think_long.status" 2>/dev/null; do sleep 120; done

train() {  # tag steps extra...
  local t="$1" s="$2"; shift 2
  LNC=2 WORLD=4 NAME="train_$t" "$C" exec $TORCHRUN /work/src/labs/lab07_finetune.py \
    --tag "$t" --data "$DATA" --seq "$SEQ" --grad-ckpt --steps "$s" \
    --save-every 100 --log-every 10 --lora-r 16 --lora-alpha 32 --strategy fsdp2 "$@" \
    > "$RUNS/$t.log" 2>&1
  grep -q "FINETUNE OK" "$RUNS/$t.log" || { echo "$t TRAIN FAILED"; tail -8 "$RUNS/$t.log"; exit 1; }
}

echo "==================== probe seq $SEQ  $(date -u +%H:%M) ===================="
train "${TAG}_probe" 4
grep -E "peak|s/step" "$RUNS/${TAG}_probe.log" | tail -2

echo "==================== $TAG train ($STEPS steps)  $(date -u +%H:%M) ===================="
train "$TAG" "$STEPS"
tail -3 "$RUNS/$TAG.log" | head -2

ADAPTER="/work/runs/$TAG/adapter_final.pt"
LIMIT=128 SEQ=4096 MAX_NEW=2048 FORMAT=chat-think \
  "$HERE/run_bird_eval_sharded.sh" "b128_$TAG" "$ADAPTER" 16 32 all || exit 1
LIMIT=2000 SEQ=4096 MAX_NEW=2048 FORMAT=chat-think \
  "$HERE/run_bird_eval_sharded.sh" "bird4k_$TAG" "$ADAPTER" 16 32 all || exit 1
echo "$TAG DONE $(date -u +%H:%M)"
