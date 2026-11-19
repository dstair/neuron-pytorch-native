#!/usr/bin/env bash
# Fine-tune on BIRD-train-for-DuckDB, then eval on full BIRD dev (seq 2048) sharded over
# all 4 cores (run_bird_eval_sharded.sh). Runs ON THE BOX. Gate on markers, never exit codes.
#
#   nohup labs/run_bird_ft.sh <tag> <steps> <lora_r> <lora_alpha> [extra lab07 args...] \
#       > ../runs/<tag>.status 2>&1 &
#   labs/run_bird_ft.sh bird_r16 457 16 32
#   SQLFT_MODEL=/work/models/Qwen3.5-9B labs/run_bird_ft.sh bird9b_r16 457 16 32 --strategy fsdp2
#     (9B: 18 GiB of weights do not fit DDP's per-core replica; FSDP2 = 4.2 GiB/core,
#      15.9 GiB peak, 16.0 s/step. SQLFT_MODEL reaches the eval too via container.sh.)
#
# Shape choices (lab11/lab12, probe runs 2026-11-09):
#   seq 1664  the longest trainable row is 1,649 tokens; the 4 databases whose DDL alone
#             exceeds ~2k tokens (works_cycles, mondial_geo, hockey, professional_
#             basketball) drop at any seq <= 2048, so 1664 keeps exactly the seq-2048 rows
#             (7,325) with less padding.
#   --grad-ckpt  seq 1664 OOMs in backward without it (NRT_RESOURCE); with it the peak is
#             13.8 GiB/core at 3.0 s per micro-batch.
#   457 steps = 1 epoch: 7,325 rows / 4 ranks = 1,831 micro-batches/rank / grad-accum 4.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
C="$SRC/env/container.sh"
RUNS="$(cd "$SRC/.." && pwd)/runs"
TORCHRUN="torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29581"
tag="$1" steps="$2" r="$3" alpha="$4"; shift 4
extra=("$@")

echo "==================== $tag train  $(date -u +%H:%M) ===================="
LNC=2 WORLD=4 NAME="train_$tag" "$C" exec $TORCHRUN /work/src/labs/lab07_finetune.py \
  --tag "$tag" --data /work/bird/bird_train_train.jsonl \
  --seq 1664 --grad-ckpt --steps "$steps" --save-every 200 --log-every 10 \
  --lora-r "$r" --lora-alpha "$alpha" "${extra[@]}" \
  > "$RUNS/$tag.log" 2>&1
grep -q "FINETUNE OK" "$RUNS/$tag.log" || { echo "$tag TRAIN FAILED"; tail -5 "$RUNS/$tag.log"; exit 1; }
tail -3 "$RUNS/$tag.log" | head -2

targets=all
for ((i = 0; i < ${#extra[@]}; i++)); do
  [[ "${extra[$i]}" == --lora-targets ]] && targets="${extra[$((i + 1))]}"
done

"$HERE/run_bird_eval_sharded.sh" "bird2k_$tag" "/work/runs/$tag/adapter_final.pt" \
  "$r" "$alpha" "$targets" || exit 1
echo "$tag DONE $(date -u +%H:%M)"
