#!/usr/bin/env bash
# LoRA sweep: train 1 epoch -> motherduck eval (cached decode) -> BIRD exec eval,
# per config, sequentially. Runs ON THE BOX. Gate on markers, never exit codes.
#
#   nohup labs/run_sweep.sh > ../../runs/sweep.log 2>&1 &
#
# Configs (all: 1097 steps = 1 epoch, alpha=2r, lr 1e-4 unless noted, DDP world=4):
#   rank axis   r4 / r8 / r32 / r64        (r16 exists as ft_epoch1)
#   lr axis     3e-5 / 3e-4 at r=16        (1e-4 exists as ft_epoch1)
#   targets     mixers (no MLP) at r=16    (all targets exist as ft_epoch1)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
C="$SRC/env/container.sh"
RUNS="$(cd "$SRC/.." && pwd)/runs"
TORCHRUN="torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29581"

sweep_one() {
  local tag="$1" lora_r="$2" lora_alpha="$3" extra_train="$4" extra_eval="$5"
  echo "==================== $tag  $(date -u +%H:%M) ===================="
  LNC=2 WORLD=4 "$C" exec $TORCHRUN /work/src/labs/lab07_finetune.py \
    --tag "$tag" --steps 1097 --save-every 400 \
    --lora-r "$lora_r" --lora-alpha "$lora_alpha" $extra_train \
    > "$RUNS/${tag}.log" 2>&1
  grep -q "FINETUNE OK" "$RUNS/${tag}.log" || { echo "$tag TRAIN FAILED"; return 1; }
  tail -3 "$RUNS/${tag}.log" | head -2

  "$C" exec python3 labs/lab07_eval_finetuned.py \
    --adapter "/work/runs/$tag/adapter_final.pt" \
    --lora-r "$lora_r" --lora-alpha "$lora_alpha" $extra_eval \
    --limit 128 --max-new 128 --decode cached --tag "${tag}_eval" \
    > "$RUNS/${tag}_eval.log" 2>&1
  grep -q "FINETUNE EVAL OK" "$RUNS/${tag}_eval.log" || { echo "$tag EVAL FAILED"; return 1; }
  grep "^n=" "$RUNS/${tag}_eval.log"

  "$C" exec python3 labs/lab10_bird_exec_eval.py \
    --adapter "/work/runs/$tag/adapter_final.pt" \
    --lora-r "$lora_r" --lora-alpha "$lora_alpha" $extra_eval \
    --limit 128 --tag "${tag}_bird" \
    > "$RUNS/${tag}_bird.log" 2>&1
  grep -q "BIRD EXEC EVAL OK" "$RUNS/${tag}_bird.log" || { echo "$tag BIRD FAILED"; return 1; }
  grep "^n=" "$RUNS/${tag}_bird.log"
}

sweep_one sweep_r4    4   8   ""                  ""
sweep_one sweep_r8    8   16  ""                  ""
sweep_one sweep_r32   32  64  ""                  ""
sweep_one sweep_r64   64  128 ""                  ""
sweep_one sweep_lr3e5 16  32  "--lr 3e-5"         ""
sweep_one sweep_lr3e4 16  32  "--lr 3e-4"         ""
sweep_one sweep_nomlp 16  32  "--lora-targets mixers" "--lora-targets mixers"

# Time-course: the r=8 run's intermediate checkpoints, motherduck eval only.
for step in 400 800; do
  tag="sweep_r8_step${step}"
  echo "==================== $tag (time-course) ===================="
  "$C" exec python3 labs/lab07_eval_finetuned.py \
    --adapter "/work/runs/sweep_r8/adapter_step${step}.pt" \
    --lora-r 8 --lora-alpha 16 \
    --limit 128 --max-new 128 --decode cached --tag "$tag" \
    > "$RUNS/${tag}.log" 2>&1
  grep "^n=" "$RUNS/${tag}.log" || echo "$tag EVAL FAILED"
done

echo "SWEEP DONE $(date -u +%H:%M)"
