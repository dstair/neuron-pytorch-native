#!/usr/bin/env bash
# After the BIRD fine-tune: re-evaluate the reference models under the SAME seq-2048
# full-dev protocol, so every comparison is paired on identical rows. Runs ON THE BOX.
#
#   nohup labs/run_bird_queue.sh > ../runs/bird_queue.status 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$(cd "$HERE/../.." && pwd)/runs"
E="$HERE/run_bird_eval_sharded.sh"

# Wait for the fine-tune + its eval to release the chip.
until grep -q -E "DONE|FAILED" "$RUNS/bird_r16.status" 2>/dev/null; do sleep 60; done

"$E" bird2k_ft_epoch1 /work/runs/ft_epoch1/adapter_final.pt 16 32 all   # motherduck r16
"$E" bird2k_base                                                        # untuned
"$E" bird2k_sweep_r4 /work/runs/sweep_r4/adapter_final.pt 4 8 all       # motherduck r4
echo "QUEUE DONE $(date -u +%H:%M)"
