#!/usr/bin/env bash
# Full-dev BIRD-for-DuckDB exec eval of ONE model, sharded over all 4 cores, then merged
# back into the unsharded row order (row-aligned for eval_sql --compare --key exec_ok).
# Runs ON THE BOX. Gates on markers.
#
#   labs/run_bird_eval_sharded.sh <tag> [adapter r alpha targets]
#   labs/run_bird_eval_sharded.sh bird2k_base
#   labs/run_bird_eval_sharded.sh bird2k_ft_epoch1 /work/runs/ft_epoch1/adapter_final.pt 16 32 all
#
# Precondition: the prompt-length buckets are compiled (lab10 --warmup-only, run once,
# alone). Concurrent first compiles of one bucket race on the NKI kernel cache.
# seq 2048: lab11 -- at seq 1024 49.9% of dev prompts lose their schema head.
#
# Env overrides (defaults = the full-dev plain-format protocol):
#   LIMIT=2000 SEQ=2048 MAX_NEW=128 FORMAT=plain
#   thinking-mode baseline on the 128 subset:
#   LIMIT=128 SEQ=4096 MAX_NEW=2048 FORMAT=chat-think labs/run_bird_eval_sharded.sh think128_9b_base
set -uo pipefail
LIMIT="${LIMIT:-2000}" SEQ="${SEQ:-2048}" MAX_NEW="${MAX_NEW:-128}" FORMAT="${FORMAT:-plain}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
C="$SRC/env/container.sh"
RUNS="$(cd "$SRC/.." && pwd)/runs"
ev="$1" adapter="${2:-}" r="${3:-16}" alpha="${4:-32}" targets="${5:-all}"
N=4

extra=()
[[ -n "$adapter" ]] && extra=(--adapter "$adapter" --lora-r "$r" --lora-alpha "$alpha"
                              --lora-targets "$targets")

echo "==================== $ev eval ($N shards, $FORMAT, limit $LIMIT, seq $SEQ, max_new $MAX_NEW)  $(date -u +%H:%M) ===================="
for ((k = 0; k < N; k++)); do
  CORES="$k" NAME="eval_${ev}_$k" "$C" exec python3 labs/lab10_bird_exec_eval.py \
    --limit "$LIMIT" --seq "$SEQ" --max-new "$MAX_NEW" --format "$FORMAT" \
    --shard "$k" --num-shards "$N" --tag "$ev" \
    "${extra[@]}" > "$RUNS/${ev}_s${k}of${N}.log" 2>&1 &
  sleep 20   # stagger the model loads (host RAM / disk), not the compiles
done
wait
for ((k = 0; k < N; k++)); do
  grep -q "BIRD EXEC EVAL OK" "$RUNS/${ev}_s${k}of${N}.log" \
    || { echo "$ev shard $k FAILED"; tail -5 "$RUNS/${ev}_s${k}of${N}.log"; exit 1; }
done
TORCH_DEVICE_BACKEND_AUTOLOAD=0 NAME="merge_$ev" "$C" exec python3 \
  labs/lab10_bird_exec_eval.py --merge-shards "$N" --limit "$LIMIT" --seq "$SEQ" \
  --max-new "$MAX_NEW" --format "$FORMAT" --tag "$ev" \
  "${extra[@]}" > "$RUNS/$ev.log" 2>&1
grep -q "BIRD EXEC EVAL OK" "$RUNS/$ev.log" || { echo "$ev MERGE FAILED"; exit 1; }
grep -E "^n=|prompt-truncated" "$RUNS/$ev.log"
echo "$ev DONE $(date -u +%H:%M)"
