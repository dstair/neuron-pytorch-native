#!/usr/bin/env bash
# Compile or benchmark Qwen3.5-4B decode on one trn2.3xlarge, in any measured topology.
#
#   deploy/bench_trn2.sh compile <topology>          # build the runtime/NEFF cache
#   deploy/bench_trn2.sh run     <topology> [args]   # timed benchmark
#
# Topologies (defaults reproduce the RESULTS.md rows):
#   tp4       one model, TP=4 across 4 LNC=2 logical cores (torchrun static_decode.py).
#             compile and run are the same command; extra args pass through to
#             static_decode.py, e.g. `--dtype bf16 --batch-size 128`.
#   dp8       8 independent TP=1 replicas, one per LNC=1 physical core, BS=32 each,
#             fused FP8 MLP + fused FP8 head.  <- fastest measured (935.1 tok/s)
#   dp4-lnc2  4 independent TP=1 replicas, one per LNC=2 logical core, BS=64 each,
#             generic FP8 path (fused kernels off, as measured: 513.0 tok/s).
#
# DP topologies: `compile` runs ONE replica (core $QWEN4B_VISIBLE_CORE) into
# $QWEN4B_RUNTIME_TEMPLATE; `run` copies that template once per replica, aligns the
# timed windows with a file barrier, and writes $QWEN4B_RUN_ROOT/<run-id>/aggregate.json.
#
# Required env:  QWEN4B_NATIVE_IMAGE, QWEN4B_MODEL_DIR, and
#                tp4: QWEN4B_CACHE_DIR
#                dp*: QWEN4B_RUNTIME_TEMPLATE (+ QWEN4B_RUN_ROOT for `run`)
# Optional env:  QWEN4B_BATCH_SIZE, QWEN4B_FP8_SCOPE (mlp-head|lm-head|all),
#                QWEN4B_FUSED_FP8_MLP / QWEN4B_FUSED_FP8_HEAD (0|1), QWEN4B_WARMUP,
#                QWEN4B_BENCH_ITERS, QWEN4B_CACHE_DIR, QWEN4B_VISIBLE_CORE, QWEN4B_RUN_ID,
#                QWEN4B_SHM_SIZE.
set -euo pipefail

usage() { sed -n '2,26p' "$0" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
mode="$1"; topology="$2"; shift 2
[[ "$mode" == compile || "$mode" == run ]] || usage

image="${QWEN4B_NATIVE_IMAGE:?set QWEN4B_NATIVE_IMAGE}"
model_dir="${QWEN4B_MODEL_DIR:?set QWEN4B_MODEL_DIR}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Per-topology defaults ────────────────────────────────────────────────────
case "$topology" in
  tp4)      lnc=2; replicas=1; def_shm=32g ;;
  dp8)      lnc=1; replicas=8; def_batch=32; def_fused=1
            [[ "$mode" == compile ]] && def_shm=16g || def_shm=10g ;;
  dp4-lnc2) lnc=2; replicas=4; def_batch=64; def_fused=0; def_shm=20g ;;
  *) echo "error: unknown topology '$topology' (tp4|dp8|dp4-lnc2)" >&2; exit 2 ;;
esac
shm="${QWEN4B_SHM_SIZE:-$def_shm}"
cc_flags="--target trn2 --lnc $lnc --optlevel 1"

# ── TP=4: a single torchrun job ──────────────────────────────────────────────
if [[ "$topology" == tp4 ]]; then
  cache_dir="${QWEN4B_CACHE_DIR:?set QWEN4B_CACHE_DIR}"
  runtime_dir="${QWEN4B_RUNTIME_DIR:-$cache_dir/runtime}"
  mkdir -p "$cache_dir" "$runtime_dir"
  exec docker run --rm --entrypoint /bin/bash \
    --privileged --device=/dev/neuron0 --shm-size="$shm" \
    -v "$model_dir":/models/Qwen3.5-4B:ro \
    -v "$source_dir":/work:ro \
    -v "$cache_dir":/ccache \
    -v "$runtime_dir":/tmp \
    -w /work \
    -e NEURON_LOGICAL_NC_CONFIG=2 \
    -e NEURON_RT_NUM_CORES=4 \
    -e "NEURON_CC_FLAGS=$cc_flags" \
    -e NEURON_COMPILE_CACHE_URL=/ccache \
    "$image" -c \
    'torchrun --nproc-per-node=4 static_decode.py "$@"' bash "$@"
fi

# ── DP: independent TP=1 replicas (static_decode_dp8.py) ─────────────────────
runtime_template="${QWEN4B_RUNTIME_TEMPLATE:?set QWEN4B_RUNTIME_TEMPLATE}"
batch_size="${QWEN4B_BATCH_SIZE:-$def_batch}"
fp8_scope="${QWEN4B_FP8_SCOPE:-mlp-head}"
fused_args=""
[[ "${QWEN4B_FUSED_FP8_MLP:-$def_fused}" == "1" ]] && fused_args+=" --fused-fp8-mlp"
[[ "${QWEN4B_FUSED_FP8_HEAD:-$def_fused}" == "1" ]] && fused_args+=" --fused-fp8-head"

replica_cmd() {  # <replica-id> <replica-count> <warmup> <iters> [extra args]
  local rid="$1" n="$2" warmup="$3" iters="$4"; shift 4
  echo "python3 static_decode_dp8.py \
    --model-path /models/Qwen3.5-4B \
    --batch-size $batch_size \
    --max-seq-len 256 \
    --warmup $warmup \
    --bench-iters $iters \
    --fp8-scope $fp8_scope \
    --lnc $lnc \
    $fused_args \
    --replica-id $rid \
    --replica-count $n $*"
}

common_env=(
  -e NEURON_LOGICAL_NC_CONFIG="$lnc"
  -e NEURON_RT_NUM_CORES=1
  -e "NEURON_CC_FLAGS=$cc_flags"
)

if [[ "$mode" == compile ]]; then
  cache_dir="${QWEN4B_CACHE_DIR:-$runtime_template/ccache}"
  compile_only=""
  [[ "$lnc" == 2 ]] && compile_only="--compile-only"   # as measured for dp4-lnc2
  mkdir -p "$runtime_template" "$cache_dir"
  exec docker run --rm --entrypoint /bin/bash \
    --privileged --device=/dev/neuron0 --shm-size="$shm" \
    -v "$model_dir":/models/Qwen3.5-4B:ro \
    -v "$source_dir":/work:ro \
    -v "$runtime_template":/tmp \
    -v "$cache_dir":/ccache \
    -w /work \
    "${common_env[@]}" \
    -e NEURON_RT_VISIBLE_CORES="${QWEN4B_VISIBLE_CORE:-0}" \
    -e NEURON_COMPILE_CACHE_URL=/ccache \
    "$image" -c "$(replica_cmd 0 1 3 5 $compile_only)"
fi

# mode == run
run_root="${QWEN4B_RUN_ROOT:?set QWEN4B_RUN_ROOT}"
warmup="${QWEN4B_WARMUP:-5}"
bench_iters="${QWEN4B_BENCH_ITERS:-50}"
run_id="${QWEN4B_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$run_root/$run_id"
runtime_root="$run_dir/runtime"
barrier_dir="$run_dir/barrier"
log_dir="$run_dir/logs"

[[ -d "$runtime_template" ]] || {
  echo "error: runtime template does not exist: $runtime_template" >&2; exit 2; }
[[ ! -e "$run_dir" ]] || {
  echo "error: run directory already exists: $run_dir" >&2; exit 2; }
mkdir -p "$runtime_root" "$barrier_dir" "$log_dir"

container_names=()
wrapper_pids=()
cleanup() {
  for name in "${container_names[@]}"; do
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

for ((replica = 0; replica < replicas; replica++)); do
  runtime_dir="$runtime_root/replica-$replica"
  sudo cp -a --reflink=auto "$runtime_template" "$runtime_dir"
  name="qwen4b-${run_id//[^a-zA-Z0-9_.-]/-}-$replica"
  container_names+=("$name")
  docker run --rm --name "$name" --entrypoint /bin/bash \
    --privileged --device=/dev/neuron0 --shm-size="$shm" \
    -v "$model_dir":/models/Qwen3.5-4B:ro \
    -v "$source_dir":/work:ro \
    -v "$runtime_dir":/tmp \
    -v "$barrier_dir":/barrier \
    -v "$log_dir":/logs \
    -w /work \
    "${common_env[@]}" \
    -e NEURON_RT_VISIBLE_CORES="$replica" \
    "$image" -c \
    "$(replica_cmd "$replica" "$replicas" "$warmup" "$bench_iters" \
        --barrier-dir /barrier --output-json "/logs/replica-$replica.json")" \
    >"$log_dir/replica-$replica.log" 2>&1 &
  wrapper_pids+=("$!")
  sleep 2
done

status=0
for pid in "${wrapper_pids[@]}"; do
  wait "$pid" || status=1
done
trap - INT TERM
(( status == 0 )) || {
  echo "error: at least one replica failed; inspect $log_dir" >&2; exit "$status"; }

python3 "$source_dir/aggregate_dp8.py" \
  "$log_dir/replica-*.json" \
  --output-json "$run_dir/aggregate.json"
echo "results: $run_dir/aggregate.json"
