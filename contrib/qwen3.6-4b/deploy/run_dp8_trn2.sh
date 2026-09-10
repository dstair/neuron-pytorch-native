#!/usr/bin/env bash
set -euo pipefail

image="${QWEN4B_NATIVE_IMAGE:?set QWEN4B_NATIVE_IMAGE}"
model_dir="${QWEN4B_MODEL_DIR:?set QWEN4B_MODEL_DIR}"
runtime_template="${QWEN4B_RUNTIME_TEMPLATE:?set QWEN4B_RUNTIME_TEMPLATE}"
run_root="${QWEN4B_RUN_ROOT:?set QWEN4B_RUN_ROOT}"
batch_size="${QWEN4B_BATCH_SIZE:-32}"
fp8_scope="${QWEN4B_FP8_SCOPE:-mlp-head}"
fused_fp8_mlp="${QWEN4B_FUSED_FP8_MLP:-1}"
fused_fp8_head="${QWEN4B_FUSED_FP8_HEAD:-1}"
warmup="${QWEN4B_WARMUP:-5}"
bench_iters="${QWEN4B_BENCH_ITERS:-50}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fused_args=""
[[ "$fused_fp8_mlp" == "1" ]] && fused_args+=" --fused-fp8-mlp"
[[ "$fused_fp8_head" == "1" ]] && fused_args+=" --fused-fp8-head"
run_id="${QWEN4B_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$run_root/$run_id"
runtime_root="$run_dir/runtime"
barrier_dir="$run_dir/barrier"
log_dir="$run_dir/logs"

[[ -d "$runtime_template" ]] || {
  echo "error: runtime template does not exist: $runtime_template" >&2
  exit 2
}
[[ ! -e "$run_dir" ]] || {
  echo "error: run directory already exists: $run_dir" >&2
  exit 2
}
mkdir -p "$runtime_root" "$barrier_dir" "$log_dir"

container_names=()
wrapper_pids=()
cleanup() {
  for name in "${container_names[@]}"; do
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

for replica in 0 1 2 3 4 5 6 7; do
  runtime_dir="$runtime_root/replica-$replica"
  sudo cp -a --reflink=auto "$runtime_template" "$runtime_dir"
  name="qwen4b-${run_id//[^a-zA-Z0-9_.-]/-}-$replica"
  container_names+=("$name")
  docker run --rm --name "$name" --entrypoint /bin/bash \
    --privileged --device=/dev/neuron0 --shm-size=10g \
    -v /opt/aws/neuron:/opt/aws/neuron:ro \
    -v "$model_dir":/models/Qwen3.5-4B:ro \
    -v "$source_dir":/work:ro \
    -v "$runtime_dir":/tmp \
    -v "$barrier_dir":/barrier \
    -v "$log_dir":/logs \
    -w /work \
    -e NEURON_LOGICAL_NC_CONFIG=1 \
    -e NEURON_RT_NUM_CORES=1 \
    -e NEURON_RT_VISIBLE_CORES="$replica" \
    -e "NEURON_CC_FLAGS=--target trn2 --lnc 1 --optlevel 1" \
    "$image" -c \
    "python3 static_decode_dp8.py \
      --model-path /models/Qwen3.5-4B \
      --batch-size $batch_size \
      --max-seq-len 256 \
      --warmup $warmup \
      --bench-iters $bench_iters \
      --fp8-scope $fp8_scope \
      $fused_args \
      --replica-id $replica \
      --replica-count 8 \
      --barrier-dir /barrier \
      --output-json /logs/replica-$replica.json" \
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
  echo "error: at least one replica failed; inspect $log_dir" >&2
  exit "$status"
}

python3 "$source_dir/aggregate_dp8.py" \
  "$log_dir/replica-*.json" \
  --output-json "$run_dir/aggregate.json"
echo "results: $run_dir/aggregate.json"
