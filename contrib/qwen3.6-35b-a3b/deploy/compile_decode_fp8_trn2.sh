#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SOURCE_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

die() {
  echo "error: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: compile_decode_fp8_trn2.sh [options]

Options:
  --mode MODE          fp8 or reference
  --fp8-impl MODE      row, dual, block_pow2, or block_pow2_coalesced
                       (default: row)
  --layout MODE        weight or token stationary (default: weight)
  --fp8-projections P  all, gate_up, or down (default: all)
  --fp8-layer-start N  First layer using FP8 projections (default: 0)
  --fp8-layer-limit N  FP8 only in layers [0,N); later layers use BF16 (default: 40)
  --residual MODE      bf16 or fp32 residual stream (default: bf16)
  --layers N           Model prefix depth (default: 4)
  --batch-size N       32, 64, 128, or 256 (default: 32)
  --diagnostic-layer S Comma-separated layer indices or all
  --reference-rounding S
                       Comma-separated reference BF16 boundaries
  --direct-state-output MODE
                       on or off (default: on)
  --compile-concurrency N
                       Concurrent cross-target rank compiles, 1-8 (default: 8)
  --cache-dir DIR      Complete host directory mounted at container /tmp
  --log-name NAME      Log basename (default derived from the graph)

Required environment:
  QWEN35_NATIVE_IMAGE
  QWEN35_MODEL_DIR
  QWEN35_FP8_MODEL_DIR (or QWEN35_FP8_MODEL_PATH)
  QWEN35_NKILIB_DIR (patched by deploy/nkilib_row_fp8_int8_storage.patch)

The physical host may be Trn1. Compilation targets Trn2 TP=8/LNC=1 and exits
after all ranks have generated their graph cache entries; Trn2 load failure on
the compile host is expected.
EOF
}

mode=fp8
fp8_impl=row
layout=weight
fp8_projections=all
fp8_layer_start=0
fp8_layer_limit=40
residual=bf16
layers=4
batch_size=32
diagnostic_layer=""
reference_rounding=""
direct_state_output=on
compile_concurrency="${QWEN35_CROSS_TARGET_COMPILE_CONCURRENCY:-8}"
cache_dir="${QWEN35_COMPILER_CACHE_DIR:-}"
log_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:-}"; shift 2 ;;
    --fp8-impl) fp8_impl="${2:-}"; shift 2 ;;
    --layout) layout="${2:-}"; shift 2 ;;
    --fp8-projections) fp8_projections="${2:-}"; shift 2 ;;
    --fp8-layer-start) fp8_layer_start="${2:-}"; shift 2 ;;
    --fp8-layer-limit) fp8_layer_limit="${2:-}"; shift 2 ;;
    --residual) residual="${2:-}"; shift 2 ;;
    --layers) layers="${2:-}"; shift 2 ;;
    --batch-size) batch_size="${2:-}"; shift 2 ;;
    --diagnostic-layer) diagnostic_layer="${2:-}"; shift 2 ;;
    --reference-rounding) reference_rounding="${2:-}"; shift 2 ;;
    --direct-state-output) direct_state_output="${2:-}"; shift 2 ;;
    --compile-concurrency) compile_concurrency="${2:-}"; shift 2 ;;
    --cache-dir) cache_dir="${2:-}"; shift 2 ;;
    --log-name) log_name="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ "$mode" == "fp8" || "$mode" == "reference" ]] ||
  die "--mode must be fp8 or reference"
[[ "$fp8_impl" == "row" || "$fp8_impl" == "dual" ||
   "$fp8_impl" == "block_pow2" ||
   "$fp8_impl" == "block_pow2_coalesced" ]] ||
  die "--fp8-impl must be row, dual, block_pow2, or block_pow2_coalesced"
[[ "$mode" == "fp8" || "$fp8_impl" == "row" ]] ||
  die "--fp8-impl applies only to --mode fp8"
[[ "$layout" == "weight" || "$layout" == "token" ]] ||
  die "--layout must be weight or token"
[[ "$fp8_projections" == "all" || "$fp8_projections" == "gate_up" ||
   "$fp8_projections" == "down" ]] ||
  die "--fp8-projections must be all, gate_up, or down"
[[ "$mode" == "fp8" || "$fp8_projections" == "all" ]] ||
  die "--fp8-projections applies only to --mode fp8"
[[ "$fp8_impl" == "row" || "$fp8_projections" == "all" ]] ||
  die "--fp8-projections requires --fp8-impl row"
[[ "$fp8_layer_start" =~ ^([0-9]|[1-3][0-9]|40)$ ]] ||
  die "--fp8-layer-start must be in [0, 40]"
[[ "$fp8_layer_limit" =~ ^([0-9]|[1-3][0-9]|40)$ ]] ||
  die "--fp8-layer-limit must be in [0, 40]"
(( fp8_layer_start <= fp8_layer_limit )) ||
  die "--fp8-layer-start must be <= --fp8-layer-limit"
[[ "$mode" == "fp8" ||
   ( "$fp8_layer_start" == "0" && "$fp8_layer_limit" == "40" ) ]] ||
  die "--fp8-layer-start/limit applies only to --mode fp8"
[[ "$fp8_impl" == "row" ||
   ( "$fp8_layer_start" == "0" && "$fp8_layer_limit" == "40" ) ]] ||
  die "--fp8-layer-start/limit requires --fp8-impl row"
[[ "$mode" == "fp8" || "$layout" == "weight" ]] ||
  die "--layout applies only to --mode fp8"
[[ "$residual" == "bf16" || "$residual" == "fp32" ]] ||
  die "--residual must be bf16 or fp32"
[[ "$mode" == "reference" || -z "$reference_rounding" ]] ||
  die "--reference-rounding requires --mode reference"
[[ "$direct_state_output" == "on" || "$direct_state_output" == "off" ]] ||
  die "--direct-state-output must be on or off"
[[ "$layers" =~ ^[1-9][0-9]*$ && "$layers" -le 40 ]] ||
  die "--layers must be in [1, 40]"
[[ "$batch_size" == "32" || "$batch_size" == "64" ||
   "$batch_size" == "128" || "$batch_size" == "256" ]] ||
  die "--batch-size must be 32, 64, 128, or 256"
[[ "$fp8_impl" != "block_pow2_coalesced" || "$batch_size" != "256" ]] ||
  die "--fp8-impl block_pow2_coalesced supports batch sizes 32, 64, and 128"
[[ "$compile_concurrency" =~ ^[1-8]$ ]] ||
  die "--compile-concurrency must be in [1, 8]"
[[ -n "$cache_dir" ]] ||
  die "pass --cache-dir or set QWEN35_COMPILER_CACHE_DIR"
[[ -n "${QWEN35_NATIVE_IMAGE:-}" ]] ||
  die "set QWEN35_NATIVE_IMAGE"
[[ -n "${QWEN35_MODEL_DIR:-}" && -d "$QWEN35_MODEL_DIR" ]] ||
  die "set QWEN35_MODEL_DIR to the BF16 checkpoint"
[[ -n "${QWEN35_NKILIB_DIR:-}" &&
   -d "$QWEN35_NKILIB_DIR/src/nkilib_src" ]] ||
  die "set QWEN35_NKILIB_DIR to the patched nki-library checkout"
fp8_model_dir="${QWEN35_FP8_MODEL_DIR:-${QWEN35_FP8_MODEL_PATH:-}}"
[[ -n "$fp8_model_dir" && -d "$fp8_model_dir" ]] ||
  die "set QWEN35_FP8_MODEL_DIR to the official FP8 checkpoint"

for command in docker sha256sum; do
  command -v "$command" >/dev/null 2>&1 ||
    die "missing required command: $command"
done

cache_dir="$(mkdir -p "$cache_dir" && cd "$cache_dir" && pwd)"
model_dir="$(cd "$QWEN35_MODEL_DIR" && pwd)"
fp8_model_dir="$(cd "$fp8_model_dir" && pwd)"
nkilib_dir="$(cd "$QWEN35_NKILIB_DIR" && pwd)"
diagnostic_tag="${diagnostic_layer:-none}"
diagnostic_tag="${diagnostic_tag//,/-}"
projection_tag=""
if [[ "$mode" == "fp8" ]]; then
  projection_tag="-${fp8_impl}"
  if [[ "$fp8_impl" == "row" ]]; then
    projection_tag+="-p${fp8_projections}-fs${fp8_layer_start}-fl${fp8_layer_limit}"
  fi
fi
log_name="${log_name:-${mode}${projection_tag}-l${layers}-b${batch_size}-d${diagnostic_tag}}"
log_dir="$cache_dir/compile_logs/$log_name"
mkdir -p "$log_dir"

if ! docker image inspect "$QWEN35_NATIVE_IMAGE" >/dev/null 2>&1; then
  docker pull "$QWEN35_NATIVE_IMAGE"
fi
resolved_image="$(
  docker image inspect \
    --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' \
    "$QWEN35_NATIVE_IMAGE"
)"
shim="${QWEN35_PLATFORM_TARGET_SHIM:-$SCRIPT_DIR/cross_compile/build/libnrt_platform_target_override.so}"
QWEN35_NATIVE_IMAGE_RESOLVED="$resolved_image" \
  "$SCRIPT_DIR/cross_compile/build_shim.sh" "$shim" >/dev/null
shim="$(cd "$(dirname "$shim")" && pwd)/$(basename "$shim")"

residual_fp32=0
if [[ "$residual" == "fp32" ]]; then
  residual_fp32=1
fi
direct_state_output_value=0
if [[ "$direct_state_output" == "on" ]]; then
  direct_state_output_value=1
fi
mode_env=(
  -e MOE_FUSED_W8=fp8
  -e "MOE_FUSED_W8_FP8_IMPL=$fp8_impl"
  -e "MOE_FUSED_W8_FP8_PROJECTIONS=$fp8_projections"
  -e "MOE_FUSED_W8_FP8_LAYER_START=$fp8_layer_start"
  -e "MOE_FUSED_W8_FP8_LAYER_LIMIT=$fp8_layer_limit"
  -e "MOE_FUSED_W8_LAYOUT=$layout"
  -e "MOE_W8_RESIDUAL_FP32=$residual_fp32"
)
if [[ "$mode" == "reference" ]]; then
  mode_env=(
    -e MOE_OFFICIAL_FP8_REFERENCE=1
    -e "MOE_OFFICIAL_FP8_REFERENCE_ROUNDING=$reference_rounding"
    -e "MOE_W8_RESIDUAL_FP32=$residual_fp32"
  )
fi
diagnostic_args=()
if [[ -n "$diagnostic_layer" ]]; then
  diagnostic_args=(--diagnostic-layer "$diagnostic_layer")
fi

export QWEN35_NATIVE_IMAGE_RESOLVED="$resolved_image"
export QWEN35_CACHE_PLATFORM_TARGET=trn2
export QWEN35_PHYSICAL_PLATFORM_TARGET=trn1
export QWEN35_TP=8
export QWEN35_LNC=1
export QWEN35_BATCH_SIZE="$batch_size"
export QWEN35_NUM_LAYERS="$layers"
export QWEN35_MAX_SEQ_LEN=256
export QWEN35_REFERENCE_ROUNDING="$reference_rounding"
export QWEN35_FP8_LAYOUT="$layout"
export QWEN35_FP8_IMPL="$fp8_impl"
export QWEN35_FP8_PROJECTIONS="$fp8_projections"
export QWEN35_FP8_LAYER_START="$fp8_layer_start"
export QWEN35_FP8_LAYER_LIMIT="$fp8_layer_limit"
export QWEN35_RESIDUAL="$residual"
export QWEN35_DIRECT_STATE_OUTPUT="$direct_state_output"
export QWEN35_COMPILE_CONCURRENCY="$compile_concurrency"
export NEURON_CC_FLAGS="--target trn2 --lnc 1"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

metadata="$cache_dir/qwen35_compile_metadata.env"
cat >"$metadata" <<EOF
export QWEN35_NATIVE_IMAGE_RESOLVED=$(printf '%q' "$resolved_image")
export QWEN35_CACHE_PLATFORM_TARGET=trn2
export QWEN35_PHYSICAL_PLATFORM_TARGET=trn1
export QWEN35_TP=8
export QWEN35_LNC=1
export QWEN35_BATCH_SIZE=$(printf '%q' "$batch_size")
export QWEN35_NUM_LAYERS=$(printf '%q' "$layers")
export QWEN35_MAX_SEQ_LEN=256
export QWEN35_FP8_MODE=$(printf '%q' "$mode")
export QWEN35_FP8_IMPL=$(printf '%q' "$fp8_impl")
export QWEN35_FP8_LAYOUT=$(printf '%q' "$layout")
export QWEN35_FP8_PROJECTIONS=$(printf '%q' "$fp8_projections")
export QWEN35_FP8_LAYER_START=$(printf '%q' "$fp8_layer_start")
export QWEN35_FP8_LAYER_LIMIT=$(printf '%q' "$fp8_layer_limit")
export QWEN35_RESIDUAL=$(printf '%q' "$residual")
export QWEN35_DIRECT_STATE_OUTPUT=$(printf '%q' "$direct_state_output")
export QWEN35_DIAGNOSTIC_LAYER=$(printf '%q' "$diagnostic_layer")
export QWEN35_REFERENCE_ROUNDING=$(printf '%q' "$reference_rounding")
export QWEN35_COMPILE_CONCURRENCY=$(printf '%q' "$compile_concurrency")
export NEURON_CC_FLAGS=$(printf '%q' "$NEURON_CC_FLAGS")
EOF

container_name="q35-fp8-${log_name//[^A-Za-z0-9_.-]/-}-$$"
rank_log_dir="/tmp/compile_logs/$log_name/ranks"
docker_log="$log_dir/docker.log"
resource_log="$log_dir/resources.log"
exit_file="$log_dir/container-exit-code"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$container_name" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so \
  -e QWEN35_CACHE_PLATFORM_TARGET=trn2 \
  -e NEURON_LOGICAL_NC_CONFIG=1 \
  -e NEURON_CC_FLAGS="$NEURON_CC_FLAGS" \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e CROSS_TARGET_COMPILE_ONLY=1 \
  -e CROSS_TARGET_MARKER_DIR=/tmp/cross-target-compile \
  -e CROSS_TARGET_MARKER_TIMEOUT_SECONDS=7200 \
  -e CROSS_TARGET_COMPILE_CONCURRENCY="$compile_concurrency" \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -e DECODE_FULLGRAPH=1 -e DECODE_SHARDED_LM_HEAD=1 \
  -e DN_NKI=1 -e DNBATCHED_V2=1 \
  -e "DN_DIRECT_STATE_OUT=$direct_state_output_value" \
  -e "DN_WIDE_CONV=${DN_WIDE_CONV:-0}" \
  -e GQATAIL=1 -e GQA_STATEFUL_KV=1 \
  -e "XLA_IR_DEBUG=${XLA_IR_DEBUG:-0}" \
  -e "XLA_HLO_DEBUG=${XLA_HLO_DEBUG:-0}" \
  -e "NEURON_FRAMEWORK_DEBUG=${NEURON_FRAMEWORK_DEBUG:-0}" \
  -e DN_K_HEADS=2 -e DN_V_HEADS=4 -e GQA_Q_HEADS=2 \
  "${mode_env[@]}" \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$shim":/opt/qwen35/libnrt_platform_target_override.so:ro \
  -v "$SOURCE_DIR":/qwen:ro \
  -v "$nkilib_dir":/nki-library:ro \
  -v "$model_dir":/models/Qwen3.5-35B-A3B:ro \
  -v "$fp8_model_dir":/models/Qwen3.5-35B-A3B-FP8:ro \
  -v "$cache_dir":/tmp \
  -w /qwen "$resolved_image" bash -lc "
    set -euo pipefail
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    rm -rf /tmp/cross-target-compile /tmp/cross-output
    mkdir -p '$rank_log_dir'
    torchrun --nproc-per-node=8 --max-restarts=0 \
      --log-dir '$rank_log_dir' --redirects=3 --tee=3 \
      kernels/tests/test_decode_fullgraph_device.py \
      --mode sharded --output-dir /tmp/cross-output \
      --model-path /models/Qwen3.5-35B-A3B \
      --expert-model-path /models/Qwen3.5-35B-A3B-FP8 \
      --num-layers '$layers' --batch-size '$batch_size' \
      --max-seq-len 256 --capture-steps 1 \
      ${diagnostic_args[*]}
  " >/dev/null

while [[ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == "true" ]]; do
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    free -h
    ps -eo pid,ppid,rss,vsz,etimes,cmd |
      grep -E 'neuronx-cc|walrus_driver|torchrun|test_decode_fullgraph' |
      grep -v grep || true
    docker stats --no-stream \
      --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$container_name" ||
      true
  } >>"$resource_log"
  sleep 30
done

exit_code="$(docker inspect --format '{{.State.ExitCode}}' "$container_name")"
printf '%s\n' "$exit_code" >"$exit_file"
docker logs "$container_name" >"$docker_log" 2>&1 || true
echo "container exit code: $exit_code"
echo "cache: $cache_dir"
echo "logs: $log_dir"
exit "$exit_code"
