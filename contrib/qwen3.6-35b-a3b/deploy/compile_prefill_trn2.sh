#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN35_SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$QWEN35_SOURCE_DIR/../.." && pwd)"
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
Usage: compile_prefill_trn2.sh [options]

Options:
  --layers N       Model prefix depth: 20, 30, or 40 (default: 40)
  --splits N       Compiled prefill regions: 1 or 4 (default: 1)
  --bucket N       Query bucket size: 512 or 1024 (default: 1024)
  --tp N           Tensor-parallel degree: 4 or 8 (default: 8)
  --lnc N          Logical NeuronCore degree: 1 or 2 (default: 1)
  --cache-platform-target TARGET
                   Torch NeuronX cache identity: trn1 or trn2 (default: trn2)
  --scratchpad-page-size-mb MIB
                   Compiler/runtime HBM scratchpad page size in MiB (default: 64)
  --optlevel N     neuronx-cc optimization level: 1, 2, or 3 (default: 1)
  --cache-dir DIR  Complete host directory mounted at container /tmp
  --log-name NAME  Compile-log basename (default derived from shape)

The supported topology pairs are TP=8/LNC=1 and TP=4/LNC=2. The target is
always Trn2; optlevel defaults to 1 and is selectable with --optlevel. The
cache-platform target affects only Torch
NeuronX's persistent-cache key. Use trn2 for a portable cross-compile on Trn1;
use trn1 only to replay a legacy cache whose keys were generated on Trn1.
The overall command may fail after compilation on Trn1 because a Trn2 NEFF
cannot load there; inspect per-rank logs and replay the cache on Trn2.
EOF
}

layers=40
splits=1
bucket=1024
tp=8
lnc=1
cache_platform_target="${QWEN35_CACHE_PLATFORM_TARGET:-trn2}"
scratchpad_page_size_mb="${QWEN35_SCRATCHPAD_PAGE_SIZE_MB:-64}"
optlevel="${QWEN35_OPTLEVEL:-1}"
cache_dir="${QWEN35_COMPILER_CACHE_DIR:-}"
log_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layers) layers="${2:-}"; shift 2 ;;
    --splits) splits="${2:-}"; shift 2 ;;
    --bucket) bucket="${2:-}"; shift 2 ;;
    --tp) tp="${2:-}"; shift 2 ;;
    --lnc) lnc="${2:-}"; shift 2 ;;
    --cache-platform-target) cache_platform_target="${2:-}"; shift 2 ;;
    --scratchpad-page-size-mb) scratchpad_page_size_mb="${2:-}"; shift 2 ;;
    --optlevel) optlevel="${2:-}"; shift 2 ;;
    --cache-dir) cache_dir="${2:-}"; shift 2 ;;
    --log-name) log_name="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ "$layers" == "20" || "$layers" == "30" || "$layers" == "40" ]] ||
  die "--layers must be 20, 30, or 40"
[[ "$splits" == "1" || "$splits" == "4" ]] ||
  die "--splits must be 1 or 4"
[[ "$bucket" == "512" || "$bucket" == "1024" ]] ||
  die "--bucket must be 512 or 1024"
[[ "$tp:$lnc" == "8:1" || "$tp:$lnc" == "4:2" ]] ||
  die "supported topology pairs are TP=8/LNC=1 and TP=4/LNC=2"
[[ "$cache_platform_target" == "trn1" || "$cache_platform_target" == "trn2" ]] ||
  die "--cache-platform-target must be trn1 or trn2"
[[ "$scratchpad_page_size_mb" =~ ^[1-9][0-9]*$ ]] &&
  (( scratchpad_page_size_mb < 4096 )) ||
  die "--scratchpad-page-size-mb must be an integer between 1 and 4095"
[[ "$optlevel" == "1" || "$optlevel" == "2" || "$optlevel" == "3" ]] ||
  die "--optlevel must be 1, 2, or 3"
[[ -n "$cache_dir" ]] ||
  die "pass --cache-dir or set QWEN35_COMPILER_CACHE_DIR"
[[ -n "${QWEN35_NATIVE_IMAGE:-}" ]] ||
  die "set QWEN35_NATIVE_IMAGE"
[[ -n "${QWEN35_MODEL_DIR:-}" && -d "$QWEN35_MODEL_DIR" ]] ||
  die "set QWEN35_MODEL_DIR to the model checkpoint"
[[ -n "${QWEN35_NKILIB_DIR:-}" && -d "$QWEN35_NKILIB_DIR/src/nkilib_src" ]] ||
  die "set QWEN35_NKILIB_DIR to the patched nki-library checkout"
[[ -d /opt/aws/neuron/lib ]] ||
  die "host Neuron runtime is missing: /opt/aws/neuron/lib"

for command in docker sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done

cache_dir="$(mkdir -p "$cache_dir" && cd "$cache_dir" && pwd)"
model_dir="$(cd "$QWEN35_MODEL_DIR" && pwd)"
nkilib_dir="$(cd "$QWEN35_NKILIB_DIR" && pwd)"
source_dir="$QWEN35_SOURCE_DIR"
log_name="${log_name:-tp${tp}-lnc${lnc}-l${layers}-s${splits}-b${bucket}-o${optlevel}}"
log_dir="$cache_dir/compile_logs/$log_name"
mkdir -p "$log_dir"

if ! docker image inspect "$QWEN35_NATIVE_IMAGE" >/dev/null 2>&1; then
  docker pull "$QWEN35_NATIVE_IMAGE"
fi
resolved_image="$(
  docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' \
    "$QWEN35_NATIVE_IMAGE"
)"
[[ -n "$resolved_image" ]] || die "unable to resolve image digest"
shim="${QWEN35_PLATFORM_TARGET_SHIM:-$SCRIPT_DIR/cross_compile/build/libnrt_platform_target_override.so}"
QWEN35_NATIVE_IMAGE_RESOLVED="$resolved_image" \
  "$SCRIPT_DIR/cross_compile/build_shim.sh" "$shim" >/dev/null
shim="$(cd "$(dirname "$shim")" && pwd)/$(basename "$shim")"
shim_sha256="$(sha256sum "$shim" | awk '{print $1}')"
if ! platform_check_output="$(
  docker run --rm --privileged \
    -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
    -e LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so \
    -e QWEN35_CACHE_PLATFORM_TARGET="$cache_platform_target" \
    -e NEURON_LOGICAL_NC_CONFIG="$lnc" \
    -v /opt/aws/neuron:/opt/aws/neuron:ro \
    -v "$shim":/opt/qwen35/libnrt_platform_target_override.so:ro \
    "$resolved_image" \
    /opt/torch-neuronx/.venv/bin/python -c \
      'from torch_neuronx import _C; print(_C._get_platform_target())' \
    2>&1
)"; then
  die "platform shim validation failed: $platform_check_output"
fi
physical_platform_target="$(printf '%s\n' "$platform_check_output" | tail -1)"
[[ "$physical_platform_target" == "trn1" || "$physical_platform_target" == "trn2" ]] ||
  die "platform shim preflight returned an invalid physical target: $physical_platform_target"

host_nrt_library="$(find /opt/aws/neuron/lib -maxdepth 1 -type f -name 'libnrt.so.*' | sort -V | tail -1)"
[[ -n "$host_nrt_library" ]] ||
  die "host Neuron runtime library is missing from /opt/aws/neuron/lib"
host_nrt_sha256="$(sha256sum "$host_nrt_library" | awk '{print $1}')"

neuronx_cc_version="$(
  docker run --rm "$resolved_image" bash -lc \
    'source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true; neuronx-cc --version' \
    2>&1 | tr '\n' ' ' | sed 's/[[:space:]]*$//'
)"

if [[ "$tp" == "8" ]]; then
  dn_k_heads=2
  dn_v_heads=4
  gqa_q_heads=2
else
  dn_k_heads=4
  dn_v_heads=8
  gqa_q_heads=4
fi

export QWEN35_NATIVE_IMAGE_RESOLVED="$resolved_image"
export QWEN35_NEURONX_CC_VERSION="$neuronx_cc_version"
export QWEN35_HOST_NRT_LIBRARY="$(basename "$host_nrt_library")"
export QWEN35_HOST_NRT_SHA256="$host_nrt_sha256"
export QWEN35_PLATFORM_TARGET_SHIM_SHA256="$shim_sha256"
export QWEN35_CACHE_PLATFORM_TARGET="$cache_platform_target"
export QWEN35_PHYSICAL_PLATFORM_TARGET="$physical_platform_target"
export QWEN35_SCRATCHPAD_PAGE_SIZE_MB="$scratchpad_page_size_mb"
export QWEN35_TP="$tp"
export QWEN35_LNC="$lnc"
export QWEN35_BATCH_SIZE=2
export QWEN35_NUM_LAYERS="$layers"
export QWEN35_MAX_SEQ_LEN=20480
export QWEN35_PREFILL_TOKENS=20000
export QWEN35_BUCKET_CHUNK="$bucket"
export QWEN35_PREFILL_SPLITS="$splits"
export CHUNK_SIZE=16
export MOE_CTE_BLOCK=512
export NEURON_CC_FLAGS="--target trn2 --lnc $lnc --optlevel $optlevel --hbm-scratchpad-page-size $scratchpad_page_size_mb"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export QWEN35_PLATFORM_TARGET_SHIM_DEBUG="${QWEN35_PLATFORM_TARGET_SHIM_DEBUG:-0}"
export QWEN35_COMPILE_HOST="${QWEN35_COMPILE_HOST:-$(hostname)}"
export QWEN35_COMPILE_REGION="${QWEN35_COMPILE_REGION:-unknown}"

command_text="NEURON_CC_FLAGS=$NEURON_CC_FLAGS NEURON_PLATFORM_TARGET_OVERRIDE=$NEURON_PLATFORM_TARGET_OVERRIDE QWEN35_CACHE_PLATFORM_TARGET=$QWEN35_CACHE_PLATFORM_TARGET LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so torchrun --nproc-per-node=$tp static_decode_35b.py --model-path /models/Qwen3.5-35B-A3B --batch-size 2 --num-layers $layers --max-seq-len 20480 --prefill-bench 20000 --bucket-chunk $bucket --bucket-compile 1 --prefill-splits $splits --skip-compile"
export QWEN35_COMPILE_COMMAND_SHA256="$(
  printf '%s' "$command_text" | sha256sum | awk '{print $1}'
)"

metadata="$cache_dir/qwen35_compile_metadata.env"
if [[ -f "$metadata" ]]; then
  existing_command_sha="$(
    bash -c 'source "$1"; printf "%s" "${QWEN35_COMPILE_COMMAND_SHA256:-}"' _ "$metadata"
  )"
  [[ -z "$existing_command_sha" || "$existing_command_sha" == "$QWEN35_COMPILE_COMMAND_SHA256" ]] ||
    die "cache directory metadata belongs to a different compile command"
fi

{
  for name in \
    QWEN35_NATIVE_IMAGE QWEN35_NATIVE_IMAGE_RESOLVED QWEN35_NEURONX_CC_VERSION \
    QWEN35_HOST_NRT_LIBRARY QWEN35_HOST_NRT_SHA256 \
    QWEN35_PLATFORM_TARGET_SHIM_SHA256 QWEN35_CACHE_PLATFORM_TARGET \
    QWEN35_PHYSICAL_PLATFORM_TARGET QWEN35_SCRATCHPAD_PAGE_SIZE_MB \
    QWEN35_SOURCE_ARCHIVE_SHA256 QWEN35_SOURCE_STATUS_SHA256 \
    QWEN35_NKILIB_REVISION QWEN35_NKILIB_STATUS_SHA256 QWEN35_NKILIB_ARCHIVE_SHA256 \
    QWEN35_COMPILE_HOST QWEN35_COMPILE_REGION QWEN35_TP QWEN35_LNC \
    QWEN35_BATCH_SIZE QWEN35_NUM_LAYERS QWEN35_MAX_SEQ_LEN QWEN35_PREFILL_TOKENS \
    QWEN35_BUCKET_CHUNK QWEN35_PREFILL_SPLITS QWEN35_COMPILE_COMMAND_SHA256 \
    CHUNK_SIZE MOE_CTE_BLOCK NEURON_CC_FLAGS NEURON_PLATFORM_TARGET_OVERRIDE \
    QWEN35_PLATFORM_TARGET_SHIM_DEBUG
  do
    printf 'export %s=%q\n' "$name" "${!name:-unknown}"
  done
} >"$metadata"

container_name="q35-prefill-${log_name//[^A-Za-z0-9_.-]/-}-$$"
resource_log="$log_dir/resources.log"
docker_log="$log_dir/docker.log"
exit_file="$log_dir/container-exit-code"
rank_log_dir="/tmp/compile_logs/$log_name/ranks"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "$container_name" --privileged \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so \
  -e QWEN35_MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e QWEN35_LNC="$lnc" \
  -e NEURON_LOGICAL_NC_CONFIG="$lnc" \
  -e NEURON_CC_FLAGS="$NEURON_CC_FLAGS" \
  -e NEURON_PLATFORM_TARGET_OVERRIDE="$NEURON_PLATFORM_TARGET_OVERRIDE" \
  -e NEURON_SCRATCHPAD_PAGE_SIZE="$QWEN35_SCRATCHPAD_PAGE_SIZE_MB" \
  -e QWEN35_CACHE_PLATFORM_TARGET="$QWEN35_CACHE_PLATFORM_TARGET" \
  -e QWEN35_PLATFORM_TARGET_SHIM_DEBUG="$QWEN35_PLATFORM_TARGET_SHIM_DEBUG" \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=512 \
  -e GQA_CTE_PREFILL=1 -e GQA_DYNAMIC_ROPE_KV=1 \
  -e DN_CHUNK_NKI=1 -e CHUNK_SIZE=16 -e DN_PAIRED_BATCH=1 \
  -e DN_NKI=1 -e GQATAIL=1 -e PREFILL_FINGERPRINT=1 \
  -e DN_K_HEADS="$dn_k_heads" -e DN_V_HEADS="$dn_v_heads" \
  -e GQA_Q_HEADS="$gqa_q_heads" \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$shim":/opt/qwen35/libnrt_platform_target_override.so:ro \
  -v "$source_dir":/work:ro \
  -v "$nkilib_dir":/nki-library:ro \
  -v "$model_dir":/models/Qwen3.5-35B-A3B:ro \
  -v "$cache_dir":/tmp \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -w /work "$resolved_image" bash -lc "
    set -o pipefail
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    mkdir -p '$rank_log_dir'
    time_command=()
    if [[ -x /usr/bin/time ]]; then
      time_command=(/usr/bin/time -v -o '/tmp/compile_logs/$log_name/time.txt')
    fi
    \"\${time_command[@]}\" torchrun --nproc-per-node='$tp' --max-restarts=0 \
        --log-dir '$rank_log_dir' --redirects=3 --tee=3 \
        static_decode_35b.py \
        --model-path /models/Qwen3.5-35B-A3B \
        --batch-size 2 --num-layers '$layers' --max-seq-len 20480 \
        --prefill-bench 20000 --bucket-chunk '$bucket' \
        --bucket-compile 1 --prefill-splits '$splits' --skip-compile
  " >/dev/null

while [[ "$(docker inspect --format '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == "true" ]]; do
  {
    date -u +%Y-%m-%dT%H:%M:%SZ
    free -h
    ps -eo pid,ppid,rss,vsz,etimes,cmd |
      grep -E 'neuronx-cc|walrus_driver|torchrun|static_decode_35b' |
      grep -v grep || true
    docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}' "$container_name" || true
  } >>"$resource_log"
  sleep 30
done

exit_code="$(docker inspect --format '{{.State.ExitCode}}' "$container_name")"
printf '%s\n' "$exit_code" >"$exit_file"
docker logs "$container_name" >"$docker_log" 2>&1 || true

echo "container exit code: $exit_code"
echo "logs: $log_dir"
echo "resolved image: $resolved_image"
echo "compiler: $neuronx_cc_version"
exit "$exit_code"
