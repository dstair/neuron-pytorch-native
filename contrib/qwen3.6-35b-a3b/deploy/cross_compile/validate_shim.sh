#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN35_SOURCE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
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

physical_target="${1:-}"
override_target="${2:-}"
image="${QWEN35_NATIVE_IMAGE_RESOLVED:-${QWEN35_NATIVE_IMAGE:-}}"
shim="${QWEN35_PLATFORM_TARGET_SHIM:-$SCRIPT_DIR/build/libnrt_platform_target_override.so}"

[[ "$physical_target" == "trn1" || "$physical_target" == "trn2" ]] ||
  die "usage: validate_shim.sh <physical-target: trn1|trn2> <override-target: trn1|trn2>"
[[ "$override_target" == "trn1" || "$override_target" == "trn2" ]] ||
  die "usage: validate_shim.sh <physical-target: trn1|trn2> <override-target: trn1|trn2>"
[[ -n "$image" ]] || die "set QWEN35_NATIVE_IMAGE or QWEN35_NATIVE_IMAGE_RESOLVED"

if [[ ! -f "$shim" || "$SCRIPT_DIR/nrt_platform_target_override.c" -nt "$shim" ]]; then
  QWEN35_NATIVE_IMAGE_RESOLVED="$image" "$SCRIPT_DIR/build_shim.sh" "$shim" >/dev/null
fi
shim="$(cd "$(dirname "$shim")" && pwd)/$(basename "$shim")"

docker run --rm --privileged \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e LD_PRELOAD=/opt/qwen35/libnrt_platform_target_override.so \
  -e QWEN35_CACHE_PLATFORM_TARGET="$override_target" \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$shim":/opt/qwen35/libnrt_platform_target_override.so:ro \
  -v "$SCRIPT_DIR/validate_shim.py":/opt/qwen35/validate_shim.py:ro \
  "$image" /opt/torch-neuronx/.venv/bin/python \
    /opt/qwen35/validate_shim.py \
    --physical "$physical_target" \
    --override "$override_target"
