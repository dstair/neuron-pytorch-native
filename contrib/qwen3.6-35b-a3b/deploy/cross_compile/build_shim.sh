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

image="${QWEN35_NATIVE_IMAGE_RESOLVED:-${QWEN35_NATIVE_IMAGE:-}}"
output="${1:-${QWEN35_PLATFORM_TARGET_SHIM:-$SCRIPT_DIR/build/libnrt_platform_target_override.so}}"

[[ -n "$image" ]] ||
  die "set QWEN35_NATIVE_IMAGE or QWEN35_NATIVE_IMAGE_RESOLVED"
command -v docker >/dev/null 2>&1 || die "missing required command: docker"

output_dir="$(mkdir -p "$(dirname "$output")" && cd "$(dirname "$output")" && pwd)"
output_name="$(basename "$output")"
temporary_name=".${output_name}.tmp.$$"

docker run --rm --entrypoint bash \
  --user "$(id -u):$(id -g)" \
  -v "$SCRIPT_DIR":/source:ro \
  -v "$output_dir":/output \
  "$image" -lc "
    set -euo pipefail
    gcc -O2 -Wall -Wextra -Werror -shared -fPIC -pthread \
      -I/opt/aws/neuron/include \
      /source/nrt_platform_target_override.c -ldl \
      -o '/output/$temporary_name'
    mv '/output/$temporary_name' '/output/$output_name'
  "

echo "$output_dir/$output_name"
