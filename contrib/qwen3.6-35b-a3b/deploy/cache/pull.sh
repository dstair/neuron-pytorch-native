#!/usr/bin/env bash

# Restore one complete Native compiler-cache root from S3 before container launch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage: pull.sh [--replace] <cache-key> [cache-directory]

Restores the complete cache hierarchy. The target must be empty unless
--replace is supplied. Mount the restored directory at /tmp in the Native
container and use the same graph, image, compiler flags, TP/LNC topology, and
Neuron generation that produced it.
EOF
}

replace=0
if [[ "${1:-}" == "--replace" ]]; then
  replace=1
  shift
fi

[[ $# -ge 1 && $# -le 2 ]] || {
  usage >&2
  exit 2
}

cache_key="$1"
cache_dir="$(resolve_cache_dir "${2:-}")"
validate_cache_key "$cache_key"
require_command aws
require_cache_s3_uri

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
manifest_uri="$(cache_manifest_uri "$cache_key")"
payload_uri="$(cache_payload_uri "$cache_key")"

aws s3 cp "$manifest_uri" "$manifest" --no-progress --only-show-errors ||
  die "cache manifest is not available: $manifest_uri"

if directory_is_nonempty "$cache_dir"; then
  [[ "$replace" -eq 1 ]] ||
    die "target cache directory is not empty: $cache_dir (use --replace to discard it)"
  assert_safe_replace_target "$cache_dir"
  rm -rf -- "$cache_dir"
fi
mkdir -p "$cache_dir"

echo "restoring complete cache root: $payload_uri -> $cache_dir"
aws s3 sync "$payload_uri" "$cache_dir/" --no-progress --only-show-errors
assert_cache_root "$cache_dir"

expected_neffs="$(awk -F= '$1 == "neff_files" { print $2; exit }' "$manifest")"
actual_neffs="$(cache_neff_count "$cache_dir")"
[[ -n "$expected_neffs" && "$expected_neffs" == "$actual_neffs" ]] ||
  die "restored NEFF count ($actual_neffs) does not match manifest ($expected_neffs)"

echo "restored $actual_neffs NEFFs, $(cache_file_count "$cache_dir") files, $(du -sh "$cache_dir" | awk '{print $1}')"
echo "manifest:"
cat "$manifest"
