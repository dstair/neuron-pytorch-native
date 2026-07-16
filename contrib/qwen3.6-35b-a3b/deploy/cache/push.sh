#!/usr/bin/env bash

# Upload one complete Native compiler-cache root under an immutable S3 cache key.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage: push.sh [--replace] <cache-key> [cache-directory]

Uploads the complete directory mounted as /tmp inside the Native container,
including hlo_cache, neff_cache, and NKI compiler subtrees. Existing S3 keys
are immutable unless --replace is supplied explicitly.
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
assert_cache_root "$cache_dir"

prefix="$(cache_prefix "$cache_key")"
payload_uri="$(cache_payload_uri "$cache_key")"
manifest_uri="$(cache_manifest_uri "$cache_key")"

if ! listing="$(aws s3 ls "$prefix/" --recursive 2>&1)"; then
  # AWS CLI returns 1 with no output for an empty, previously unused prefix.
  # Real authorization and transport failures include a diagnostic.
  if [[ -n "$listing" ]]; then
    echo "$listing" >&2
    die "unable to inspect cache destination: $prefix/"
  fi
  listing=""
fi
if [[ -n "$listing" && "$replace" -ne 1 ]]; then
  die "cache key already exists: $prefix/ (use --replace only for a deliberate refresh)"
fi

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
write_manifest "$manifest" "$cache_key" "$cache_dir"

sync_args=(s3 sync "$cache_dir/" "$payload_uri" --no-progress --only-show-errors)
if [[ "$replace" -eq 1 ]]; then
  sync_args+=(--delete)
fi

echo "uploading complete cache root: $cache_dir -> $payload_uri"
aws "${sync_args[@]}"
aws s3 cp "$manifest" "$manifest_uri" --no-progress --only-show-errors

echo "uploaded $(cache_neff_count "$cache_dir") NEFFs, $(cache_file_count "$cache_dir") files, $(du -sh "$cache_dir" | awk '{print $1}')"
echo "manifest: $manifest_uri"
