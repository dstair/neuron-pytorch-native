#!/usr/bin/env bash

set -euo pipefail

CACHE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN35_SOURCE_DIR="$(cd "$CACHE_SCRIPT_DIR/../.." && pwd)"
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

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_cache_s3_uri() {
  [[ -n "${QWEN35_COMPILER_CACHE_S3_URI:-}" ]] ||
    die "set QWEN35_COMPILER_CACHE_S3_URI in $ENV_FILE or the environment"
  CACHE_S3_URI="${QWEN35_COMPILER_CACHE_S3_URI%/}"
  [[ "$CACHE_S3_URI" =~ ^s3://[^/]+(/.*)?$ ]] ||
    die "QWEN35_COMPILER_CACHE_S3_URI must be an s3:// URI"
}

validate_cache_key() {
  local key="$1"
  [[ "$key" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "cache key must contain only letters, digits, dot, underscore, and hyphen"
}

resolve_cache_dir() {
  local supplied="${1:-}"
  local cache_dir="${supplied:-${QWEN35_COMPILER_CACHE_DIR:-}}"
  [[ -n "$cache_dir" ]] ||
    die "pass a cache directory or set QWEN35_COMPILER_CACHE_DIR in $ENV_FILE"
  printf '%s\n' "${cache_dir%/}"
}

cache_prefix() {
  printf '%s/%s\n' "$CACHE_S3_URI" "$1"
}

cache_payload_uri() {
  printf '%s/cache/\n' "$(cache_prefix "$1")"
}

cache_manifest_uri() {
  printf '%s/manifest.env\n' "$(cache_prefix "$1")"
}

assert_cache_root() {
  local cache_dir="$1"
  [[ -d "$cache_dir" ]] || die "cache directory does not exist: $cache_dir"
  [[ -d "$cache_dir/neff_cache" ]] ||
    die "cache directory has no neff_cache subtree: $cache_dir"
  [[ -n "$(find "$cache_dir/neff_cache" -type f -name '*.neff' -print -quit)" ]] ||
    die "cache directory has no NEFF files: $cache_dir"
}

cache_file_count() {
  find "$1" -type f | wc -l | tr -d '[:space:]'
}

cache_neff_count() {
  find "$1" -type f -name '*.neff' | wc -l | tr -d '[:space:]'
}

cache_byte_count() {
  du -sb "$1" | awk '{print $1}'
}

directory_is_nonempty() {
  [[ -d "$1" && -n "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

assert_safe_replace_target() {
  local cache_dir="$1"
  [[ "$cache_dir" != "/" && "$cache_dir" != "." && "$cache_dir" != ".." ]] ||
    die "refusing to replace unsafe cache directory: $cache_dir"
}

write_manifest() {
  local manifest="$1"
  local key="$2"
  local cache_dir="$3"
  local revision="unknown"
  local detected_revision

  if detected_revision="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"; then
    revision="$detected_revision"
  fi

  cat >"$manifest" <<EOF
schema_version=1
cache_key=$key
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cache_root_basename=$(basename "$cache_dir")
cache_files=$(cache_file_count "$cache_dir")
neff_files=$(cache_neff_count "$cache_dir")
cache_bytes=$(cache_byte_count "$cache_dir")
git_revision=$revision
native_image=${QWEN35_NATIVE_IMAGE:-unknown}
neuron_cc_flags=${NEURON_CC_FLAGS:-unknown}
cache_layout=complete_container_tmp_mount
required_container_mount=/tmp
EOF
}
