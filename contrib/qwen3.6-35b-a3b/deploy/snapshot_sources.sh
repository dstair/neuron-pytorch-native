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

[[ $# -eq 1 ]] || die "usage: snapshot_sources.sh <output-directory>"
[[ -n "${QWEN35_NKILIB_DIR:-}" ]] ||
  die "set QWEN35_NKILIB_DIR to the patched nki-library checkout"
[[ -d "$QWEN35_NKILIB_DIR/.git" ]] ||
  die "QWEN35_NKILIB_DIR is not a git checkout: $QWEN35_NKILIB_DIR"

for command in git tar sha256sum; do
  command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done

output_dir="$1"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
model_archive="$output_dir/neuron-pytorch-native-qwen35.tar.gz"
nkilib_archive="$output_dir/nki-library.tar.gz"
metadata="$output_dir/snapshot.env"

archive_paths() {
  local repo="$1"
  local archive="$2"
  shift 2
  git -C "$repo" ls-files -co --exclude-standard -z -- "$@" |
    tar -C "$repo" --null --files-from=- -czf "$archive"
}

status_sha256() {
  local repo="$1"
  {
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --porcelain=v1
    git -C "$repo" diff --binary
  } | sha256sum | awk '{print $1}'
}

archive_paths "$REPO_ROOT" "$model_archive" contrib/qwen3.6-35b-a3b
archive_paths "$QWEN35_NKILIB_DIR" "$nkilib_archive" src/nkilib_src test

model_revision="$(git -C "$REPO_ROOT" rev-parse HEAD)"
nkilib_revision="$(git -C "$QWEN35_NKILIB_DIR" rev-parse HEAD)"
model_archive_sha256="$(sha256sum "$model_archive" | awk '{print $1}')"
nkilib_archive_sha256="$(sha256sum "$nkilib_archive" | awk '{print $1}')"
model_status_sha256="$(status_sha256 "$REPO_ROOT")"
nkilib_status_sha256="$(status_sha256 "$QWEN35_NKILIB_DIR")"

{
  printf 'export QWEN35_SOURCE_ARCHIVE_SHA256=%q\n' "$model_archive_sha256"
  printf 'export QWEN35_SOURCE_STATUS_SHA256=%q\n' "$model_status_sha256"
  printf 'export QWEN35_SOURCE_GIT_REVISION=%q\n' "$model_revision"
  printf 'export QWEN35_NKILIB_ARCHIVE_SHA256=%q\n' "$nkilib_archive_sha256"
  printf 'export QWEN35_NKILIB_STATUS_SHA256=%q\n' "$nkilib_status_sha256"
  printf 'export QWEN35_NKILIB_REVISION=%q\n' "$nkilib_revision"
} >"$metadata"

git -C "$REPO_ROOT" status --porcelain=v1 >"$output_dir/model-status.txt"
git -C "$QWEN35_NKILIB_DIR" status --porcelain=v1 >"$output_dir/nkilib-status.txt"
git -C "$REPO_ROOT" diff --binary -- contrib/qwen3.6-35b-a3b >"$output_dir/model.patch"
git -C "$QWEN35_NKILIB_DIR" diff --binary >"$output_dir/nkilib.patch"
(
  cd "$output_dir"
  sha256sum "$(basename "$model_archive")" "$(basename "$nkilib_archive")"
) >"$output_dir/SHA256SUMS"

echo "model archive: $model_archive ($model_archive_sha256)"
echo "nkilib archive: $nkilib_archive ($nkilib_archive_sha256)"
echo "snapshot metadata: $metadata"
