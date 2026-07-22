#!/bin/bash

PROFILE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN35_SOURCE_DIR="$(cd "$PROFILE_SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$QWEN35_SOURCE_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "error: set $name in $ENV_FILE or the environment" >&2
    exit 2
  fi
}

require_dir() {
  local name="$1"
  require_env "$name"
  if [[ ! -d "${!name}" ]]; then
    echo "error: $name is not a directory: ${!name}" >&2
    exit 2
  fi
}

require_qwen35_model() {
  require_dir QWEN35_MODEL_DIR
  QWEN35_MODEL_DIR="$(cd "$QWEN35_MODEL_DIR" && pwd)"
}

require_env QWEN35_NATIVE_IMAGE
require_env QWEN35_PROFILE_ROOT

mkdir -p "$QWEN35_PROFILE_ROOT"
QWEN35_PROFILE_ROOT="$(cd "$QWEN35_PROFILE_ROOT" && pwd)"
