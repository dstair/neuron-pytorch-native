#!/bin/bash

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN27_SOURCE_DIR="$(cd "$DEPLOY_DIR/.." && pwd)"
REPO_ROOT="$(cd "$QWEN27_SOURCE_DIR/../.." && pwd)"
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

require_env QWEN27_VLLM_IMAGE
require_dir QWEN27_MODEL_DIR
require_dir QWEN27_VLLM_PLUGIN_DIR

QWEN27_MODEL_DIR="$(cd "$QWEN27_MODEL_DIR" && pwd)"
QWEN27_VLLM_PLUGIN_DIR="$(cd "$QWEN27_VLLM_PLUGIN_DIR" && pwd)"
