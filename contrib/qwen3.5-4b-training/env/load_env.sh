#!/usr/bin/env bash
# Sourced by remote.sh / container.sh / setup_box.sh. Loads DEPLOYMENT state (which
# instance, which SSH key, which non-public image) from a gitignored .env, so none of it
# lives in tracked files. See the repo-root .env.example for the variables.
#
# Lookup: $ENV_FILE if set, else the first .env found walking up from this directory.
# That covers both layouts: the repo root in a git checkout, and $WORK_ROOT/.env on the
# box (the package is rsynced to $WORK_ROOT/src, and the repo root is not pushed).
# Values already in the environment win over the file.
_qt_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_qt_file=""
if [[ -n "${ENV_FILE:-}" ]]; then
  _qt_file="$ENV_FILE"
else
  _d="$_qt_dir"
  while [[ "$_d" != "/" ]]; do
    if [[ -f "$_d/.env" ]]; then _qt_file="$_d/.env"; break; fi
    _d="$(dirname "$_d")"
  done
  unset _d
fi
if [[ -n "$_qt_file" && -f "$_qt_file" ]]; then
  # Snapshot the caller's environment, load the file, then restore the snapshot so
  # explicit env vars take precedence over .env.
  _qt_saved="$(export -p)"
  set -a; . "$_qt_file"; set +a
  eval "$_qt_saved"
  unset _qt_saved
fi
unset _qt_dir _qt_file

# require_env VAR "what it is"  -- fail loudly instead of guessing a default.
require_env() {
  if [[ -z "${!1:-}" ]]; then
    echo "error: $1 is unset ($2)." >&2
    echo "  Set it in the repo-root .env (copy .env.example) or in the environment." >&2
    exit 2
  fi
}
