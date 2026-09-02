#!/usr/bin/env bash
# Shared environment resolution for the on-box benchmark/gate runners.
#
# Everything here is DEPLOYMENT state, not code: which container image, which
# instance, which scratch directory. None of it belongs in git, so it comes from
# the repo-root `.env` (gitignored; see `.env.example`) or from the environment.
#
# Source this from a runner, then use $IMAGE / $MODEL / $NKILIB / $SRC / $WORK:
#   . "$(dirname "$0")/bench_env.sh"
#
# Two values are REQUIRED because neither has a safe default:
#   QWEN35_NATIVE_IMAGE  the Neuron DLC reference -- embeds an AWS account ID.
#   QWEN35_WORK_DIR      the scratch root (logs, compiler caches, nkilib, staged
#                        src). Deliberately NOT derived from this script's path:
#                        on a box the package sits at $WORK/src/contrib/<pkg>
#                        (three levels down) but in a git checkout it is two, so
#                        any derivation silently encodes one layout and writes
#                        logs to the wrong place in the other.
#
# Overridable (env or .env), with the default shown:
#   QWEN35_SRC_DIR        -- <this script>/..            (the staged package)
#   QWEN35_NKILIB_DIR     -- <work>/nki-library
#   QWEN35_MODEL_DIR      -- <work>/../Qwen3.5-35B-A3B   (BF16 weights)
#   QWEN35_FP8_MODEL_DIR  -- <model>-FP8                 (FP8 expert weights)

_bench_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the repo-root .env if there is one, searching upward so this works both
# from a git checkout and from a staged src/ tree on a box.
if [ -n "${ENV_FILE:-}" ]; then
  [ -f "$ENV_FILE" ] && . "$ENV_FILE"
else
  _d="$_bench_env_dir"
  while [ "$_d" != "/" ]; do
    if [ -f "$_d/.env" ]; then . "$_d/.env"; break; fi
    _d="$(dirname "$_d")"
  done
  unset _d
fi

SRC="${QWEN35_SRC_DIR:-$(cd "$_bench_env_dir/.." && pwd)}"
WORK="${QWEN35_WORK_DIR:-}"
IMAGE="${QWEN35_NATIVE_IMAGE:-}"

if [ -z "$WORK" ]; then
  echo "error: QWEN35_WORK_DIR is unset -- the scratch root for logs, compiler" >&2
  echo "  caches, nkilib and the staged src (on the Trn2 box: /mnt/nvme/<workdir>)." >&2
  echo "  Set it in the repo-root .env (copy .env.example) or in the environment." >&2
  exit 2
fi

# Scripts that only need the paths (the watchdogs) set BENCH_ENV_REQUIRE_IMAGE=0.
if [ -z "$IMAGE" ] && [ "${BENCH_ENV_REQUIRE_IMAGE:-1}" = "1" ]; then
  echo "error: QWEN35_NATIVE_IMAGE is unset." >&2
  echo "  Set it in the repo-root .env (copy .env.example) or in the environment," >&2
  echo "  e.g. export QWEN35_NATIVE_IMAGE=<acct>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>" >&2
  echo "  Pin by digest (@sha256:...) for anything you intend to quote: the :latest tag" >&2
  echo "  drifts between hosts, and the image is part of the NEFF cache key." >&2
  exit 2
fi

NKILIB="${QWEN35_NKILIB_DIR:-$WORK/nki-library}"
MODEL="${QWEN35_MODEL_DIR:-$(dirname "$WORK")/Qwen3.5-35B-A3B}"
FP8MODEL="${QWEN35_FP8_MODEL_DIR:-${MODEL}-FP8}"

unset _bench_env_dir
