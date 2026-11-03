#!/usr/bin/env bash
# `docker run IMAGE --model-path ...` validates the GPU stack, then serves.
# `docker run IMAGE <command>` runs that command instead.
set -Eeuo pipefail

if [[ $# -eq 0 || $1 == -* ]]; then
  if [[ "${SGLANG_V100_SKIP_STARTUP_CHECK:-0}" != 1 ]]; then
    python /opt/sglang/scripts/v100/check_v100_env.py
  fi
  exec python -m sglang.launch_server "$@"
fi

exec "$@"
