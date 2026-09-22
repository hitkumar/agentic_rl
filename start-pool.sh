#!/usr/bin/env bash

set -euo pipefail

POOL_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
POOL_API_URL="http://127.0.0.1:8000/v1"

export POOLSIDE_STANDALONE_BASE_URL="$POOL_API_URL"
export POOLSIDE_API_KEY="EMPTY"
export POOLSIDE_STANDALONE_MODEL="Laguna-S-2.1-NVFP4"
export POOLSIDE_STANDALONE_CONTEXT_LENGTH="49152"

if ! command -v pool >/dev/null 2>&1; then
  echo "Error: pool is not installed or is not on PATH." >&2
  exit 1
fi

if ! curl --fail --silent --show-error --max-time 10 \
  "$POOL_API_URL/models" >/dev/null; then
  echo "Error: Laguna is not responding at $POOL_API_URL." >&2
  echo "Start it with: docker start laguna-vllm" >&2
  exit 1
fi

exec pool --mode always-allow --directory "$POOL_WORKSPACE" "$@"
