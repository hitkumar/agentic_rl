#!/usr/bin/env bash

set -euo pipefail

OPENCODE_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENCODE_LOCAL_CONFIG="$OPENCODE_WORKSPACE/opencode-local.json"
OPENCODE_LOCAL_API_URL="http://127.0.0.1:8000/v1"

if ! command -v opencode >/dev/null 2>&1; then
  echo "Error: opencode is not installed or is not on PATH." >&2
  echo "Install it with: npm install -g opencode-ai" >&2
  exit 1
fi

if ! curl --fail --silent --show-error --max-time 10 \
  "$OPENCODE_LOCAL_API_URL/models" >/dev/null; then
  echo "Error: the local model server is not responding at $OPENCODE_LOCAL_API_URL." >&2
  echo "Start it with: docker start laguna-vllm" >&2
  exit 1
fi

export OPENCODE_CONFIG="$OPENCODE_LOCAL_CONFIG"

exec opencode "$OPENCODE_WORKSPACE" "$@"
