#!/usr/bin/env bash
# Start the two vLLM servers for local τ-bench runs, each detached (setsid nohup) so it outlives the shell:
#   agent: Muse Glimmer 30B on GPUs 0-1, port 8000, served as "glimmer"
#   user simulator: Qwen3.8-27B on GPUs 2-3, port 8001, served as "qwen-user"
#
# Glimmer reads its reasoning level from a "Reasoning strength: <level>." line in the system prompt. The harness's
# system prompt doesn't have one, so Glimmer's chat template appends it, filling in the template input
# reasoning_strength. --default-chat-template-kwargs passes that input on every request, set from REASONING_STRENGTH
# (default high, the level of Meta's published τ-bench numbers), e.g. REASONING_STRENGTH=low bash
# eval/tau_bench/serve.sh. Tool calls and reasoning use vLLM's muse_glimmer parsers.
#
# The user simulator thinks before each reply at low effort, as Sierra's leaderboard runs do with gpt-5.2 as the user:
# Qwen's chat template turns reasoning_effort into a "keep your thinking brief" instruction. USER_REASONING_EFFORT sets
# it (low, medium or xhigh; off turns thinking off). The qwen3 reasoning parser keeps the thinking out of the reply text, so the agent never sees
# it. No tool parser: in airline and retail the simulated user has no tools.
#
# Run from the repo root: bash eval/tau_bench/serve.sh. Logs go to outputs/tau_bench/logs/{glimmer,qwen-user}.log and
# process IDs to {glimmer,qwen-user}.pid; a server is ready when GET /health on its port returns 200. Stop both with:
#   pkill -f "vllm serve"
set -euo pipefail

VLLM=eval/tau_bench/.venv/bin/vllm
LOG_DIR=outputs/tau_bench/logs
REASONING_STRENGTH=${REASONING_STRENGTH:-high}
USER_REASONING_EFFORT=${USER_REASONING_EFFORT:-low}
if [ "$USER_REASONING_EFFORT" = off ]; then
  USER_TEMPLATE_KWARGS='{"enable_thinking": false}'
else
  USER_TEMPLATE_KWARGS="{\"enable_thinking\": true, \"reasoning_effort\": \"$USER_REASONING_EFFORT\"}"
fi
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0,1 setsid nohup "$VLLM" serve meta-models/Muse-Glimmer-30B \
  --served-model-name glimmer --port 8000 --tensor-parallel-size 2 --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --default-chat-template-kwargs "{\"reasoning_strength\": \"$REASONING_STRENGTH\"}" \
  > "$LOG_DIR/glimmer.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/glimmer.pid"

CUDA_VISIBLE_DEVICES=2,3 setsid nohup "$VLLM" serve Qwen/Qwen3.8-27B \
  --served-model-name qwen-user --port 8001 --tensor-parallel-size 2 --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs "$USER_TEMPLATE_KWARGS" \
  > "$LOG_DIR/qwen-user.log" 2>&1 < /dev/null &
echo $! > "$LOG_DIR/qwen-user.pid"

echo "Started glimmer (port 8000) and qwen-user (port 8001); logs in $LOG_DIR"
