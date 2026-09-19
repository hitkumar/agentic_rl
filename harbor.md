# Harbor benchmark notes

This document records the local Harbor setup and the benchmark work completed on
the DGX Spark. Commands assume they are run from the repository root.

## Goal

Use Harbor to:

- run agent benchmarks in isolated task containers;
- compare local models with a consistent agent harness;
- save complete trajectories for inspection and future RL work; and
- establish a small, fast workflow before running full benchmark suites.

## Machine and installed software

- NVIDIA DGX Spark with one GB10 GPU
- ARM64 Ubuntu 24.04
- 128 GB unified memory, reported as 121.6 GiB by Docker
- Approximately 273 GB/s memory bandwidth
- Docker Engine 29.6.2
- Harbor 0.23.0, installed as a `uv` tool

Install and verify Harbor:

```bash
uv tool install harbor
harbor --version
docker info
```

Harbor writes job artifacts under `jobs/`. That directory is intentionally in
`.gitignore` because it contains generated logs, recordings, trajectories, and
results that can become large.

## Docker access

Docker is running normally. The initial `permission denied` error for
`/var/run/docker.sock` occurred because the user session had not picked up its
new `docker` group membership.

The permanent setup is:

```bash
sudo usermod -aG docker "$USER"
```

Log out and back in after running it. `newgrp docker` can update the current
terminal immediately. A process that was started before the group change can
also run an individual command through the group:

```bash
sg docker -c 'docker info'
```

A separate script to start Docker is unnecessary. Docker starts as a system
service; the relevant checks are whether the daemon is running and whether the
current user can access its socket.

## Harbor hello-world check

Run Harbor's built-in task with the reference solution:

```bash
harbor run -d harbor/hello-world -a oracle
```

This completed successfully with reward `1.0`.

In Harbor, `oracle` means the benchmark's known reference solution. It verifies
that the task container and grader work; it is not a model or a general-purpose
agent.

Open the results viewer with:

```bash
harbor view jobs
```

## Agent, harness, rollout, and trajectory

For these experiments:

- **Model server:** vLLM exposes the model through an OpenAI-compatible API.
- **Agent harness:** Terminus-2 sends prompts to the model, executes its terminal
  commands, returns observations, and repeats.
- **Rollout:** one complete attempt by the agent on one task.
- **Trajectory:** the stored sequence of prompts, reasoning, tool calls,
  observations, token IDs, log probabilities, and metrics from that rollout.

Terminus-2 is therefore the harness around the model. The model produces the
decisions; Terminus-2 manages the interaction loop.

Qwen returns internal reasoning separately from its final JSON response.
Terminus stores these parts as follows:

- `reasoning_content` is Qwen's internal reasoning, extracted by vLLM's Qwen
  reasoning parser.
- `message` combines the JSON response's `analysis` and `plan` fields into
  readable text.
- `tool_calls` contains the JSON response's `commands`, converted by Terminus
  into standardized `bash_command` records. These are not native API tool calls.
- `observation` is the terminal output captured from `tmux` after Terminus runs
  those commands.

On the next turn, Qwen receives the original conversation, its previous raw
JSON response as an assistant message, and the observation as a new user
message. The previous `reasoning_content` is omitted unless Terminus is run with
`interleaved_thinking=true`.

## Qwen3.8-27B vLLM server

The model currently tested is the original BF16
`Qwen/Qwen3.8-27B` checkpoint. It is served under the short API name
`Qwen3.8-27B` because Harbor's `hosted_vllm` adapter expects a model identifier
in the form `hosted_vllm/<simple-name>`.

The server command used was:

```bash
docker run -d \
  --name qwen38-vllm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  nvcr.io/nvidia/vllm:26.08-py3 \
  vllm serve Qwen/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 4 \
    --reasoning-parser qwen3
```

The NVIDIA container contains vLLM `0.27.1+93523f72.dev`. The Hugging Face and
vLLM caches are mounted from the host so downloaded weights and compiled kernels
survive container replacement. Restarting the server still reloads the weights
from disk into unified memory.

Check the server:

```bash
docker ps
docker logs -f qwen38-vllm
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

The configured 65,536-token context is a selected limit, not a hard-coded model
limit. It provides enough context for the smoke test while leaving memory for
the model and multiple sequences.

Observed resource use:

- model weights: approximately 51.1 GiB;
- KV cache: approximately 27.9 GiB;
- KV capacity: approximately 439,000 tokens; and
- single-request BF16 decode: approximately 3.8-3.9 tokens/second.

The slow decode rate is primarily a memory-bandwidth limit. Generating each new
token requires reading most of the dense 51 GiB model. The 128 GB memory capacity
is sufficient, and the 64K context setting is not the cause of the delay.

## ARM64 task containers

The Terminal-Bench `write-compressor` task image is AMD64, while the DGX Spark
host is ARM64. The first real trial failed with a platform mismatch. AMD64
emulation was registered with:

```bash
docker run --privileged --rm tonistiigi/binfmt --install amd64
docker run --rm --platform linux/amd64 alpine:3.22 uname -m
```

The verification command should print `x86_64`. Registration normally needs to
be repeated after reboot.

QEMU only runs the AMD64 benchmark task container. It does not run vLLM, which
uses native ARM64 and CUDA. QEMU can make task setup and shell commands slower,
but it does not explain the model's 3.9-token/second generation rate.

## Terminal-Bench 2.1 smoke run

The command that reached the agent loop was:

```bash
harbor run \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model hosted_vllm/Qwen3.8-27B \
  --agent-kwarg api_base=http://127.0.0.1:8000/v1 \
  --agent-kwarg reasoning_effort=xhigh \
  --agent-kwarg temperature=1.0 \
  --agent-kwarg collect_rollout_details=true \
  --agent-kwarg 'model_info={"max_input_tokens":65536,"max_output_tokens":65536,"input_cost_per_token":0.0,"output_cost_per_token":0.0}' \
  --n-tasks 1 \
  --n-concurrent 1 \
  --job-name qwen38-27b-tbench21-smoke-02 \
  --allow-agent-host 127.0.0.1 \
  --yes
```

Important details:

- `hosted_vllm/Qwen3.8-27B` must match the server's short model name.
- `api_base` points Terminus-2 at the local OpenAI-compatible API.
- `model_info` tells Harbor the context limits and prevents it from trying to
  look up commercial token prices.
- `collect_rollout_details=true` saves token IDs and log probabilities in the
  trajectory.
- `--n-tasks 1 --n-concurrent 1` makes this a smoke test, not a benchmark score.

Harbor selected the `write-compressor` task. The environment started correctly
after AMD64 emulation was enabled. Qwen completed two agent turns:

| Turn | Prompt tokens | Completion tokens | Tool calls |
| ---: | ---: | ---: | ---: |
| 1 | 847 | 237 | 2 |
| 2 | 1,424 | 266 | 1 |

The run produced 2,271 prompt tokens and 503 saved completion tokens. A third
response was still generating when Harbor raised `AgentTimeoutError`. The trial
therefore received reward `0.0`. This one failed task is not a meaningful model
score.

Inspect the run:

```bash
less jobs/qwen38-27b-tbench21-smoke-02/result.json
less jobs/qwen38-27b-tbench21-smoke-02/write-compressor__zGCubTE/agent/trajectory.json
less jobs/qwen38-27b-tbench21-smoke-02/write-compressor__zGCubTE/agent/terminus_2.pane
```

The terminal recording is:

```text
jobs/qwen38-27b-tbench21-smoke-02/write-compressor__zGCubTE/agent/recording.cast
```

The timeout happened because `reasoning_effort=xhigh` allowed a long third
response while BF16 generation ran at only 3.9 tokens/second. For another Qwen
smoke test, use a smaller output limit and a lower reasoning effort before
increasing either setting for a full evaluation.

## Model choices considered

### Qwen3.5-4B

Qwen3.5-4B is the practical candidate for early RL experiments. Estimated BF16
decode on this Spark is 20-30 tokens/second, and the weights require roughly
8-10 GB. It is small enough to generate many rollouts and to experiment with
fine-tuning. Qwen publishes general benchmarks for its smaller models, but not
official Terminal-Bench 2.1 results.

### Laguna S 2.1

Poolside publishes an [official DGX Spark recipe](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
for Laguna S 2.1 using:

- `poolside/Laguna-S-2.1-NVFP4`;
- the `poolside/Laguna-S-2.1-DFlash-NVFP4` draft model;
- vLLM with native NVFP4 kernels;
- 262,144-token context;
- `--gpu-memory-utilization 0.85`;
- `--max-num-seqs 32`; and
- seven speculative tokens per step.

The two checkpoints total approximately 73 GB. Poolside reports 600-800
tokens/second for prefill, around 15 tokens/second for prose decode, and 22-24
tokens/second for code decode on one DGX Spark. This makes it a good local model
for capable benchmark baselines and trajectory generation.

Poolside's reported 70.2% Terminal-Bench 2.1 result is not a Terminus-2 result.
They used Harbor with their own `pool` agent harness, a maximum of 500 steps,
four attempts per task, and an internal sandbox service. They also patched some
task images and verifiers. Poolside publishes the
[final trajectories](https://trajectories.poolside.ai/), but those differences
prevent an exact reproduction with a stock Harbor command.

For direct model comparisons here, keep the dataset and harness fixed. A useful
sequence is:

1. Run Qwen3.5-4B with Terminus-2 to develop the rollout and RL pipeline.
2. Run Laguna S 2.1 with Terminus-2 as the stronger local baseline.
3. Run Laguna S 2.1 with `pool` separately to measure the effect of its native
   harness.

Qwen3.8-27B BF16 remains useful as a fidelity reference, but its current decode
speed makes it inefficient for repeated local agent rollouts.

## Current status

- Docker works and the user has Docker access.
- Harbor hello-world passes.
- Terminal-Bench AMD64 task images run through QEMU on the ARM64 host.
- Qwen3.8-27B is served successfully through vLLM.
- Harbor captures detailed Terminus-2 trajectories.
- The first Qwen Terminal-Bench smoke trial reached the agent loop but timed
  out during its third model response.
- No full benchmark score has been produced yet.
