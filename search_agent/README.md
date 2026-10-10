# Search agent

## Context

Based on Jasper Lu's [Training search agents with GRPO](https://jasperlu.com/blog/training-search-agents-grpo/)
([code](https://github.com/jasper-lu/sec-search-rl)).

- **Task:** given a multi-hop question over SEC filings (Harness-1 data), find and curate every chunk that supports
  the question's 3, 5, or 7 facts.
- **Harness:** lexical tools only (`bm25_search`, `grep_corpus`, `read_document`, `curate`, `drop_curated`,
  `finish`); the curated set is the output.
- **Reward:** F4 (recall weighted 16x over precision), scored per fact: a fact counts if any of its supporting chunks
  is curated. Curating nothing gets -0.2.
- **Training:** gpt-oss-20b with a rank-32 LoRA on Tinker, Dr. GRPO (no std normalization, no KL), 64 queries x 8
  rollouts per step, LR 1e-4.
- **Results:** held-out F1 rose from 0.19 to 0.33, mostly from recall (0.16 to 0.32). Smaller batches (8 queries/step)
  reward-hacked by curating everything.
- **Format penalty:** gpt-oss drifted into malformed Harmony tool calls (99% of episodes); a -0.1 penalty cut them to
  under 1% and lifted F1 to 0.36 (recall 0.43), peaking at 0.39.

This repo trains the same model on the same data with full fine-tuning in SkyRL on 8x A100, instead of a LoRA on
Tinker. Results are in [results.md](results.md); known issues and upgrade notes at the end of this file.

## Data

Run everything from the repo root.

Install dependencies:

```bash
uv sync --locked
```

Download the Harness-1 SEC data and prepare `search_agent/data/queries.parquet`, `search_agent/data/corpus.parquet`, and
the RL training splits `search_agent/data/train.parquet` and `search_agent/data/dev.parquet`:

```bash
uv run python -u search_agent/prepare_data.py
```

The defaults build the 256-query ablation set; add `--train-size 1024` for the full set (see results.md).

Build the BM25 search index `search_agent/data/index.sqlite3` used by the search tools in `search_agent/retrieval.py`:

```bash
uv run python -u search_agent/retrieval.py
```

## Eval

Serve gpt-oss-20b with vLLM on one GPU (ready in about 3.5 minutes), then run the agent on dev queries with `search_agent/trajectory.py`. Both need the `train` dependency group:

```bash
uv sync --locked --group train
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve unsloth/gpt-oss-20b-BF16 --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.85 --port 8000
uv run python -u -m search_agent.trajectory --limit 8
```

The baseline in results.md is 4 trials on each of the first 32 dev queries; `--output` saves every rollout with its
transcript:

```bash
uv run python -u -m search_agent.trajectory --limit 32 --samples 4 --context-length 30720 \
  --output search_agent/data/rollouts/<name>.jsonl
```

Browse saved rollouts in the viewer, then open http://localhost:8081:

```bash
uv run python -u -m search_agent.viewer
```

## Training

`search_agent/train.sh` runs GRPO with SkyRL on 8 GPUs, as Jasper's full run: full fine-tuning, LR 3e-6, 64 queries
x 8 rollouts per step for 24 steps (1.5 epochs), 30,720-token context. Evals run at steps 0, 8, 16 and 24 on his
held-out set: the first 32 dev queries x 4 samples, searched over the 124k-chunk ablation corpus
(`outputs/search_agent/dev32_sec256.parquet`). It needs the `train` dependency group and the SkyRL v0.3.0 checkout
described in `pyproject.toml`. Extra `key=value` arguments override its defaults; its header comments explain the
settings and give a one-step smoke test.

```bash
RUN_NAME=<name> bash search_agent/train.sh
```

Add `search.format_penalty=0.1` to subtract 0.1 from a training trajectory's reward when any of its tool calls has an
off-form Harmony header, as in Jasper's format-penalty run. Add `search.discovery_bonus=0.2
search.curated_chunk_cost=0.02` for his f4s reward: F4 + 0.2 x trajectory recall (the share of facts with a gold
chunk anywhere in the search results) - 0.02 per curated chunk. All three are off by default; eval rewards stay plain
F4.

Checkpoints go to `outputs/search_agent/checkpoints/<name>`, eval dumps to `outputs/search_agent/exports/<name>`,
TensorBoard logs to `outputs/search_agent/logs/tensorboard/<name>`:

```bash
.venv/bin/tensorboard --logdir outputs/search_agent/logs/tensorboard
```

Resuming from a checkpoint does not continue training yet; see below.

Put exploration notebooks in `search_agent/notebooks/`; the folder is gitignored, so they stay local.

## Known issues and upgrade notes

Resume does not continue training. In a test with 1 step per epoch, a run resumed from step 1 with `epochs=2` loaded
the checkpoint, ran eval and exited without training step 2. A likely cause, untested: SkyRL restores the dataloader
at the end of epoch 1, so epoch 2 iterates no batches. Resuming mid-epoch is not tested.

Fixes to recheck when upgrading SkyRL (v0.3.0), vLLM or transformers. Each is commented where it's made; smaller
compatibility shims are commented in `skyrl_patches.py`.

| Problem | Cause | Fix | Where |
|---|---|---|---|
| vLLM servers unreachable | Host has no IPv4, so Ray's node address is IPv6; SkyRL puts it unbracketed in URLs and binds the servers to IPv4 | Ray node address 127.0.0.2, added to `no_proxy` | train.sh |
| Patches don't reach Ray workers, or crash them | Patches must run in every worker; importing SkyRL in a Ray setup hook crashes Ray | `skyrl_patches.py`, which doesn't import SkyRL, as the `worker_process_setup_hook` | train.py, skyrl_patches.py |
| Both vLLM engines on GPUs 0-3 | vLLM assigns each worker its GPUs after start; our setup hook had already initialized CUDA, so the assignment had no effect | `PYTORCH_NVML_BASED_CUDA_CHECK=1`, `FLASHINFER_CUDA_ARCH_LIST=8.0` | train.sh |
| Weight sync silently keeps the old MoE expert weights in vLLM | SkyRL wraps the sync in vLLM's layerwise reload, which vLLM's gpt-oss expert loader bypasses | No-op the layerwise reload | skyrl_patches.py |
| Attention backward ~180x slower | SkyRL adds gpt-oss's attention sinks through a flex-attention score_mod, whose gradient is computed with atomic adds | Attention without sinks, then the sinks applied from the logsumexp | skyrl_patches.py |
| Generation stalls | Tool calls ran on the one event loop shared by all episodes; threads contend for the GIL | Pool of 32 processes (`tool_pool`) | trajectory.py |
| Out of memory in the backward pass on a 30k-token sequence | SkyRL loads the 21 GB/GPU Adam state before the forward and backward passes | Load it just before the optimizer step | skyrl_patches.py |
