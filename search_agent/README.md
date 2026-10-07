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
Tinker. Results are in [results.md](results.md); setup problems and their fixes in [train_debug.md](train_debug.md).

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

Checkpoints go to `outputs/search_agent/checkpoints/<name>`, eval dumps to `outputs/search_agent/exports/<name>`,
TensorBoard logs to `outputs/search_agent/logs/tensorboard/<name>`:

```bash
.venv/bin/tensorboard --logdir outputs/search_agent/logs/tensorboard
```

Resuming from a checkpoint does not continue training yet; see train_debug.md.

Put exploration notebooks in `search_agent/notebooks/`; the folder is gitignored, so they stay local.
