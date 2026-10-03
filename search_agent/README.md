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

## Data

Run from the repo root.

Install dependencies:

```bash
uv sync --locked
```

Download the Harness-1 SEC data and prepare `search_agent/data/queries.parquet` and `search_agent/data/corpus.parquet`:

```bash
uv run python -u search_agent/prepare_data.py
```

Build the BM25 search index `search_agent/data/index.sqlite3` used by the search tools in `search_agent/retrieval.py`:

```bash
uv run python -u search_agent/retrieval.py
```

Serve gpt-oss-20b with vLLM on one GPU (ready in about 3.5 minutes), then run the agent on dev queries with `search_agent/trajectory.py`. Both need the `train` dependency group:

```bash
uv sync --locked --group train
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve unsloth/gpt-oss-20b-BF16 --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.85 --port 8000
uv run python -u -m search_agent.trajectory --limit 8
```

Put exploration notebooks in `search_agent/notebooks/`; the folder is gitignored, so they stay local.
