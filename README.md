# Agentic RL

Experiments and training infrastructure for agentic reinforcement learning.

## Projects

- [`search_agent/`](search_agent/README.md): GRPO training of gpt-oss-20b as a search agent over SEC filings,
  reproducing Jasper Lu's [Training search agents with GRPO](https://jasperlu.com/blog/training-search-agents-grpo/)
  with full fine-tuning in SkyRL on 8x A100. Results in [search_agent/results.md](search_agent/results.md).
- [Harbor](#harbor): agent benchmarks in task containers, with notes from the DGX Spark setup in
  [harbor.md](harbor.md), and an exporter from Harbor trajectories to Hugging Face traces.

## Python environment

This project follows [Prime-RL's environment pattern](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/pyproject.toml):
Python 3.12, uv dependency management, and an editable package under `src/`.
Dependencies come in two groups besides the base set: `dev` (pytest, Ruff, pandas, ipykernel), installed by default,
and `train` (SkyRL, vLLM, openai-harmony), used by `search_agent/` for its evals and training.

With [uv installed](https://docs.astral.sh/uv/getting-started/installation/), run
these commands from the project root:

```bash
uv sync --locked
uv run python -V
uv run python -c "import agentic_rl; print(agentic_rl.__file__)"
```

`uv sync --locked` creates `.venv` and installs the versions recorded in `uv.lock`,
including the development tools and this package in editable mode. It reports an
error if the lockfile and project requirements disagree.

Use `uv run` to run commands inside the environment. For an interactive shell,
activation is optional:

```bash
source .venv/bin/activate
```

Point your editor at `.venv/bin/python`.

## Managing dependencies

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for
development tools. These commands update `pyproject.toml`, `uv.lock`, and `.venv`
together. After editing dependencies in `pyproject.toml` manually, run `uv sync`
to regenerate the lockfile and update the environment.

Commit `pyproject.toml`, `uv.lock`, and `.python-version`; `.venv` and caches are
ignored. See [uv's project documentation](https://docs.astral.sh/uv/concepts/projects/layout/)
for how these files work together.

The `train` group installs SkyRL from a local checkout of its v0.3.0 release, not PyPI, and mirrors SkyRL's CUDA
wheel indexes and overrides (Linux, x86_64 only). Clone it once before syncing the group:

```bash
git clone --branch skyrl-v0.3.0 https://github.com/NovaSky-AI/SkyRL ~/oss/SkyRL-v0.3.0
uv sync --locked --group train
```

`pyproject.toml` comments explain the pins; re-sync them from SkyRL's `pyproject.toml` when bumping it.

## Harbor

[Harbor](https://github.com/harbor-framework/harbor) is used to run agent
benchmarks and inspect their results. Install it as a standalone tool:

```bash
uv tool install harbor
harbor --version
```

Harbor uses Docker for local task environments. Check that Docker is running
and accessible:

```bash
docker info
```

Run Harbor's hello-world task with its reference-solution agent:

```bash
harbor run -d harbor/hello-world -a oracle
```

The command writes its results to `jobs/`. Open Harbor's local results viewer
with:

```bash
harbor view jobs
```

### Hugging Face trace export

Convert Harbor's ATIF `trajectory.json` into the JSONL format used by the
[Hugging Face Agent Traces viewer](https://huggingface.co/docs/hub/agent-traces):

```bash
uv run atif-to-hf-trace jobs/<job>/<trial>/agent/trajectory.json
```

The default output is `trajectory.hf.jsonl` beside the input file. Upload that
file to a Hugging Face dataset or Storage Bucket to open the session timeline,
messages, reasoning, tool calls, and tool results in the web viewer. Use `-o`
to choose another output path and `--name` to set its display name.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
```

Once tests are added, run them with `uv run pytest`.
