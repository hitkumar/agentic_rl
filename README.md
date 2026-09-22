# Agentic RL

Experiments and training infrastructure for agentic reinforcement learning.

## Python environment

This project follows [Prime-RL's environment pattern](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/pyproject.toml):
Python 3.12, uv dependency management, and an editable package under `src/`.
The initial environment includes pytest and Ruff. Training dependencies will be
added as the training and inference backends are chosen.

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

For GPU dependencies, select a compatible set of PyTorch, CUDA, vLLM, and any
attention kernels for the target GPU and CPU architecture before adding them.
Prime-RL defines its own CUDA indexes, wheel URLs, and dependency overrides, so
its full dependency list needs review before reusing it in this project.

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
