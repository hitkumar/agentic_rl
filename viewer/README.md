# Trajectory viewer

A small web viewer for agent runs, shared across projects: pick a run, see its tasks and trials with their scores, and
open a trial to read the conversation (messages, reasoning, tool calls and results) and how it was graded. Each project
plugs in a source module that reads its own run files; the server and page are the same for every project.

## Run it

From the repo root:

```bash
uv run python -u -m viewer.server --source tau_bench
```

then open http://localhost:8081. `--port` changes the port; `--runs-dir` points the source at a different folder of
runs. New runs show up in the run dropdown on reload.

### Viewing from another machine

The server listens only on 127.0.0.1, so to open it from a laptop, forward the port over SSH. In a terminal on the
laptop:

```bash
ssh -N -L 8081:localhost:8081 <server-host>
```

then open http://localhost:8081 in the laptop's browser. Keep that terminal open while browsing. If you started the
viewer with `--port N`, use `-L N:localhost:N` and open http://localhost:N.

## Sources

| Source | Reads | Score |
|---|---|---|
| `tau_bench` | `tau2 run` results, `~/oss/tau2-bench/data/simulations/<run>/results.json` | reward (0 or 1) |

## Adding a project

Write `viewer/sources/<name>.py` and start the viewer with `--source <name>`. The module defines:

- `DEFAULT_RUNS_DIR`: where the project's runs live.
- `run_names(runs_dir)`: the runs, newest first.
- `run_index(runs_dir, run)`: the run's headline numbers, settings, and per-task trial scores (scores between 0 and 1,
  which the page colours from red to green).
- `trial_detail(runs_dir, run, task_id, trial)`: one trial, as stats, task context, a list of messages, and verifier
  sections; `None` if the trial doesn't exist.

The exact JSON shapes are in the docstring at the top of [`server.py`](server.py), and
[`sources/tau_bench.py`](sources/tau_bench.py) is a worked example. The source does all project-specific work:
converting its transcript format into the generic message list (roles `agent`, `user`, `tool`, `system`, each with
text, reasoning, tool calls and an error flag), and turning its grading into verifier sections (tables of
label/value/status rows, or blocks of text). The page needs no changes for a new project.
