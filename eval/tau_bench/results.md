# Results

## Setup

- **Harness:** the official τ-bench code, [`tau2-bench`](https://github.com/sierra-research/tau2-bench) at `v1.0.1`
  (commit `fc0055d`, July 2026), cloned to `~/oss/tau2-bench` with its own environment (`uv sync` there). This is the
  version Artificial Analysis and Sierra's own leaderboard runs use.
- **Agent:** Muse Glimmer 30B at reasoning strength high (the level of Meta's published numbers), temperature 1.0,
  top_p 0.95, top_k 64 (Meta's recommended sampling).
- **Simulated user:** Qwen3.8-27B, thinking at low effort, temperature 1.0, top_p 0.95, top_k 20 (Qwen's recommended
  sampling for thinking). All local, no API models.
- **Serving:** vLLM 0.32 from this folder's environment (`uv sync` here), started by [`serve.sh`](serve.sh): Glimmer on
  GPUs 0-1 (port 8000), Qwen on GPUs 2-3 (port 8001).
- **Grading (airline):** reward is 1 if the database the agent leaves behind matches the target database (the starting
  database after replaying the task's reference solution) and the agent said any required information, else 0. No LLM
  judge.
- **Metric:** pass^1 is the mean reward over all conversations (with 4 trials per task, the same as averaging the 4
  runs' pass rates, which is how Meta reports it).

Meta's published numbers for Glimmer at high reasoning, from the
[Muse Glimmer 30B report](https://arxiv.org/abs/2610.24628), as a reference point: airline 78.0, retail 87.1, telecom
94.1 (Meta's own runs, averaged over 4; appendix A.1), τ³-banking 23.5 (from Artificial Analysis; Table 2).

## Airline baseline: official harness, 50 tasks × 4 trials

All 50 airline tasks, 4 trials each, run on 2026-10-24.

| | pass^1 |
|---|---|
| Glimmer, high reasoning (ours) | 0.835 |
| Meta's report | 0.780 |

The 4 runs scored 0.86, 0.78, 0.84 and 0.86. 35 tasks passed all 4 trials, 12 passed some, and 3 (tasks 32, 39 and 44)
failed all 4. This is the baseline our own harness will be compared against, task by task.

To reproduce, start the servers with `bash eval/tau_bench/serve.sh` from the repo root, then from `~/oss/tau2-bench`:

```
.venv/bin/tau2 run --domain airline --num-trials 4 --max-concurrency 10 \
  --agent-llm hosted_vllm/glimmer --agent-llm-args '{"api_base": "http://localhost:8000/v1", "temperature": 1.0, "top_p": 0.95}' \
  --user-llm hosted_vllm/qwen-user --user-llm-args '{"api_base": "http://localhost:8001/v1", "temperature": 1.0, "top_p": 0.95}' \
  --save-to glimmer_airline_50x4
```

It took about an hour.

## Browsing runs

Results go to `~/oss/tau2-bench/data/simulations/<save-to>/results.json`. Browse them with the trajectory viewer (from
the repo root), then open http://localhost:8081:

```
uv run python -u -m viewer.server --source tau_bench
```
