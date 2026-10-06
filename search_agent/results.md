# Results

## Baseline: gpt-oss-20b before training (Jasper's "Initial explorations")

### Setup

Every run: 4 trials on each of the 32 eval queries (the first 32 dev queries), `unsloth/gpt-oss-20b-BF16` on one vLLM
server, temperature 1.0, 40 turns, 2,048 generation tokens per reply, ablation data (124,395 chunks). The developer
message (Jasper's instructions and tool definitions) is the same in every run. Rollouts with transcripts are in
`search_agent/data/rollouts/` (gitignored); browse them with `uv run python -u -m search_agent.viewer`.

| Run | System message | Context length | Rollouts file |
|---|---|---|---|
| ours-65k | ours | 65,536 | `baseline_eval32_x4.jsonl` |
| ours-30k | ours | 30,720 | `ours_30k_eval32_x4.jsonl` |
| jasper-30k | Jasper's | 30,720 | `jasper_sysprompt_30k_eval32_x4.jsonl` |

- Our system message is gpt-oss's standard header (`You are ChatGPT…`, `Reasoning: medium`, `# Valid channels…`) plus
  the tool-routing line. Jasper's renderer (`gpt_oss_no_sysprompt`) sends only the tool-routing line.
- Context length is the cap on prompt plus responses (`--context-length`). 30,720 is what Jasper's eval uses.

To reproduce ours-30k:

```
uv run python -u -m search_agent.trajectory --limit 32 --samples 4 --context-length 30720 \
  --output search_agent/data/rollouts/ours_30k_eval32_x4.jsonl
```

ours-65k is the same without `--context-length` (65,536 is the default). jasper-30k also replaced the system message
with Jasper's, through a one-off script that is not in the repo.

### Metrics

| | ours-65k | ours-30k | jasper-30k | Jasper's blog |
|---|---|---|---|---|
| best-of-4 F1 | 0.428 | 0.379 | 0.345 | 0.323 |
| best-of-4 precision | 0.688 | 0.620 | 0.578 | 0.490 |
| best-of-4 recall | 0.335 | 0.302 | 0.273 | 0.269 |
| trial 1 F1 | 0.214 | 0.248 | 0.249 | 0.195 |
| mean F1, all 128 rollouts | 0.276 | 0.249 | 0.221 | 0.193 |
| mean F1, 3-fact queries (15) | 0.370 | 0.356 | 0.316 | 0.286 |
| mean F1, 5-fact queries (17) | 0.192 | 0.155 | 0.136 | 0.111 |
| mean reward (F4) | 0.169 | 0.120 | 0.084 | |
| turns per rollout | 18.9 | 17.8 | 17.7 | |
| queries with F1 0 in all 4 trials | 9/32 | 11/32 | 13/32 | |
| queries with nothing curated in any trial | 0/32 | 5/32 | 6/32 | 7/32 |
| rollouts with nothing curated | 29 | 51 | 58 | |

- F1 is scored on the curated set: precision = curated chunks that are gold for any fact / chunks curated; recall =
  facts with at least one gold chunk curated / facts.
- Best-of-4 takes each metric's maximum over a query's 4 trials independently, then averages over queries, as in
  Jasper's `evaluate.py`. So best-of-4 F1 is not computed from best-of-4 precision and recall.
- Trial 1 is a single sample per query, so it is noisy.

### Why ours-65k beats the blog

Best-of-4 F1 is 0.428 against the blog's 0.323, a gap of 0.105:

| Change | best-of-4 F1 | Accounts for |
|---|---|---|
| ours-65k | 0.428 | |
| context 65,536 → 30,720 (ours-30k) | 0.379 | 0.049 |
| our system message → Jasper's (jasper-30k) | 0.345 | 0.034 |
| Jasper's blog | 0.323 | 0.022, unexplained |

The remaining 0.022 is within sampling noise for 4 trials on 32 queries. In ours-65k, 26 rollouts used more than
30,720 tokens, and most of them scored 0.

Checked and ruled out:

- Queries: all 32 blog rows map to our 32 eval queries.
- Scoring: Jasper's `score_submission` matches `rewards.score`, and his qrels are the union of fact chunks, as ours.
- Tool limits: snippet 220 chars, read 4,000 chars, 30 curated, k 10/25, 40 turns, and 2,048 generation tokens all
  match.

## Overfit check: 32 queries, full fine-tuning with SkyRL

TLDR: the pipeline learns. Training reward rose from 0.09 to 0.33 in 4 steps, then plateaued at about 0.30–0.33 through
step 7, below the 0.43 best-of-4 reward of the selected queries. Stopped after 7 of 30 planned steps.

Setup: the 32 train queries with the highest reward std among 64 baseline queries (4 trials each), as in Mercor's
overfit test. 8 rollouts per query, batch 32, one optimizer step per batch (so one step per epoch), LR 1e-5, full
fine-tuning of `unsloth/gpt-oss-20b-BF16` on 8x A100, 30,720-token context. Run: `overfit32`.

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| reward (F4) | 0.087 | 0.102 | 0.281 | 0.334 | 0.325 | 0.335 | 0.295 |
| F1 | 0.205 | 0.207 | 0.297 | 0.313 | 0.282 | 0.294 | 0.273 |
| turns | 18.8 | 8.1 | 10.4 | 16.7 | 18.4 | 20.0 | 18.3 |
| ended by plain reply (no finish call) | 0.08 | 0.42 | 0.68 | 0.57 | 0.33 | 0.41 | 0.56 |
| trainer vs vLLM logprob, mean abs diff | 0.026 | 0.030 | 0.031 | 0.026 | 0.025 | 0.025 | 0.027 |
| step time (s) | 1,080 | 450 | 412 | 650 | 940 | 871 | 747 |

- The logprob diff sits at Mercor's "below 0.03 is healthy" mark and doesn't grow, so weight sync is sound.
- More episodes end with a plain reply instead of `finish`. That scores the same as `finish`, so it isn't penalized.

Fixes that made the run possible (all in `skyrl_patches.py` except the tool pool):

| Problem | Fix | Effect |
|---|---|---|
| Attention backward ~180x slower with gpt-oss's attention sinks in SkyRL's flex attention | Compute attention without sinks, then apply them from the logsumexp | Training step practical at 30k tokens |
| Tool calls ran on the event loop shared by all 256 episodes, stalling generation | Run them in a pool of 32 processes (`trajectory.tool_pool`) | Step-1 generation 818 s → 507 s |
| Out of memory at step 3: SkyRL loads the 21 GB/GPU Adam state before the forward and backward passes | Load it just before the optimizer step | Peak fits; 30.9k-token sequences train with Adam state allocated |
