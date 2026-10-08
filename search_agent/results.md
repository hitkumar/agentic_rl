# Results

## Data

Both datasets are identical to Jasper's (same query ids, query text, gold facts, chunk ids and chunk text), built with
`prepare_data.py` (seed 42) and gitignored:

| Dataset | Train queries | Dev queries | Corpus chunks | Location | Built with |
|---|---|---|---|---|---|
| Full | 1,024 | 64 | 237,533 | [`data/`](data/) | `--train-size 1024 --dev-size 64` |
| Ablation | 256 | 64 | 124,395 | [`data/sec_256/`](data/sec_256/) | `--train-size 256 --dev-size 64` (default) |

The 64 dev queries are the same in both. The baseline, LR sweep and full run evaluate on the first 32, searched over
the ablation corpus, as Jasper's held-out eval (the sweep via `outputs/search_agent/dev32.parquet`, the full run via
`dev32_sec256.parquet`, which points its episodes at `data/sec_256/` while training uses `data/`). Otherwise the code
reads `data/`, so to train or run baselines on the ablation set, swap its files into `data/`. The baseline, overfit
check and LR sweep below used the ablation set.

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

## LR sweep: 1e-6, 3e-6, 1e-5

TLDR: use 3e-6. It has the highest eval reward from step 4 on (0.258 at step 6, base 0.158) and is the only LR that
stopped curating nothing without collapsing; its F1 gain (0.283 → 0.324) is within eval noise. 1e-6 barely moved;
1e-5 collapsed to ~5-turn episodes with F1 0.11.

Setup, as Jasper's sweep: one epoch of the 256 train queries, 32 queries x 8 rollouts per step (8 steps), full
fine-tuning, otherwise as the overfit check. Eval every 2 steps on the 32 eval queries x 4 samples (128 rollouts).
Runs `lr_1.0e-6`, `lr_3.0e-6`, `lr_1.0e-5`, launched by `outputs/search_agent/lr_sweep/run.sh`.

Eval, mean over the 128 eval rollouts. Reward is F4. Base model: F1 0.283, reward 0.158.

| LR | step 2: F1, reward | step 4: F1, reward | step 6: F1, reward | step 8: F1, reward |
|---|---|---|---|---|
| 1e-6 | 0.242, 0.125 | 0.287, 0.166 | 0.265, 0.163 | 0.228, 0.119 |
| 3e-6 | 0.236, 0.130 | 0.295, 0.226 | **0.324, 0.258** | 0.290, 0.252 |
| 1e-5 | 0.163, 0.129 | 0.126, 0.103 | 0.119, 0.129 | 0.108, 0.144 |

Eval behaviour at step 8:

| | base | 1e-6 | 3e-6 | 1e-5 |
|---|---|---|---|---|
| turns | 19.2 | 13.2 | 20.0 | 5.2 |
| ended by `finish` | 0.58 | 0.67 | 0.90 | 0.95 |
| nothing curated | 0.32 | 0.34 | 0.01 | 0.02 |
| trainer vs vLLM logprob diff, max over training | | 0.028 | 0.029 | 0.033 |

- 3e-6 learned to search and then curate: nothing-curated fell from 0.32 to 0.01 while turns stayed ~20.
- 1e-5 stopped searching (~1.6 searches per episode at step 4, against ~6 at 3e-6) and curated worse chunks. Its
  logprob diff passed 0.03 at steps 4–6.
- 3e-6 train reward fell over the last three steps (0.236 → 0.135) while eval held; 1e-6 also dropped on the last
  batch. Possibly harder batches, but untested; the cause is unknown.

Conclusions:

- 3e-6 is the best of the three. Its reward leads at steps 4, 6 and 8 (0.226–0.258, against at most 0.166 for the
  others), and nothing-curated fell from 0.32 to 0.01. Its reward lead is beyond eval noise at steps 6 and 8 and
  borderline at step 4; the fall in nothing-curated is well beyond it.
- Most of the reward gain is the model learning not to curate nothing, which scores -0.2. That is real learning, but
  F1 has not yet clearly improved.
- Eval noise: two evals of the same model differ by up to 0.047 F1 (95th percentile, from resampling the two
  base-model evals). The base model scored 0.249 in ours-30k and 0.283 here; same queries, prompt and sampling, so
  the gap is sampling noise. 3e-6's F1 gain over the base model (0.041) is below that bound.
- Not tested: stability past 8 steps, and LRs between 3e-6 and 1e-5. 1e-5 broke within 2–4 steps, so a long run at
  3e-6 should be watched for falling turns.
- The full run uses 3e-6, now `train.sh`'s default, with 64 queries per step instead of 32. The LR was tuned at 32;
  the larger batch should only make steps less noisy, but that is untested.

## Full run: 1,024 queries, LR 3e-6

TLDR: training works at full scale. Held-out F1 rose from 0.256 to 0.438 and reward (F4) from 0.131 to 0.424 over 24
steps, above Jasper's plain-F4 run (F4 0.316) and level with his best recipe, F4 plus a format penalty (0.418 at step
24). Side effect: its `finish` calls broke; 99% of eval episodes end with a plain reply after repeated failed
`finish` calls.

Setup, as Jasper's full run: 1,024 train queries over the 237,533-chunk corpus, 64 queries x 8 rollouts per step for 24
steps (1.5 epochs), LR 3e-6, full fine-tuning, otherwise as the LR sweep. Eval at steps 0, 8, 16 and 24 on his
held-out set: the first 32 dev queries x 4 samples (128 rollouts), searched over the 124,395-chunk ablation corpus.
Run `full_lr3e-6`, launched by `RUN_NAME=full_lr3e-6 bash search_agent/train.sh`; about 10 hours on 8x A100.

Eval, mean over the 128 eval rollouts. Reward is F4. Base model (step 0): F1 0.256, reward 0.131.

| Run | step 8: F1, reward | step 16: F1, reward | step 24: F1, reward |
|---|---|---|---|
| ours, LR 3e-6 | 0.251, 0.213 | 0.368, 0.350 | **0.438, 0.424** |
| Jasper, plain F4 (blog F1, repo F4) | 0.245, – | 0.253, 0.202 | 0.302, 0.316 |
| Jasper, F4 + format penalty (repo F4) | – | –, 0.353 | –, 0.418 |

Jasper's base model: F1 0.195 (blog), F4 0.166 (repo). His repo F4 is the mean of two evals of 32 queries x 1 sample.

Eval behaviour:

| | step 0 | step 8 | step 16 | step 24 |
|---|---|---|---|---|
| turns | 18.7 | 19.2 | 27.1 | 23.9 |
| nothing curated | 0.39 | 0.02 | 0.03 | 0.00 |
| ended by `finish` | 0.56 | 0.87 | 0.68 | 0.01 |
| ended by plain reply (no tool call) | 0.09 | 0.11 | 0.20 | 0.99 |
| ended by malformed tool call | 0.11 | 0.02 | 0.00 | 0.00 |

Train reward by step (64 queries x 8 rollouts each, a different batch every step):

| Step | 1 | 4 | 8 | 12 | 16 | 20 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|
| reward | 0.041 | 0.163 | 0.184 | 0.288 | 0.254 | 0.342 | 0.419 | 0.366 |

Conclusions:

- Training went in three phases. Steps 1–8: the model stopped curating nothing (eval 0.39 → 0.02), which lifted
  reward but not F1; among episodes that curated something, F4 fell, because queries it used to give up on now get
  weak guesses. Steps 8–12: episodes got longer (train turns 17.5 → 26). Steps 12–24: it found more gold chunks, and
  held-out F1 rose 0.25 → 0.44.
- The F1 gain over the base model (0.182) is about four times the eval noise (0.047); the reward gain (0.293) is
  well beyond its 0.053.
- The two base models score about the same: ours is higher on F1 (0.249–0.283 across three evals, against 0.195) and
  lower on F4 (0.120–0.158, against 0.166), both gaps near eval noise. One harness difference may favour his base F4,
  untested: his parser runs tool calls with malformed headers, ours ends the episode, often before anything is
  curated.
- Its `finish` calls broke: from step 8 the model calls `finish` about 6 times per episode with arguments that are
  not valid JSON (mostly `{""}`), which our harness rejected at no cost, then ends with a plain reply (0.09 → 0.99).
  Plain F4 scores every ending the same, so nothing pushed back. Jasper's harness ends such an episode at -0.2; the
  next run adopts that rule.
- The trainer vs vLLM logprob diff rose slowly from 0.026 to 0.035 over the run, with no collapse (turns, reward and
  malformed calls stayed healthy).
- One training run, one seed; run-to-run variance is not measured.

## Format penalty and Jasper's harness rules

TLDR: the fixes made the model end cleanly at no cost to search quality. At step 24, 98% of eval episodes end with a
successful `finish` call (previous run: 1%) and 1% have an off-form tool-call header (98%); held-out F1 and reward
match the previous run within eval noise (0.448 vs 0.438, 0.430 vs 0.424), level with Jasper's format-penalty run
(F4 0.418). Unlike his, the penalty did not raise recall.

Setup: as the previous run (`full_lr3e-6`), with two changes:

- Harness, matching Jasper's: an episode ends at a flat -0.2, curated set unscored, when a tool call's arguments are
  not valid JSON (e.g. `finish` with `{""}`) or a reply is cut off at 2,048 tokens; unknown argument keys are dropped.
  This applies to evals too.
- Format penalty (training only): -0.1, once per episode, if any tool call's header is not the canonical
  `<|channel|>commentary to=functions.X <|constrain|>json<|message|>`. Eval rewards stay F4 without it.

Run `base_format_base_fp_harness`, launched by
`RUN_NAME=base_format_base_fp_harness bash search_agent/train.sh search.format_penalty=0.1`; about 11 hours.

Eval, mean over the 128 eval rollouts. Base model (step 0): F1 0.256, reward 0.131 in the previous run; F1 0.271,
reward 0.044 in this one, lower on reward because invalid-JSON and cut-off episodes now score -0.2.

| Run | step 8: F1, reward | step 16: F1, reward | step 24: F1, reward |
|---|---|---|---|
| previous run (plain F4, old harness) | 0.251, 0.213 | 0.368, 0.350 | 0.438, 0.424 |
| this run (format penalty, Jasper's harness rules) | 0.284, 0.230 | 0.385, 0.345 | **0.448, 0.430** |
| Jasper, plain F4 (blog F1, repo F4) | 0.21, – | 0.23, 0.202 | 0.33, 0.316 |
| Jasper, F4 + format penalty (blog F1, repo F4) | 0.24, – | 0.34, 0.353 | 0.36, 0.418 |

Jasper's F1 here is from his format-penalty F1 chart, which plots both of his runs.

Eval at step 24. Endings: share of episodes ending with a successful `finish`, a plain reply, or invalid JSON (-0.2);
off-form: share of episodes with any off-form tool-call header.

| Run | precision | recall | `finish` | plain reply | invalid JSON | off-form | nothing curated | turns |
|---|---|---|---|---|---|---|---|---|
| previous run | 0.477 | 0.423 | 0.01 | 0.99 | – | 0.98 | 0.00 | 23.9 |
| this run | 0.487 | 0.433 | 0.98 | 0.00 | 0.02 | 0.01 | 0.02 | 18.7 |

Training, this run (64 queries x 8 rollouts per step):

| Step | 1 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| reward (with penalties) | -0.076 | 0.126 | 0.147 | 0.288 | 0.289 | 0.397 | 0.357 |
| episodes with off-form header | 0.52 | 0.15 | 0.38 | 0.05 | 0.01 | 0.00 | 0.01 |
| ended by invalid JSON | 0.19 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 |
| ended by `finish` | 0.29 | 0.82 | 0.95 | 0.91 | 0.92 | 0.97 | 0.98 |

Conclusions:

- The two fixes address different failures. The invalid-JSON rule fixed the broken `finish` calls within two steps
  (invalid-JSON endings 0.19 → 0.01), because a broken call now costs the whole episode. The format penalty fixed the
  headers more slowly and unevenly: off-form episodes fell to 0.15 by step 4, rebounded to 0.47 at step 7, then fell
  to about 0.01 from step 16.
- Search quality is unchanged within eval noise (0.047 F1). On the same training batches (both runs see the same
  batch at each step), training F1 is equal over steps 1–8 and higher in this run over steps 9–24 (+0.03 to +0.05),
  converging by step 24. Training reward is lower over steps 1–8 only because the penalties apply.
- Unlike Jasper's runs, the penalty did not raise recall (+0.01 here, +0.11 for him). Our previous run already reached
  0.42 recall, his format-penalty level; his plain run stopped at 0.32. Format drift did not hurt our plain run.
- Episodes are shorter (18 turns against 24–28 at the end of the previous run), mostly because the previous run spent
  about 6 turns per episode on failed `finish` calls. Jasper's format-penalty run instead rose to 28–30 turns.
- About 1–2% of episodes still end on invalid JSON, mostly malformed search arguments (single quotes, text after the
  closing brace); Jasper's format-penalty run keeps a similar floor (96–97% valid episodes).
- The trainer vs vLLM logprob diff stayed at 0.027–0.029 (previous run: rose to 0.035), and entropy stayed near 0.95
  through step 16 before falling to 0.79 (previous run: 0.65 by step 6).
- One run, one seed.
