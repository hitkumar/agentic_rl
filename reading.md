# Reading: agentic RL post-training

Blogs, papers and repos on post-training agents with RL.

## Jasper Lu: Training search agents with GRPO (2026-09)

[Blog](https://jasperlu.com/blog/training-search-agents-grpo/) ·
[code](https://github.com/jasper-lu/sec-search-rl). Reproduced in [search_agent/](search_agent/).

- **Task:** multi-hop questions over SEC filings ([Harness-1](https://arxiv.org/abs/2606.02373) data). The agent
  uses lexical tools (BM25, grep, read) to curate chunks that support each question's 3, 5 or 7 facts.
- **Setup:** gpt-oss-20b with a rank-32 LoRA on Thinking Machines' Tinker API and tinker-cookbook. Dr. GRPO (no std
  normalization, no KL), 64 queries x 8 rollouts per step, 24 steps.
- **Reward vs eval:** trains on F4, which weights recall 16x over precision because a missed chunk costs more than an
  extra one. Reports held-out F1. Curating nothing gets -0.2.
- **Batch size:** 8 queries per step reward-hacked F4 by curating everything; 64 trained stably.

Reward experiments, held-out at step 24:

| Reward | Eval F1 | Eval recall |
|---|---|---|
| Base model | 0.19 | 0.16 |
| F4 | 0.33 | 0.32 |
| F4 + 0.2 x trajectory recall (discovery bonus) | 0.33 | 0.30 |
| F4 - 0.1 for malformed tool calls (format penalty) | 0.36 | 0.43 |

- **Discovery bonus:** rewards gold chunks seen anywhere in search results, not just curated. It did not change the
  final score.
- **Format penalty:** a lenient parser let malformed tool calls through, so they spread to 99% of episodes. The
  penalty cut them below 1% and, unexpectedly, raised recall.

## Mercor: A 397B RL training guide with SkyRL (2026-09)

[Blog](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/) ·
[code](https://github.com/Mercor-Intelligence/ApexAgents-SkyRL-Recipe).

- **Task:** long-horizon knowledge work (consulting, banking, law) in simulated company worlds with files, email and
  chat, reached through MCP tools or code. Trained on 1,928 expert tasks; evaluated on the 480-task APEX-Agents
  benchmark.
- **Setup:** RL with no SFT warmup, on SkyRL (fully async, in-flight weight sync), vLLM for rollouts, Megatron for
  training, Harbor for the rollout lifecycle and Modal for sandboxes.
- **Result:** Qwen3.5-397B-A17B went from 16.11% to 27.29% Pass@1. Qwen3.6-35B-A3B surpassed Opus 4.5. Gains
  transferred to an unseen harness and to Terminal-Bench, with no loss on HLE or GPQA.

De-risking before spending compute:
- **Fix the harness first:** reading base-model traces surfaced broken tools and missing packages. Fixes plus a parse
  retry, truncated tool results and a context nudge raised the untrained 35B from 22.74% to 28.69%, about one
  epoch of training's worth.
- **Token in, token out:** pass the engine's token IDs to the trainer instead of re-tokenizing text, or training goes
  silently off-policy.
- **Check trainer vs inference logprobs:** a mean difference below 0.03 is healthy. This check caught a vLLM bug.
- **Overfit 32 tasks first:** if they can't be learned, the environment or grader is broken. This exposed a weak
  file-diff grader.

Algorithm ablations on the 35B:
- **Loss aggregation:** averaging per prompt group instead of per token gave +3.9 points, because trajectories range
  from 2k to 128k tokens and per-token averaging lets the longest dominate.
- **Context nudge:** telling the model to wrap up when 20% of the context is left gave +3.0 points, from training
  alone. Fewer rollouts overflow and get zeroed.
- **No gain:** DPPO matched the GLM-5 loss on score. Overlong filtering and length penalties did not help.

Main lesson: data mattered more than algorithms. The best knob gave +3.9 points; post-training as a whole gave 10-12.

## Coinbase: Owning Intelligence series, an Onramp fraud agent (2026-10)

[Part 1: build](https://www.coinbase.com/blog/part-one-how-coinbase-built-evaluated-and-post-trained-a-fraud-agent) ·
[Part 2: evaluate](https://www.coinbase.com/blog/part-two-owning-intelligence-series) ·
[Part 3: post-train](https://www.coinbase.com/blog/part-three-owning-intelligence-series).

- **Build:** an LLM reviews selected guest-checkout transactions with their recent history, after the existing ML
  model and rules. It returns only a risk level and can block but not unblock. With Opus, an online A/B test showed
  30% fewer fraudulent transactions and 22% less fraud value. Projected savings were about 3x the inference cost.
- **Evaluate:** a fixed benchmark of 16,140 historical transactions (813 fraud), scored on precision, recall, F1 and
  dollar-weighted recall. Newer Opus, Sonnet and GPT versions all scored lower on recall and F1 than their
  predecessors. Sonnet 5's recall fell by 22 points, so model upgrades need a domain benchmark.
- **Post-train:** Qwen3.5-9B with LoRA, GRPO on SkyRL (Megatron and vLLM) on Anyscale. The reward is verifiable,
  scored against known fraud outcomes, and balances fraud and legitimate examples so a constant prediction can't
  win. Training took one 96 GB GPU and under $100.
- **Result:** the 9B beat Opus 4.5 on all four metrics: F1 +9.6, recall +12.0, dollar-weighted recall +35.4. The post
  calls these percent, but its chart labels them percentage points. Self-hosted serving cut median latency by 55%.

Note: this is single-turn classification with RL, not a multi-turn tool-using agent.

## Anthropic: Demystifying evals for AI agents (2026-01)

[Blog](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). A useful anchor post on evals.

- **Harbor:** runs agents like Claude Code or Codex against benchmark tasks like Terminal-Bench 2.0.
- **Types of graders:**
  - Code-based graders: deterministic, including format graders.
  - Model-based graders: LLM as a judge.
  - Human graders: can be used to calibrate model-based graders.
- **Capability eval:** define one when you are trying to add a new capability to the agent.
- **Regression eval:** once you saturate a capability eval, it becomes a regression eval, so you can check that the
  model doesn't get worse on it over time.
- **Graders:** define them unambiguously, so the rubric makes it clear to a human whether a trajectory accomplishes
  the task.
- **Transcripts:** reading them is very important for understanding eval failures and agent performance in general.
  Calibrating the LLM judge against humans is also important.
- **pass@k:** whether the agent can do the task at all.
- **pass^k:** how reliable the agent is.

Open questions:

- We can use a grader to evaluate the final outcome of an agentic task (like checking the database for a flight
  booking), but what is the best way to evaluate the trajectory the agent takes?
  - Default: grade the outcome, not the path (Anthropic Step 5; tau-bench and Terminal-Bench check final state).
  - Check the trajectory only for hard rules (e.g. verify identity before a refund) or when tool choice is the skill
    being tested (Claude for Chrome: DOM vs. screenshots). Track turns, tokens and cost, but don't grade on them.

- How do we balance over-triggering and under-triggering for tool calls? A model should answer from its own
  knowledge when it can, but still call the tool when it needs to.
  - Some rows always need the tool (fresh or private data, actions). For the rest, it depends on what the
    student knows, not the teacher.
  - Teachers (with tools) produce a reference answer. Resolve teacher disagreements instead of dropping them, and have
    humans spot-check references.
  - Sample the student k times without tools. Fraction correct above a threshold (not pass@k): "no tool".
  - Otherwise give the student the tool result. If it becomes correct: "call tool", with the teacher's arguments.
  - If it is still wrong, the tool doesn't help, so leave the row out.
  - Calibrate the grader that compares student answers to references against humans.
  - In RL, a small bonus for calling the tool helps if the model never explores calls. Give it only on "call tool"
    rows, with a penalty on "no tool" rows, and drop it once call rates look right (the labels go stale).

- How do we calibrate an LLM judge with human judgements?
  - Two humans label the same traces. Check they agree (Cohen's kappa), else fix the rubric.
  - Resolve their disagreements (third annotator or discussion) to get a gold label for every trace.
  - Iterate on the judge prompt/model until its kappa with gold is close to the human-human kappa. Tune on one split,
    report on a held-out split.
  - Also check the confusion matrix (judges tend to be lenient) and re-validate when the judge, rubric or data changes.
  - **Cohen's kappa:** agreement between two annotators, corrected for agreement by chance.
    κ = (p_o − p_e) / (1 − p_e). 1 is perfect, 0 is chance level, > 0.8 is usually considered very high.
    - p_o (observed agreement): fraction of items where both gave the same label.
    - p_e (chance agreement): for each label, multiply the fraction of items A gave that label by the fraction B
      gave it, then sum over labels. E.g. A says Fail 70% / Pass 30%, B says Fail 60% / Pass 40%:
      p_e = 0.7 × 0.6 + 0.3 × 0.4 = 0.54. With p_o = 0.80, κ = 0.26 / 0.46 ≈ 0.57.

## Anthropic: Quantifying infrastructure noise in agentic coding evals (2026-02)

[Blog](https://www.anthropic.com/engineering/infrastructure-noise)

- Giving containers more CPU and memory alone raised Terminal-Bench 2.0 scores by 6 points, more than the gap between top models.
- Infra settings change eval scores, so control them and report them with the results.

## Terminal-Bench 2.0: Benchmarking agents on hard, realistic tasks in command line interfaces (2026-01)

[Paper](https://arxiv.org/abs/2601.11868)

- **Sandbox:** an isolated, disposable environment where the agent runs one trial. It keeps the agent's actions from affecting anything outside, and gives every trial the same clean starting state.
- **Eval harness vs. agent harness:** the eval harness (Harbor) starts a container per trial, runs the agent, then runs the graders. The agent harness (Claude Code, Codex) is the system under test, so you always measure model + agent harness together.
- Harbor is introduced which is the eval platform.

# Harness / Agentic Engineering

## Jediah Katz (Cursor, now SpaceXAI): Is the Agent Harness Dead? (AI Engineer talk)

[Talk](https://www.youtube.com/watch?v=7Y4sbZgBdnc)

- Very interesting perspective on harnesses. Coding harnesses might be dead, because models have learned a lot of what
  the harness used to cover for (exploring the codebase, checking your work, etc.).
- This happens through a **data flywheel**: the harness brings harder tasks within reach → more people do them with
  agents → that data trains the next model → the model learns the skill natively and the harness can drop it.
- Harnesses in general are not dead though:
  1. They let the model do tasks the current model can't do on its own, kind of like hacks but still very important.
     This opens the next frontier and restarts the flywheel.
  2. They matter for security and cost. Cost comes from caching, token efficiency and orchestrating across multiple
     models. Harness choice barely changed scores in the studies he cited, but cost per task differed a lot.
- The cycle: build a harness, and once the model catches up, delete it, like what's happening with coding these days.
