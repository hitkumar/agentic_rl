# Harbor: how the eval infrastructure works

Notes on [Harbor](https://github.com/harbor-framework/harbor), the framework from the Terminal-Bench team for
running agents on benchmark tasks in sandboxes. Written for an eval-platform system design interview, so the focus
is infrastructure: how a job runs, how it scales, and where it stops scaling.

Sources: the Harbor repo (commit `7331d8a`, 2026-10-17, cloned at `~/oss/harbor`) and its docs
(`docs-mintlify/`, the source of harborframework.com/docs), the
[Terminal-Bench 2.0 paper](https://arxiv.org/abs/2601.11868), the
[terminal-bench-experiments](https://github.com/laude-institute/terminal-bench-experiments) configs, and the
external posts listed at the end. Paths below are relative to `src/harbor/` unless they start with `docs-mintlify/`.

## 1. The big picture

### System design

```
                     +---------------------------+
                     | User: harbor run + config |
                     +-------------+-------------+
                                   |
+----------------------------------v-----------------------------------+
| HARBOR HOST (one Python process, one asyncio event loop)             |
|                                                                      |
|  +-------------+    +-------------------+    +--------------------+  |
|  | Job planner |--->| TrialQueue        |--->| Trial runners      |  |
|  | tasks x     |    | semaphore =       |    | one coroutine per  |  |
|  | agents x    |    | n_concurrent,     |    | trial: start,      |  |
|  | attempts    |    | retries, backoff  |    | agent, verify, stop|  |
|  +-------------+    +-------------------+    +---------+----------+  |
|                                                        |             |
|  +-------------+    +-------------------+    +---------v----------+  |
|  | Hooks,      |    | Metrics: mean,    |    | Environment        |  |
|  | plugins     |    | pass@k, custom    |    | interface: start,  |  |
|  +-------------+    +-------------------+    | stop, exec, files  |  |
|                                              +---------+----------+  |
+--------------------------------------------------------+-------------+
         ^                                               |
         | tasks                                         | exec + file transfer
+--------+---------+                                     v
| REGISTRY         |          +--------------------------+---------------+
| Harbor Hub,      |          | SANDBOX BACKENDS                         |
| registry.json,   |          | local: Docker, Podman, Kata              |
| local task cache |          | cloud: Daytona, Modal, E2B, Vercel, ...  |
+------------------+          | k8s / VMs: GKE, OpenShift, EC2, SkyPilot |
                              +--------------------+---------------------+
                                                   |
                                                   v
          +----------------------------------------+---------+
          | SANDBOX (one per trial, fresh each time)         |      +------------+
          |  +---------------------+  +-------------------+  |      | MODEL API  |
          |  | main container:     |  | sidecars          |  |----->| Anthropic, |
          |  | task env + agent    |  | (optional):       |  |      | OpenAI,    |
          |  | (Claude Code, ...)  |  | services, MCP,    |  |      | Fireworks  |
          |  |                     |  | egress proxy      |  |      +------------+
          |  +---------------------+  +-------------------+  |
          +-------------------------+------------------------+
                                    | artifacts only (separate mode)
                                    v
          +-------------------------+------------------------+
          | VERIFIER: tests/test.sh -> reward.txt / .json    |
          | same sandbox (default) or a fresh one (separate) |
          +-------------------------+------------------------+
                                    | logs, trajectory, reward pulled to host
                                    v
          +-------------------------+------------------------+      +------------+
          | RESULTS (local disk): jobs/<job>/<trial>/        |----->| Harbor Hub |
          | result.json, trajectory.json (ATIF), logs        |      | upload,    |
          +--------------------------------------------------+      | viewer     |
                                                                    +------------+
```

Terminus 2 is the exception to "agent inside the sandbox": its loop runs on the Harbor host and types into a tmux
session in the main container through `exec`.

### Trial flow

```
harbor run (one Python process, one asyncio event loop, on your machine)
 |
 +- Job: expands config into trials = n_attempts x tasks x agents
 |
 +- TrialQueue: asyncio.Semaphore(n_concurrent) caps trials running at once
 |
 +- Trial (one coroutine per trial)
      1. start environment -> sandbox: local Docker or a cloud provider
      2. healthcheck
      3. install + run agent -> agent calls the model API
      4. collect logs, trajectory, artifacts
      5. run verifier (tests/test.sh) -> writes reward.txt / reward.json
      6. stop sandbox, write result.json to the local job directory
```

Three separate places are involved: the **Harbor process** (orchestration, always on one host), the **sandboxes**
(local or remote), and the **model server**. Only the sandboxes move when you switch to a cloud provider.

## 2. Core concepts

| Concept | What it is |
|---|---|
| **Task** | A directory: `instruction.md`, `task.toml` (timeouts, resources, network), `environment/` (Dockerfile or docker-compose), `tests/test.sh` (the grader), `solution/solve.sh` (reference solution). |
| **Dataset** | A versioned set of tasks, e.g. `terminal-bench@2.0`, pulled from a registry. |
| **Job** | One `harbor run`: datasets × agents × models × attempts, plus concurrency and retry settings. |
| **Trial** | One attempt of one agent on one task, in a fresh sandbox. |
| **Environment** | The sandbox backend that runs the task's container(s). |
| **Agent** | The agent harness under test (Claude Code, Codex, Terminus 2, ...). |
| **Verifier** | Runs the task's tests after the agent finishes and reads the reward. |

The agent sees only the instruction and the container. `tests/` is copied in at verification time, and
`solution/` is used only by the oracle agent.

## 3. Life of a job

- **Entry.** `harbor run` is an alias for `harbor job start` (`cli/main.py`). Config files (YAML/JSON) are layered,
  then CLI flags override them. Each `-m` model becomes its own agent config.
- **Planning** (`job.py`, `job_plan.py`). `Job.create` resolves datasets to tasks, pre-downloads every task, and
  expands trials as a plain Cartesian product: `n_attempts × tasks × agents`. Defaults: `n_attempts=1`,
  `n_concurrent_trials=4`.
- **Concurrency** (`trial/queue.py`). Everything runs in one process on one asyncio event loop. A global
  `asyncio.Semaphore(n_concurrent)` gates trials. All trials are created up front in a `TaskGroup` and wait on the
  semaphore. An optional per-agent semaphore (`agents[].n_concurrent`) limits one agent within the global cap.
- **No global LLM rate limiter.** The semaphores are the only throttle. Rate limits are handled by agents' own
  retries and by job-level retries.

## 4. Life of a trial

`trial/trial.py`, `trial/single_step.py`:

| Phase | What happens | Timeout |
|---|---|---|
| Environment start | Build or pull the image, start the container(s) | `build_timeout_sec` (default 600 s) |
| Healthcheck | Docker HEALTHCHECK semantics: start period, then N consecutive failures fail it | Own retry counts |
| Agent setup | Install the agent into the container (installed agents) | Agent's setup default (360 s) |
| Agent run | Agent works on the instruction | Task `agent.timeout_sec` |
| Collect | Download agent logs, trajectory and declared artifacts | None |
| Verify | Upload `tests/`, run `test.sh`, parse the reward | Task `verifier.timeout_sec` (default 600 s) |
| Stop | Tear down the sandbox (shielded from cancellation) | None |

- Every timeout is `min(base, max) × multiplier`. A global `timeout_multiplier` and per-phase multipliers let you
  stretch limits without editing tasks.
- **Hooks.** Events (`START`, `ENVIRONMENT_START`, `AGENT_START`, `AGENT_END`, `VERIFICATION_START`, `END`,
  `CANCEL`) drive the progress UI, live stats and plugins (e.g. upload each trial to a database as it finishes).
- **Multi-step tasks** repeat agent → verify per step in the same environment, and can stop early if a step's
  reward is below `min_reward`.

## 5. Environments and sandboxes

### One interface, many backends

`environments/base.py` defines what every backend implements: `start`, `stop`, `exec`, `upload_file/dir`,
`download_file/dir`, plus a shared healthcheck and network-policy hooks. Backends declare capabilities (GPUs,
compose, network allowlists, Windows, ...). A task that asks for something a backend can't do is rejected at job
start, not silently downgraded.

About 34 backends exist (`models/environment_type.py`):
- **Local:** docker (default), podman, apple-container, singularity, kata (Docker plus a VM-backed runtime).
- **Cloud sandbox APIs:** daytona, modal, e2b, runloop, novita, vercel, hf-sandbox, ... (about 20).
- **Kubernetes:** gke, ack, openshift. **VMs:** ec2, skypilot.

Because the trial only talks to this interface, the same agent and task run unchanged on any backend.

### How images get to the sandbox

- **Local Docker.** Each trial runs `docker compose build` under a per-task `asyncio.Lock`, so concurrent trials of
  one task build once and the rest hit Docker's layer cache. Images persist across jobs unless `--delete`.
  `task.toml` can name a prebuilt `docker_image` to skip building.
- **Cloud providers each do their own thing:**
  - **Daytona:** content-hashed snapshot (`harbor__<hash>__snapshot`) built from the Dockerfile and reused.
  - **Modal:** pull from a registry, or build from the Dockerfile on Modal.
  - **E2B:** template named `<name>__<hash>`, built once.
  - **GKE:** Cloud Build pushes to Artifact Registry, tag = content hash, skipped if it exists.
- **No warm pools** in the main providers. Reuse happens at the image, snapshot or template level, so every trial
  still pays sandbox creation time.
- The TB2 team pre-built Daytona snapshots for all tasks, 100 at a time (`scripts/create_snapshots.py` in the
  experiments repo).

### Exec and file transfer

- **Docker:** `docker compose exec` per command (one subprocess per call). Files go via `docker compose cp`, with
  a tar-over-stdin fallback. Log directories are bind-mounted from the host.
- **Cloud:** each provider's own exec API (Daytona polls a session command, Modal `sandbox.exec`, GKE an exec
  stream). Directories are usually sent as one tar.gz (Modal's code notes per-file uploads silently dropped files).
- **Terminus 2** drives tmux inside the container through this same `exec` call, so it works on every backend.

### Resources

`task.toml [environment]` sets `cpus`, `memory_mb`, `storage_mb`, `gpus`. How they are enforced is a policy:
`limit`, `request`, `guarantee` (request = limit), `auto` or `ignore`, and `auto` means different things per
backend:
- **Docker:** a hard memory ceiling equal to `memory_mb`, no reservation.
- **GKE:** requests only by default. `memory_limit_multiplier` adds a ceiling (1 = strict, 3 = Anthropic's 3×
  headroom).
- **Modal, Daytona, E2B:** requests only (soft). Daytona rounds memory down to whole GB.

This is exactly the variable Anthropic's infrastructure-noise post measured: strict request = limit caused 5.8%
infra errors from OOM kills, and uncapped resources raised Terminal-Bench 2.0 scores by 6 points.

### Isolation and security

- **Network policy per phase:** `public`, `no-network` or `allowlist`, set separately for setup, agent and
  verifier. On local Docker, an egress sidecar (nftables plus a transparent proxy) enforces allowlists and is up
  before the agent starts.
- **Separate verifier sandbox (optional):** the agent's sandbox is stopped, and a fresh one is built for grading.
  Only declared artifacts cross over. This stops the agent tampering with tests or the reward file, and makes
  regrading possible.
- **Secrets:** host env vars are forwarded only to the phase and agent that needs them. Model keys go only to the
  agent whose provider matches.
- **Isolation strength depends on the backend:** plain Docker shares the host kernel; Docker Sandboxes (`sbx`),
  Vercel and Kata give each trial a VM.

## 6. Failures, retries and resume

- **Error policy** (`models/trial/error_policy.py`, `docs-mintlify/jobs/error-classification.mdx`). Each exception
  maps to a cause, which decides: retry? still grade? is the reward valid? The key split:
  - **Agent's fault:** agent timeout or context overflow. Graded, reward valid, not retried.
  - **Infra's fault:** sandbox build failure, OOM, start timeout, model provider 5xx or rate limit. Retried, reward
    not valid.
- **Retries** (`trial/queue.py`). Each retry is a fresh trial in a fresh sandbox, with exponential backoff
  (`min_wait × multiplier^attempt`, capped). Default `max_retries=0`. The TB2 runs used 2-3 retries.
  - Caveat: the default retry exclude-list is hard-coded and has drifted from the policy table.
  - Caveat: the global semaphore stays held during backoff.
- **Cancellation (Ctrl-C).** Trials record the cancel, salvage logs and artifacts, stop their sandbox, and still
  write `result.json`.
- **Resume** (`harbor job resume`). Rescans the job directory, deletes half-finished trials, matches planned
  trials against finished ones by config, and runs only what's missing. It refuses if the job config changed.
- **Regrade.** Re-runs only the verifier on recorded agent outputs (needs separate-verifier mode), without
  re-running the agent.

## 7. Results and data

- **Job directory** (local disk):
  ```
  jobs/<job>/config.json, lock.json, result.json, job.log
  jobs/<job>/<task>__<id>/result.json, trial.log, exception.txt,
                         agent/ (trajectory.json), verifier/ (reward, test output), artifacts/
  ```
- **Metrics** (`metrics/`): mean (default), sum, min, max, a custom `metric.py` per dataset, and unbiased pass@k for
  binary rewards. A missing reward counts as 0.
- **Trajectories: ATIF** (`rfcs/0001-trajectory-format.md`). One JSON format for every agent: steps, tool calls,
  observations, token counts, cost, and optionally token IDs and logprobs (so RL can avoid retokenization drift).
  Each agent's native log is converted after the run.
- **Live view:** `--stream` polls the sandbox every 2 s for session files and updates a live trajectory view.
- **Upload and sharing:** `harbor upload` sends trial tarballs and trajectories to Harbor Hub (Supabase storage).
  It's resumable, so already-uploaded trials are skipped. A local viewer app compares jobs and trials.
- **Registry:** datasets are versioned by digest in Harbor Hub (the legacy `registry.json` pins git commits).
  Tasks are cached locally by content hash. 85 adapters convert existing benchmarks (SWE-bench, etc.), each with a
  parity check against the original benchmark's scores.

## 8. Hosted Harbor (the multi-tenant version)

Harbor Hub can run jobs for you (alpha, gated access). It's the closest thing to an eval service in the repo.
- **API:** `POST /job-submit` (or `harbor run --launch`). An `Idempotency-Key` header is required, so client
  retries can't launch duplicate jobs.
- **Limits:** 1–1000 concurrent trials and up to 50,000 trials per job. Quota errors return 429.
- **Reproducibility:** agent versions and task refs are pinned at submission. Local tasks are published as
  temporary private packages and referenced by digest.
- **Secrets:** sent alongside the config, never inside it. They're encrypted (KMS) and only decrypted in the trial
  runner. In "gateway" mode the agent gets a scoped proxy token instead of the real model key.
- **Status and control:** per-trial status counts, cancel with a reason, retry failed trials with filters.
- **Where it runs:** a Supabase backend (edge functions, Postgres). From reserved env-var prefixes, workers likely
  run on Modal or GCP (inference).

## 9. How it's run at scale

- **Terminal-Bench 2.0** (paper and experiments repo): 32,155 trials (6 agents × 16 models × ≥5 attempts). Runs
  used Daytona at 8–100 concurrent trials per job, `n_attempts: 5`, 2–3 retries, and pinned agent CLI versions.
  Most trials finish in under 20 minutes, some take 2 hours and close to 100M tokens.
- **Results pipeline** in the experiments repo: a trial-completed hook uploads each trial to Supabase and writes a
  row to Postgres with per-phase timestamps, token counts and the exception. A script queries for missing or
  invalid trials and generates a rerun config. Scoring counts only clean trials and agent timeouts, with a
  standard error from per-task variance.
- **Daytona case study** (vendor marketing): 37k sandboxes in one week, versus 4–6 local Docker containers before.
- **Anthropic** runs Terminal-Bench 2.0 on its own GKE cluster: the infrastructure-noise post.
- **RL training:** SkyRL's Harbor generator and Mercor's 397B run use Harbor trials as RL rollouts. Mercor ran each
  trial as a Ray task in a Modal sandbox at 300–550 concurrent rollouts, and hit LLM-judge rate limits above 800.
- **Fireworks:** the Specialized Intelligence Index takes partner tasks "in Harbor-compatible format", runs pinned
  harnesses, and blocks benchmark repos and mirrors during runs.

## 10. Where the single-host design stops scaling

What the code shows:
- **One process, one event loop** schedules every trial, runs every hook and updates every metric. If the host
  dies, the job dies, and only resume recovers it.
- **Per-completion work grows with job size.** Live metrics are recomputed over all rewards on every trial
  completion, and the full job stats JSON is rewritten on every trial event. A custom `metric.py` runs as a
  blocking subprocess that stalls the loop.
- **All state is on local disk,** with no lock on the job directory. Resume rescans and re-parses every
  `result.json`.
- **Logs and artifacts are pulled back to the one host** after each trial (or polled every 2 s when streaming).
- **No global model rate limiting,** and the default retry backoff is constant, so a rate-limited provider gets
  hit again every second.
- **Wasted capacity:** sandboxes can sit up and idle while holding a slot (e.g. during retry backoff).

These match the Fireworks job description: "eliminate single-host coordination bottlenecks, resolve log-syncing
latency, and build seamless programmatic access."

What a multi-tenant platform would change (my design notes, not from Harbor):
- **Orchestrator + queue + metadata DB.** A job API (with idempotency keys, like Hosted Harbor) hands jobs to an
  orchestrator that expands them into trials. Stateless workers pull trials from a queue, run them, and record
  status and results in a metadata DB. Details below.
- **Results in a database and object storage,** not local JSON. Trajectories and logs stream straight from the
  sandbox to object storage. Aggregates are computed incrementally or on read.
- **Global rate limiting per model endpoint and per tenant,** with backoff and jitter. Separate quotas per tenant.
- **Sandbox fleet:** pre-built images per task digest, warm pools for popular tasks, VM-level isolation for
  untrusted code, per-tenant network policy, and resource settings recorded with every result.
- **Keep Harbor's good ideas:** the task format, the environment interface, the error policy (infra vs. agent
  failures, valid vs. invalid rewards), separate verifier sandboxes, ATIF trajectories, and pinned versions for
  reproducibility.

### Distributed design: queue plus metadata DB

```
Orchestrator
  1. INSERT trial row (status = queued)   -> Metadata DB
  2. send message {trial_id}              -> Queue

Worker
  3. receive {trial_id}                   <- Queue
  4. claim: queued -> running, set lease  -> Metadata DB
  5. run the trial, heartbeat the lease   -> Metadata DB
  6. write result, status = done          -> Metadata DB (logs, trajectory -> object storage)
```

- **Two stores, two jobs.** The metadata DB holds the full record of every trial (config, status, attempts,
  worker, lease, result) and is the source of truth. The queue holds only a small message per trial, the trial
  ID, meaning "ready to run". Job status, failed-trial lists and cancels are all DB queries, never queue reads.
- **The orchestrator writes to both: DB first, then queue.** If it crashes after the DB write but before the
  enqueue, the row sits in `queued` with no message. A periodic sweep re-sends messages for rows stuck in `queued`.
  The reverse order would leave messages for trials the DB doesn't know about.
- **Duplicates are expected.** Queues deliver at least once, so the worker claims the trial with a conditional
  update (`... SET status = 'running' WHERE id = ? AND status = 'queued'`). If no row changed, someone else has it,
  so the worker drops the message. Results are keyed by trial ID, so a rerun can't double-count.
- **The queue dispatches; the DB tracks liveness.** The worker acks the message as soon as its claim succeeds,
  then heartbeats the lease in the DB while the trial runs. A reaper requeues trials whose lease expired (dead
  worker) and marks them failed after N attempts. This keeps 2-hour trials independent of queue lock limits.
- **Cancel through the DB.** Messages can't be pulled back out of a queue. Mark the job cancelled; workers check
  at claim time and on every heartbeat, then stop the sandbox.
- **Fairness and concurrency live in the orchestrator.** Putting all of one job's 50k trials in a single FIFO
  queue starves other tenants. The orchestrator enqueues only up to each job's and tenant's concurrency quota and
  tops up as trials finish (this replaces Harbor's `n_concurrent` semaphore).
- **Retries:** the orchestrator re-enqueues with backoff and keeps the attempt count in the DB, applying Harbor's
  error policy (retry infra failures, not agent failures).
- **Which queue.** Claim rates are low (trials take minutes to hours), so SQS, RabbitMQ or Redis Streams all work.
  Kafka is overkill: classic consumer groups cap parallelism at one consumer per partition and commit offsets in
  order, so a long trial blocks the ones behind it. Share groups (KIP-932, "Queues for Kafka") add per-message
  acks, but their record lock is capped at 1 hour, which is fine only because of the ack-on-claim pattern above.
  Temporal is an alternative if you want per-step retries of the trial workflow.

## External sources

Copies saved in `~/oss/harbor_ext/`.
- Daytona customer story (Laude Institute): daytona.io/customers/laude
- Terminal-Bench 2.0 announcement: tbench.ai/news/announcement-2-0
- Docker Sandboxes for Harbor: docker.com/blog (`-e sbx`, a microVM per task, 64 concurrent)
- Vercel: vercel.com/changelog (`--env vercel`, Firecracker microVM per trial)
- SkyRL + Harbor: novasky-ai.notion.site/skyrl-harbor
- Mercor's 397B SkyRL guide (`~/oss/mercor_blog/`)
- Anthropic, "Quantifying infrastructure noise in agentic coding evals" (`~/oss/anthropic_eng/`)
- Fireworks Specialized Intelligence Index methodology (`~/oss/fireworks_blog/_sii_index_page.txt`)
