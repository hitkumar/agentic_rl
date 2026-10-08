"""One search episode: gpt-oss-20b calls the search tools until it finishes, token IDs in and out.

Adapted from jasper-lu/sec-search-rl (src/sec_rl/environment.py). Not ported: his off-by-default penalties
(unfinished episodes, invalid curations), the f4s trajectory-recall reward, and his lenient tool-call parser; the
system message is gpt-oss's standard one, not his tool-routing line only.

Each turn samples one assistant reply, which stops at its tool call (<|call|>); gpt-oss makes one call per turn. The
call runs against SearchTools, its result is appended as a Harmony tool message, and the next turn samples from the
extended token sequence. Sampled tokens are appended exactly as returned, never re-rendered: Harmony renders some
headers differently from how the model writes them, and training must see the tokens that were sampled.

The episode ends when the model calls finish, replies without a tool call, or runs out of turns or context; the
curated set is scored. As in Jasper's harness, a turn the harness cannot use ends the episode at a flat EMPTY_REWARD
(-0.2) instead, curated set unscored: a tool call whose arguments are not valid JSON (e.g. finish with {""}), or a
reply cut off at MAX_GENERATION_TOKENS. Unknown argument keys are dropped, as his tool validation does.

Run the dev queries against a vLLM server (see search_agent/README.md to start one):
  uv run python -u -m search_agent.trajectory --limit 8

Jasper's blog table (initial explorations: 4 trials on each of the 32 eval queries, scored by F1):
  uv run python -u -m search_agent.trajectory --limit 32 --samples 4 --output rollouts.jsonl
"""

import argparse
import asyncio
import json
import multiprocessing
import re
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx
import pandas as pd
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

from search_agent.prompts import SYSTEM_PROMPT
from search_agent.retrieval import DATA_DIR, Index
from search_agent.rewards import EMPTY_REWARD, Score, reward, score
from search_agent.tools import TOOL_SPECS, SearchTools, Session

MAX_TURNS = 40
# Prompt plus responses; the default for --context-length. The vLLM server's --max-model-len must be at least this.
CONTEXT_LENGTH = 65_536
MAX_GENERATION_TOKENS = 2_048
# Processes running tool calls; see tool_pool.
TOOL_PROCESSES = 32

TOOLS = [ToolDescription.new(**spec) for spec in TOOL_SPECS]
TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}
TOOL_PARAMETERS = {spec["name"]: set(spec["parameters"]["properties"]) for spec in TOOL_SPECS}
# Stop reasons for turns the harness cannot use; the episode scores a flat EMPTY_REWARD (see the module docstring).
BROKEN_TURN_ENDINGS = ("invalid_json", "truncated")

ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
# <|call|> ends a tool call, <|return|> a final answer.
STOP_TOKEN_IDS = ENCODING.stop_tokens_for_assistant_actions()
ASSISTANT_START = ENCODING.encode("<|start|>assistant", allowed_special="all")
# The one header form Harmony renders for a tool call, as in Jasper's environment.py. The model also writes others
# (e.g. "code" instead of "<|constrain|>json"), which run_tool still executes.
CANONICAL_TOOL_HEADER = re.compile(r"<\|channel\|>commentary to=functions\.[a-z0-9_]+ <\|constrain\|>json<\|message\|>")


@dataclass
class Sample:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str  # "stop" at a stop token, "length" at max_tokens


class Sampler(Protocol):
    async def __call__(self, token_ids: list[int], max_tokens: int) -> Sample: ...


class VLLMSampler:
    """Samples from a vLLM server's /v1/completions endpoint, with token IDs as the prompt."""

    def __init__(self, client: httpx.AsyncClient, model: str, temperature: float = 1.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    async def __call__(self, token_ids: list[int], max_tokens: int) -> Sample:
        response = await self.client.post(
            "/v1/completions",
            json={
                "model": self.model,
                "prompt": token_ids,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "stop_token_ids": STOP_TOKEN_IDS,
                "logprobs": 0,
                "return_token_ids": True,
                "skip_special_tokens": False,
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return Sample(choice["token_ids"], choice["logprobs"]["token_logprobs"], choice["finish_reason"])


@dataclass
class Trajectory:
    prompt_ids: list[int]
    # Everything after the prompt: sampled replies interleaved with tool results.
    response_ids: list[int] = field(default_factory=list)
    # 1 for sampled tokens, 0 for tool results.
    loss_mask: list[int] = field(default_factory=list)
    # Sampler logprobs for sampled tokens, 0.0 for tool results.
    logprobs: list[float] = field(default_factory=list)
    # One of: finish, no_tool_call, parse_error, invalid_json, truncated, context_full, max_turns.
    stop_reason: str = "max_turns"
    turns: int = 0
    session: Session = field(default_factory=Session)
    reward: float = 0.0
    # Tool-call headers in the sampled turns, and those not in the canonical form.
    call_headers: int = 0
    off_format_calls: int = 0
    # Executed tool calls whose observation is an error, and those of them to finish.
    failed_tool_calls: int = 0
    failed_finish_calls: int = 0


def render_prompt(query: str) -> list[int]:
    messages = [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM)),
        Message.from_role_and_content(
            Role.DEVELOPER, DeveloperContent.new().with_instructions(SYSTEM_PROMPT).with_function_tools(TOOLS)
        ),
        Message.from_role_and_content(Role.USER, query),
    ]
    return ENCODING.render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)


def tool_pool(processes: int = TOOL_PROCESSES) -> Executor:
    """Processes that run the tool calls, each with its own Indexes.

    A search takes up to a few hundred ms, mostly in Python. Run on the event loop, it would hold up every other
    episode; run in threads, the searches would contend for the GIL and overrun grep_corpus's time limit.
    """
    return ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn"))


# In each tool_pool process: an Index per corpus, opened on first use.
indexes: dict[str, Index] = {}


def run_tool_in_pool(session: Session, corpus: str, recipient: str, arguments: str) -> tuple[dict, Session]:
    """run_tool in a tool_pool process; also returns the session, whose changes are otherwise lost there."""
    if corpus not in indexes:
        indexes[corpus] = Index(DATA_DIR / corpus)
    return run_tool(SearchTools(indexes[corpus], session), recipient, arguments), session


def run_tool(tools: SearchTools, recipient: str, arguments: str) -> dict:
    name = recipient.removeprefix("functions.")
    if name not in TOOL_NAMES:
        return {"error": f"Unknown tool {recipient}."}
    try:
        kwargs = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return {"error": f"Arguments are not valid JSON: {exc}"}
    if not isinstance(kwargs, dict):
        return {"error": "Arguments must be a JSON object."}
    kwargs = {key: value for key, value in kwargs.items() if key in TOOL_PARAMETERS[name]}
    try:
        return getattr(tools, name)(**kwargs)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}


async def run_trajectory(
    sampler: Sampler,
    tools: Executor,
    query: str,
    facts: list[dict],
    context_length: int = CONTEXT_LENGTH,
    corpus: str = "",
) -> Trajectory:
    """Runs the tool calls in tools, a tool_pool, against the index in DATA_DIR / corpus.

    corpus "" is DATA_DIR itself; e.g. "sec_256" is the 256-query ablation set's smaller corpus.
    """
    trajectory = Trajectory(prompt_ids=render_prompt(query))
    loop = asyncio.get_running_loop()
    while trajectory.turns < MAX_TURNS:
        max_tokens = min(
            MAX_GENERATION_TOKENS, context_length - len(trajectory.prompt_ids) - len(trajectory.response_ids)
        )
        if max_tokens <= 0:
            trajectory.stop_reason = "context_full"
            break
        sample = await sampler(trajectory.prompt_ids + trajectory.response_ids, max_tokens)
        trajectory.turns += 1
        text = ENCODING.decode(sample.token_ids)
        trajectory.call_headers += text.count("to=functions.")
        trajectory.off_format_calls += text.count("to=functions.") - len(CANONICAL_TOOL_HEADER.findall(text))
        trajectory.response_ids += sample.token_ids
        trajectory.loss_mask += [1] * len(sample.token_ids)
        trajectory.logprobs += sample.logprobs
        if sample.finish_reason == "length":
            # Cut off by the context limit rather than the per-turn cap: scored, as Jasper's harness ends the episode
            # before a turn that may not fit and scores it.
            trajectory.stop_reason = "truncated" if max_tokens == MAX_GENERATION_TOKENS else "context_full"
            break
        try:
            reply = ENCODING.parse_messages_from_completion_tokens(sample.token_ids, Role.ASSISTANT)
        except HarmonyError:
            trajectory.stop_reason = "parse_error"
            break
        call = reply[-1] if reply else None
        recipient = (call.recipient or "") if call else ""
        content = call.content[0] if call and call.content else None
        if not recipient.startswith("functions.") or not isinstance(content, TextContent):
            trajectory.stop_reason = "no_tool_call"
            break
        try:
            json.loads(content.text)
        except json.JSONDecodeError:
            trajectory.stop_reason = "invalid_json"
            break
        observation, trajectory.session = await loop.run_in_executor(
            tools, run_tool_in_pool, trajectory.session, corpus, recipient, content.text
        )
        if "error" in observation:
            trajectory.failed_tool_calls += 1
            trajectory.failed_finish_calls += recipient == "functions.finish"
        if trajectory.session.finished:
            trajectory.stop_reason = "finish"
            break
        tool_message = (
            Message.from_author_and_content(
                Author.new(Role.TOOL, recipient), json.dumps(observation, ensure_ascii=False)
            )
            .with_recipient("assistant")
            .with_channel("commentary")
        )
        tool_ids = ENCODING.render(tool_message) + ASSISTANT_START
        trajectory.response_ids += tool_ids
        trajectory.loss_mask += [0] * len(tool_ids)
        trajectory.logprobs += [0.0] * len(tool_ids)
    if trajectory.stop_reason in BROKEN_TURN_ENDINGS:
        trajectory.reward = EMPTY_REWARD
    else:
        trajectory.reward = reward(trajectory.session.curated_ids, facts)
    return trajectory


@dataclass
class Rollout:
    query_id: str
    trial: int  # 1-based
    trajectory: Trajectory
    f1: Score  # beta=1, the eval metric


def report(rollouts: list[Rollout]) -> None:
    """Print per-query F1 and the summary rows of Jasper's blog table: best-of-N F1/precision/recall and trial-1 F1.

    Best-of-N takes each metric's maximum over a query's trials independently, as in Jasper's evaluate.py.
    """
    by_query: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        by_query.setdefault(rollout.query_id, []).append(rollout)
    groups = list(by_query.values())

    def mean(values) -> float:
        values = list(values)
        return sum(values) / len(values)

    def best(metric: str) -> float:
        return mean(max(getattr(r.f1, metric) for r in group) for group in groups)

    print(f"\n{'query':<8} best_f1 trial1_f1  per-trial f1")
    for query_id, group in by_query.items():
        f1s = [r.f1.f_beta for r in group]
        print(f"{query_id:<8} {max(f1s):7.3f} {f1s[0]:9.3f}  " + " ".join(f"{f1:.3f}" for f1 in f1s))
    print(
        f"\nbest-of-{len(groups[0])}: f1={best('f_beta'):.3f} precision={best('precision'):.3f} "
        f"recall={best('recall'):.3f}"
    )
    print(f"trial 1: f1={mean(group[0].f1.f_beta for group in groups):.3f}")
    print(
        f"all {len(rollouts)} rollouts: f1={mean(r.f1.f_beta for r in rollouts):.3f} "
        f"precision={mean(r.f1.precision for r in rollouts):.3f} recall={mean(r.f1.recall for r in rollouts):.3f} "
        f"reward={mean(r.trajectory.reward for r in rollouts):.3f} turns={mean(r.trajectory.turns for r in rollouts):.1f}"
    )
    print(f"stop reasons: {dict(Counter(r.trajectory.stop_reason for r in rollouts).most_common())}")
    never = sum(all(not r.trajectory.session.curated_ids for r in group) for group in groups)
    print(f"queries with nothing curated in any trial: {never}/{len(groups)}")


def write_rollouts(rollouts: list[Rollout], path: Path) -> None:
    with path.open("w") as f:
        for r in rollouts:
            record = {
                "query_id": r.query_id,
                "trial": r.trial,
                "f1": r.f1.f_beta,
                "precision": r.f1.precision,
                "recall": r.f1.recall,
                "reward": r.trajectory.reward,
                "stop_reason": r.trajectory.stop_reason,
                "turns": r.trajectory.turns,
                "curated_ids": r.trajectory.session.curated_ids,
                "transcript": ENCODING.decode(r.trajectory.response_ids),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="unsloth/gpt-oss-20b-BF16")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=8, help="number of queries to run, concurrently")
    parser.add_argument("--samples", type=int, default=1, help="trajectories per query")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH, help="token cap on prompt plus responses")
    parser.add_argument(
        "--corpus", default="", help="search the index in this subdirectory of search_agent/data, e.g. sec_256"
    )
    parser.add_argument("--output", type=Path, help="write every rollout, with its decoded transcript, as JSONL")
    args = parser.parse_args()

    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    queries = queries[queries.split == args.split].head(args.limit)
    with tool_pool() as tools:
        async with httpx.AsyncClient(base_url=args.server, timeout=None) as client:
            sampler = VLLMSampler(client, args.model)

            async def run(row, trial: int) -> Rollout:
                facts = json.loads(row.facts)
                trajectory = await run_trajectory(sampler, tools, row.query, facts, args.context_length, args.corpus)
                f1 = score(trajectory.session.curated_ids, facts, beta=1.0)
                print(
                    f"{row.query_id} trial {trial}: f1={f1.f_beta:.3f} p={f1.precision:.3f} r={f1.recall:.3f} "
                    f"reward={trajectory.reward:.3f} stop={trajectory.stop_reason} turns={trajectory.turns} "
                    f"curated={len(trajectory.session.curated_ids)} tokens={len(trajectory.response_ids)}",
                    flush=True,
                )
                return Rollout(row.query_id, trial, trajectory, f1)

            rollouts = await asyncio.gather(
                *(run(row, trial) for row in queries.itertuples() for trial in range(1, args.samples + 1))
            )
    report(rollouts)
    if args.output:
        write_rollouts(rollouts, args.output)
        print(f"wrote {len(rollouts)} rollouts to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
