"""One search episode: gpt-oss-20b calls the search tools until it finishes, token IDs in and out.

Adapted from jasper-lu/sec-search-rl (src/sec_rl/environment.py), without its harness extras (parse-failure retries,
context nudges, reward penalties).

Each turn samples one assistant reply, which stops at its tool call (<|call|>); gpt-oss makes one call per turn. The
call runs against SearchTools, its result is appended as a Harmony tool message, and the next turn samples from the
extended token sequence. Sampled tokens are appended exactly as returned, never re-rendered: Harmony renders some
headers differently from how the model writes them, and training must see the tokens that were sampled.

The episode ends when the model calls finish, replies without a tool call, or runs out of turns or context. The
curated set is scored either way.

Run the dev queries against a vLLM server (see search_agent/README.md to start one):
  uv run python -u -m search_agent.trajectory --limit 8
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
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
from search_agent.rewards import reward
from search_agent.tools import TOOL_SPECS, SearchTools, Session

MAX_TURNS = 40
MAX_TRAJECTORY_TOKENS = 65_536
MAX_GENERATION_TOKENS = 2_048

TOOLS = [ToolDescription.new(**spec) for spec in TOOL_SPECS]
TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}

ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
# <|call|> ends a tool call, <|return|> a final answer.
STOP_TOKEN_IDS = ENCODING.stop_tokens_for_assistant_actions()
ASSISTANT_START = ENCODING.encode("<|start|>assistant", allowed_special="all")


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
    # One of: finish, no_tool_call, parse_error, truncated, context_full, max_turns.
    stop_reason: str = "max_turns"
    turns: int = 0
    session: Session = field(default_factory=Session)
    reward: float = 0.0


def render_prompt(query: str) -> list[int]:
    messages = [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM)),
        Message.from_role_and_content(
            Role.DEVELOPER, DeveloperContent.new().with_instructions(SYSTEM_PROMPT).with_function_tools(TOOLS)
        ),
        Message.from_role_and_content(Role.USER, query),
    ]
    return ENCODING.render_conversation_for_completion(Conversation.from_messages(messages), Role.ASSISTANT)


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
    try:
        return getattr(tools, name)(**kwargs)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}


async def run_trajectory(sampler: Sampler, index: Index, query: str, facts: list[dict]) -> Trajectory:
    trajectory = Trajectory(prompt_ids=render_prompt(query))
    tools = SearchTools(index, trajectory.session)
    while trajectory.turns < MAX_TURNS:
        max_tokens = min(
            MAX_GENERATION_TOKENS, MAX_TRAJECTORY_TOKENS - len(trajectory.prompt_ids) - len(trajectory.response_ids)
        )
        if max_tokens <= 0:
            trajectory.stop_reason = "context_full"
            break
        sample = await sampler(trajectory.prompt_ids + trajectory.response_ids, max_tokens)
        trajectory.turns += 1
        trajectory.response_ids += sample.token_ids
        trajectory.loss_mask += [1] * len(sample.token_ids)
        trajectory.logprobs += sample.logprobs
        if sample.finish_reason == "length":
            trajectory.stop_reason = "truncated"
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
        observation = run_tool(tools, recipient, content.text)
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
    trajectory.reward = reward(trajectory.session.curated_ids, facts)
    return trajectory


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="unsloth/gpt-oss-20b-BF16")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=8, help="number of queries to run, concurrently")
    args = parser.parse_args()

    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    queries = queries[queries.split == args.split].head(args.limit)
    index = Index()
    async with httpx.AsyncClient(base_url=args.server, timeout=None) as client:
        sampler = VLLMSampler(client, args.model)

        async def run(row) -> Trajectory:
            trajectory = await run_trajectory(sampler, index, row.query, json.loads(row.facts))
            print(
                f"{row.query_id}: reward={trajectory.reward:.3f} stop={trajectory.stop_reason} turns={trajectory.turns} "
                f"curated={len(trajectory.session.curated_ids)} tokens={len(trajectory.response_ids)} "
                f"sampled={sum(trajectory.loss_mask)}",
                flush=True,
            )
            return trajectory

        trajectories = await asyncio.gather(*(run(row) for row in queries.itertuples()))
    print(f"mean reward over {len(trajectories)}: {sum(t.reward for t in trajectories) / len(trajectories):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
