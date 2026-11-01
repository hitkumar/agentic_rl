"""Grade one airline conversation, as tau2-bench v1.0.1 does (src/tau2/evaluator/), for the two checks airline uses:

- DB: replay the task's reference actions on a fresh database to get the target, and compare its hash with the
  database the agent left behind. Any sequence of tool calls that ends in the same state passes.
- COMMUNICATE: every string in the task's communicate_info appears (case-insensitive, commas removed from the agent's
  text) in at least one of the agent's text messages.

The reward is the product of the checks in the task's reward_basis, so 1 or 0. tau2 rebuilds the agent's database by
replaying its tool calls on a fresh copy; here the conversation's own database is graded directly, which is the same
state because the tools are deterministic.
"""

from typing import Any

from eval.tau_bench.airline.data_model import FlightDB, load_db
from eval.tau_bench.airline.tools import AirlineTools
from eval.tau_bench.tool import call_tool


def target_db(task: dict) -> tuple[FlightDB, list[str]]:
    """The starting database after the task's reference actions, plus any errors they raised (tau2 ignores them)."""
    db = load_db()
    tools = AirlineTools(db)
    errors = []
    for action in task["evaluation_criteria"]["actions"] or []:
        content, error = call_tool(tools, action["name"], action["arguments"])
        if error:
            errors.append(f"{action['name']}: {content}")
    return db, errors


def communicate_checks(task: dict, messages: list[dict]) -> list[dict]:
    agent_texts = [m["content"] for m in messages if m["role"] == "assistant" and (m.get("content") or "").strip()]
    return [
        {"info": info, "met": any(info.lower() in text.lower().replace(",", "") for text in agent_texts)}
        for info in task["evaluation_criteria"].get("communicate_info") or []
    ]


def grade(task: dict, final_db: FlightDB, messages: list[dict]) -> dict[str, Any]:
    """Reward and its parts for one conversation; `messages` use the chat roles assistant (agent), user and tool."""
    criteria = task["evaluation_criteria"]
    basis = criteria["reward_basis"]
    unsupported = set(basis) - {"DB", "COMMUNICATE"}
    if unsupported:
        raise ValueError(f"reward_basis {sorted(unsupported)} is not implemented (airline uses DB and COMMUNICATE)")

    target, target_errors = target_db(task)
    db_match = target.hash() == final_db.hash()
    checks = communicate_checks(task, messages)
    breakdown = {}
    if "DB" in basis:
        breakdown["DB"] = 1.0 if db_match else 0.0
    if "COMMUNICATE" in basis:
        breakdown["COMMUNICATE"] = 1.0 if all(c["met"] for c in checks) else 0.0
    reward = 1.0
    for value in breakdown.values():
        reward *= value
    return {
        "reward": reward,
        "reward_basis": basis,
        "reward_breakdown": breakdown,
        "db_match": db_match,
        "communicate_checks": checks,
        "target_action_errors": target_errors,
    }
