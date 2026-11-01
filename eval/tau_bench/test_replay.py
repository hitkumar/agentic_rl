"""Checks of the airline tools and grader that need no models: every task's reference actions run cleanly, replaying
them scores reward 1, and doing nothing fails every task whose reference actions change the database.

From the repo root:

  uv run --project eval/tau_bench python -m pytest eval/tau_bench/test_replay.py
"""

import json

import pytest

from eval.tau_bench.airline.data_model import load_db, load_tasks
from eval.tau_bench.airline.tools import AirlineTools
from eval.tau_bench.grade import grade, target_db
from eval.tau_bench.tool import call_tool

TASKS = load_tasks()
WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}


def ids(task: dict) -> str:
    return f"task{task['id']}"


@pytest.mark.parametrize("task", TASKS, ids=ids)
def test_reference_actions_run_cleanly(task: dict) -> None:
    _, errors = target_db(task)
    assert errors == []


@pytest.mark.parametrize("task", TASKS, ids=ids)
def test_reference_solution_scores_1(task: dict) -> None:
    """A conversation that makes exactly the reference tool calls and says every required item gets reward 1."""
    db = load_db()
    tools = AirlineTools(db)
    messages = []
    for i, action in enumerate(task["evaluation_criteria"]["actions"] or []):
        call = {"id": f"call_{i}", "name": action["name"], "arguments": action["arguments"]}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        content, _ = call_tool(tools, action["name"], json.loads(json.dumps(action["arguments"])))
        messages.append({"role": "tool", "content": content, "tool_call_id": call["id"]})
    required = task["evaluation_criteria"].get("communicate_info") or []
    messages.append({"role": "assistant", "content": "Here you go: " + ", ".join(required)})
    result = grade(task, db, messages)
    assert result["reward"] == 1.0, result


@pytest.mark.parametrize(
    "task",
    [t for t in TASKS if any(a["name"] in WRITE_TOOLS for a in t["evaluation_criteria"]["actions"] or [])],
    ids=ids,
)
def test_doing_nothing_fails_write_tasks(task: dict) -> None:
    result = grade(task, load_db(), [{"role": "assistant", "content": "Hi! How can I help you today?"}])
    assert not result["db_match"]
