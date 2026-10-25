"""τ-bench source for the trajectory viewer: reads the results.json files that `tau2 run --save-to <name>` writes to
<tau2-bench>/data/simulations/<name>/. See viewer/server.py for the shapes returned.

A trial's score is its reward: 1 if the agent's final database matches the target database (the starting database after
replaying the task's reference actions) and every graded check in the task's reward basis passed, else 0.
"""

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

DEFAULT_RUNS_DIR = Path.home() / "oss" / "tau2-bench" / "data" / "simulations"

ROLES = {"assistant": "agent", "user": "user", "tool": "tool", "system": "system"}


def run_names(runs_dir: Path) -> list[str]:
    paths = sorted(runs_dir.glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.parent.name for p in paths]


def load_run(runs_dir: Path, run: str) -> dict:
    path = runs_dir / run / "results.json"
    return _load(path, path.stat().st_mtime)


@lru_cache(maxsize=8)
def _load(path: Path, mtime: float) -> dict:
    return json.loads(path.read_text())


def task_title(task: dict) -> str:
    instructions = task["user_scenario"]["instructions"]
    reason = instructions.get("reason_for_call") if isinstance(instructions, dict) else instructions
    return " ".join(str(reason or "").split())


def pass_hat_k(trials: list[bool], k: int) -> float:
    """Chance that k trials drawn from this task's trials all succeed: C(successes, k) / C(trials, k)."""
    return math.comb(sum(trials), k) / math.comb(len(trials), k)


def run_index(runs_dir: Path, run: str) -> dict:
    data = load_run(runs_dir, run)
    by_task: dict[str, list[dict]] = {}
    for sim in data["simulations"]:
        by_task.setdefault(str(sim["task_id"]), []).append(sim)
    tasks = []
    for task in data["tasks"]:
        sims = sorted(by_task.get(str(task["id"]), []), key=lambda s: s["trial"])
        if not sims:
            continue
        tasks.append(
            {
                "id": str(task["id"]),
                "title": task_title(task),
                "trials": [
                    {"trial": s["trial"], "score": s["reward_info"]["reward"], "note": s["termination_reason"]}
                    for s in sims
                ],
            }
        )
    sims = data["simulations"]
    successes = [[t["score"] == 1.0 for t in task["trials"]] for task in tasks]
    num_trials = min((len(t) for t in successes), default=0)
    summary = [{"label": "pass^1 (mean reward)", "value": f"{sum(map(sum, successes)) / max(len(sims), 1):.3f}"}]
    if num_trials > 1:
        value = sum(pass_hat_k(t, num_trials) for t in successes) / len(successes)
        summary.append({"label": f"pass^{num_trials} (all {num_trials} trials pass)", "value": f"{value:.3f}"})
    summary += [
        {"label": "tasks × trials", "value": f"{len(tasks)} × {num_trials}"},
        {"label": "mean conversation time", "value": f"{sum(s['duration'] for s in sims) / max(len(sims), 1):.0f} s"},
    ]
    summary += [
        {"label": f"ended: {k}", "value": str(v)} for k, v in Counter(s["termination_reason"] for s in sims).items()
    ]
    info = data["info"]
    return {
        "summary": summary,
        "info": [
            {"label": "domain", "value": info["environment_info"]["domain_name"]},
            {"label": "agent", "value": f"{info['agent_info']['llm']} {json.dumps(sampling(info['agent_info']))}"},
            {"label": "user", "value": f"{info['user_info']['llm']} {json.dumps(sampling(info['user_info']))}"},
            {"label": "tau2 commit", "value": str(info.get("git_commit", ""))[:7]},
        ],
        "score_label": "reward",
        "tasks": tasks,
    }


def sampling(role_info: dict) -> dict:
    return {k: v for k, v in (role_info.get("llm_args") or {}).items() if k != "api_base"}


def reasoning_of(message: dict) -> str:
    raw = message.get("raw_data")
    if not isinstance(raw, dict) or not raw.get("choices"):
        return ""
    reply = raw["choices"][0].get("message") or {}
    return (reply.get("reasoning_content") or reply.get("reasoning") or "").strip()


def pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2)
    except (TypeError, ValueError):
        return text


def trial_detail(runs_dir: Path, run: str, task_id: str, trial: int) -> dict | None:
    data = load_run(runs_dir, run)
    sim = next((s for s in data["simulations"] if str(s["task_id"]) == task_id and s["trial"] == trial), None)
    if sim is None:
        return None
    task = next(t for t in data["tasks"] if str(t["id"]) == task_id)
    reward = sim["reward_info"]
    db = reward.get("db_check") or {}

    messages = []
    for m in sim["messages"]:
        usage = m.get("usage") or {}
        meta = f"turn {m.get('turn_idx')}"
        if usage.get("completion_tokens"):
            meta += f" · {usage['completion_tokens']} tokens"
        messages.append(
            {
                "role": ROLES.get(m["role"], m["role"]),
                "text": pretty(m.get("content") or "").strip()
                if m["role"] == "tool"
                else (m.get("content") or "").strip(),
                "reasoning": reasoning_of(m),
                "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in m.get("tool_calls") or []],
                "error": bool(m.get("error")),
                "meta": meta,
            }
        )

    instructions = task["user_scenario"]["instructions"]
    if isinstance(instructions, dict):
        parts = [
            (k.replace("_", " ").capitalize(), instructions.get(k))
            for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions")
        ]
        user_text = "\n\n".join(f"{label}:\n{value.strip()}" for label, value in parts if value)
    else:
        user_text = str(instructions)
    context = [{"title": "Simulated user's instructions (private)", "text": user_text}]
    purpose = (task.get("description") or {}).get("purpose")
    if purpose:
        context.append({"title": "Task purpose", "text": purpose})

    criteria = task["evaluation_criteria"] or {}
    basis = reward.get("reward_basis") or []
    breakdown = reward.get("reward_breakdown") or {}
    verifier = [
        {
            "title": f"Reward = product of {' × '.join(basis)} = {reward['reward']:g}",
            "rows": [
                {"label": name, "value": f"{value:g}", "status": "good" if value == 1 else "bad"}
                for name, value in breakdown.items()
            ],
        },
        {
            "title": "Database check",
            "rows": [
                {
                    "label": "agent's final database matches the target",
                    "value": "yes" if db.get("db_match") else "no",
                    "status": "good" if db.get("db_match") else "bad",
                }
            ],
        },
    ]
    actions = criteria.get("actions") or []
    verifier.append(
        {
            "title": f"Reference actions ({len(actions)}), replayed on a fresh database to build the target",
            "text": "\n".join(f"{a['name']}({json.dumps(a.get('arguments', {}))})" for a in actions)
            or "None: the target is the starting database, so the agent passes only if it changes nothing.",
        }
    )
    checks = reward.get("communicate_checks") or []
    if criteria.get("communicate_info") or checks:
        verifier.append(
            {
                "title": "Information the agent had to say (COMMUNICATE)",
                "rows": [
                    {
                        "label": c.get("info", ""),
                        "value": "said" if c.get("met") else "missing",
                        "status": "good" if c.get("met") else "bad",
                    }
                    for c in checks
                ]
                or [{"label": i, "value": "", "status": None} for i in criteria.get("communicate_info") or []],
            }
        )
    if criteria.get("nl_assertions"):
        graded = "NL_ASSERTION" in basis
        verifier.append(
            {
                "title": "Natural-language assertions" + ("" if graded else " (not graded for this task)"),
                "text": "\n".join(criteria["nl_assertions"]),
            }
        )

    by_task = sorted(s["trial"] for s in data["simulations"] if str(s["task_id"]) == task_id)
    task_ids = [
        str(t["id"]) for t in data["tasks"] if any(str(s["task_id"]) == str(t["id"]) for s in data["simulations"])
    ]
    i = task_ids.index(task_id)
    return {
        "title": task_title(task),
        "trials": by_task,
        "prev_task": task_ids[i - 1] if i > 0 else None,
        "next_task": task_ids[i + 1] if i + 1 < len(task_ids) else None,
        "stats": [
            {"label": "reward", "value": f"{reward['reward']:g}", "status": "good" if reward["reward"] == 1 else "bad"},
            {
                "label": "database match",
                "value": "yes" if db.get("db_match") else "no",
                "status": "good" if db.get("db_match") else "bad",
            },
            {"label": "ended", "value": sim["termination_reason"], "status": None},
            {"label": "messages", "value": str(len(sim["messages"])), "status": None},
            {"label": "time", "value": f"{sim['duration']:.0f} s", "status": None},
        ],
        "context": context,
        "messages": messages,
        "end": f"Ended: {sim['termination_reason']}.",
        "verifier": verifier,
    }
