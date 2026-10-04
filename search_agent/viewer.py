"""Browse eval rollouts in the browser: pick a run, see its queries and trials, open a trial's trajectory and verifier.

Serves search_agent/viewer.html and a small JSON API over the JSONL files that `trajectory.py --output` writes to
search_agent/data/rollouts/. A new file there shows up in the run dropdown on reload.

  uv run python -u -m search_agent.viewer

Then open http://localhost:8081.
"""

import argparse
import json
import re
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from search_agent.retrieval import DATA_DIR

ROLLOUTS_DIR = DATA_DIR / "rollouts"
PAGE = Path(__file__).with_name("viewer.html")


@lru_cache(maxsize=1)
def load_queries() -> dict[str, dict]:
    queries = pd.read_parquet(DATA_DIR / "queries.parquet").to_dict("records")
    return {
        str(row["query_id"]): {"query": row["query"], "answer": row["answer"], "facts": json.loads(str(row["facts"]))}
        for row in queries
    }


def run_names() -> list[str]:
    """Rollout files, newest first."""
    paths = sorted(ROLLOUTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in paths]


def load_run(name: str) -> list[dict]:
    path = ROLLOUTS_DIR / f"{name}.jsonl"
    return _load_run(path, path.stat().st_mtime)


@lru_cache(maxsize=8)
def _load_run(path: Path, mtime: float) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def run_index(name: str) -> dict:
    """Per-query trial scores (no transcripts) and the run's best-of-N summary."""
    queries = load_queries()
    by_query: dict[str, list[dict]] = {}
    for r in load_run(name):
        by_query.setdefault(r["query_id"], []).append(r)
    rows = []
    for query_id, trials in by_query.items():
        trials.sort(key=lambda r: r["trial"])
        rows.append(
            {
                "query_id": query_id,
                "query": queries[query_id]["query"],
                "num_facts": len(queries[query_id]["facts"]),
                "trials": [
                    {key: r[key] for key in ("trial", "f1", "precision", "recall", "reward", "stop_reason", "turns")}
                    | {"num_curated": len(r["curated_ids"])}
                    for r in trials
                ],
            }
        )
    rollouts = [r for row in by_query.values() for r in row]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # Each metric's max over a query's trials is taken independently, as in trajectory.report.
    summary = {
        "queries": len(by_query),
        "rollouts": len(rollouts),
        "trials_per_query": max((len(t) for t in by_query.values()), default=0),
        "best_f1": mean([max(r["f1"] for r in t) for t in by_query.values()]),
        "best_precision": mean([max(r["precision"] for r in t) for t in by_query.values()]),
        "best_recall": mean([max(r["recall"] for r in t) for t in by_query.values()]),
        "trial1_f1": mean([t[0]["f1"] for t in by_query.values()]),
        "mean_f1": mean([r["f1"] for r in rollouts]),
        "mean_reward": mean([r["reward"] for r in rollouts]),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in rollouts).most_common()),
        "nothing_curated_queries": sum(not any(r["curated_ids"] for r in t) for t in by_query.values()),
    }
    return {"run": name, "summary": summary, "queries": rows}


def parse_steps(transcript: str) -> list[dict]:
    """Split a decoded Harmony response into turns: the assistant's messages, its tool call, and the tool result.

    Parsing is by string and lenient, so replies Harmony rejected (stop reason parse_error) still display; each call
    keeps its raw header so the malformed part is visible.
    """
    steps: list[dict] = []
    step: dict = {"messages": [], "call": None, "result": None}
    # The prompt ended with <|start|>assistant, so the transcript starts mid-message.
    for message in ("<|start|>assistant" + transcript).split("<|start|>")[1:]:
        header, _, body = message.partition("<|message|>")
        body = re.sub(r"<\|(end|call|return)\|>$", "", body)
        if header.startswith("functions."):
            try:
                step["result"] = json.loads(body)
            except json.JSONDecodeError:
                step["result"] = body
            steps.append(step)
            step = {"messages": [], "call": None, "result": None}
            continue
        recipient = re.search(r"to=(\S+)", header)
        if recipient:
            step["call"] = {"recipient": recipient.group(1), "arguments": body, "header": header.removeprefix("assistant")}
        else:
            channel = re.search(r"<\|channel\|>(\w+)", header)
            step["messages"].append({"channel": channel.group(1) if channel else "", "text": body})
    if step["messages"] or step["call"]:
        steps.append(step)
    return steps


def trial_detail(name: str, query_id: str, trial: int) -> dict | None:
    rollout = next((r for r in load_run(name) if r["query_id"] == query_id and r["trial"] == trial), None)
    if rollout is None:
        return None
    query = load_queries()[query_id]
    facts = query["facts"]
    # A chunk can be gold for several facts.
    gold: dict[str, list[int]] = {}
    for i, fact in enumerate(facts, 1):
        for chunk_id in fact["chunk_ids"]:
            gold.setdefault(chunk_id, []).append(i)
    curated = rollout["curated_ids"]
    trials = sorted(r["trial"] for r in load_run(name) if r["query_id"] == query_id)
    return {
        "run": name,
        "query_id": query_id,
        "trial": trial,
        "trials": trials,
        "query": query["query"],
        "answer": query["answer"],
        "facts": [
            {
                "fact": fact["fact"],
                "is_final_answer": fact.get("is_final_answer", False),
                "chunk_ids": fact["chunk_ids"],
                "found_by": [c for c in curated if c in fact["chunk_ids"]],
            }
            for fact in facts
        ],
        "gold": gold,
        "curated_ids": curated,
        "scores": {key: rollout[key] for key in ("f1", "precision", "recall", "reward", "stop_reason", "turns")},
        "steps": parse_steps(rollout["transcript"]),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/") if p]
        if not parts:
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if parts[0] != "api" or len(parts) < 2 or parts[1] != "runs":
            self._json(404, {"error": "not found"})
            return
        if len(parts) == 2:
            self._json(200, run_names())
            return
        if parts[2] not in run_names():
            self._json(404, {"error": f"unknown run {parts[2]}"})
            return
        if len(parts) == 3:
            self._json(200, run_index(parts[2]))
        elif len(parts) == 5 and parts[4].isdigit():
            detail = trial_detail(parts[2], parts[3], int(parts[4]))
            self._json(200, detail) if detail else self._json(404, {"error": "unknown trial"})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving {len(run_names())} runs from {ROLLOUTS_DIR} on http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
