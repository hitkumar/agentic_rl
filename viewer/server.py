"""Browse agent trajectories in the browser: pick a run, see its tasks and trials, open a trial's conversation and how
it was graded.

The page (viewer/viewer.html) and this server are shared across projects; each project plugs in a source module,
viewer/sources/<name>.py, that reads its own run files and returns plain JSON in the shapes below.

  uv run python -u -m viewer.server --source tau_bench

Then open http://localhost:8081. A new run shows up in the run dropdown on reload.

A source module defines:

  DEFAULT_RUNS_DIR: Path            where its runs live (overridable with --runs-dir)
  run_names(runs_dir) -> list[str]  runs, newest first
  run_index(runs_dir, run) -> {
      "summary": [{"label", "value"}],          headline numbers, shown as tiles
      "info": [{"label", "value"}],             run settings (models, sampling), shown as one line
      "score_label": str,                       what a trial's score is, e.g. "reward"
      "tasks": [{"id", "title", "trials": [{"trial", "score" (0-1), "note"}]}],
  }
  trial_detail(runs_dir, run, task_id, trial) -> None if unknown, else {
      "title": str,                             the task, in one line
      "trials": [int],                          this task's trials, for prev/next trial
      "prev_task", "next_task": str | None,     neighbouring task ids, for prev/next task
      "stats": [{"label", "value", "status": "good" | "bad" | None}],
      "context": [{"title", "text"}],           task setup, shown above the conversation
      "messages": [{"role": "agent" | "user" | "tool" | "system", "text", "reasoning",
                    "tool_calls": [{"name", "arguments"}], "error": bool, "meta"}],
      "end": str,                               how the conversation ended
      "verifier": [{"title", "rows": [{"label", "value", "status"}]} | {"title", "text"}],
  }
"""

import argparse
import importlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote

PAGE = Path(__file__).with_name("viewer.html")


def make_handler(source: ModuleType, runs_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/") if p]
            if not parts:
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if parts[0] != "api" or len(parts) < 2 or parts[1] != "runs":
                self._json(404, {"error": "not found"})
                return
            runs = source.run_names(runs_dir)
            if len(parts) == 2:
                self._json(200, {"source": source.__name__.rsplit(".", 1)[-1], "runs_dir": str(runs_dir), "runs": runs})
                return
            if parts[2] not in runs:
                self._json(404, {"error": f"unknown run {parts[2]}"})
                return
            if len(parts) == 3:
                self._json(200, source.run_index(runs_dir, parts[2]))
            elif len(parts) == 5 and parts[4].lstrip("-").isdigit():
                detail = source.trial_detail(runs_dir, parts[2], parts[3], int(parts[4]))
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

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="module name under viewer/sources/, e.g. tau_bench")
    parser.add_argument("--runs-dir", type=Path, help="override the source's DEFAULT_RUNS_DIR")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    source = importlib.import_module(f"viewer.sources.{args.source}")
    runs_dir = args.runs_dir or source.DEFAULT_RUNS_DIR
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(source, runs_dir))
    print(f"serving {len(source.run_names(runs_dir))} runs from {runs_dir} on http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
