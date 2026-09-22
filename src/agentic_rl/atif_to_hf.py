"""Convert Harbor ATIF trajectories to Hugging Face Session Traces JSONL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _content_to_text(content: Any) -> str:
    """Render ATIF text or multimodal content as text for the HF trace viewer."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)

    rendered: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            rendered.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "text":
            rendered.append(str(part.get("text", "")))
            continue
        source = part.get("source") or {}
        path = source.get("path", "") if isinstance(source, dict) else ""
        media_type = source.get("media_type", part_type) if isinstance(source, dict) else part_type
        rendered.append(f"[{media_type}]({path})" if path else f"[{media_type}]")
    return "\n".join(piece for piece in rendered if piece)


def _timestamp_ms(timestamp: Any) -> int | None:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _tool_calls(calls: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    if not isinstance(calls, list):
        return converted
    for call in calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("tool_call_id")
        name = call.get("function_name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        arguments = call.get("arguments", {})
        converted.append(
            {
                "id": call_id,
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    return converted


def convert_atif(
    trajectory: dict[str, Any],
    *,
    fallback_id: str = "trajectory",
    name: str | None = None,
    harness: str = "harbor",
) -> list[dict[str, Any]]:
    """Return an ATIF trajectory as Hugging Face Session Trace records."""
    schema_version = trajectory.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.startswith("ATIF-v"):
        raise ValueError("input is not an ATIF trajectory: missing ATIF schema_version")

    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("input is not an ATIF trajectory: steps must be a list")

    agent = trajectory.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    session_id = trajectory.get("session_id") or trajectory.get("trajectory_id") or fallback_id
    session_id = str(session_id)
    agent_name = str(agent.get("name") or "agent")
    model_name = agent.get("model_name")

    header: dict[str, Any] = {
        "type": "session",
        "harness": harness,
        "id": session_id,
        "name": name or f"{agent_name}: {session_id}",
        "atifSchemaVersion": schema_version,
        "agent": agent_name,
    }
    if model_name:
        header["model"] = model_name
    if trajectory.get("final_metrics") is not None:
        header["finalMetrics"] = trajectory["final_metrics"]

    records = [header]
    role_by_source = {"agent": "assistant", "system": "system", "user": "user"}

    for step in steps:
        if not isinstance(step, dict):
            continue
        source = step.get("source")
        role = role_by_source.get(source)
        if role is None:
            raise ValueError(f"unsupported ATIF step source: {source!r}")

        message: dict[str, Any] = {
            "role": role,
            "content": _content_to_text(step.get("message")),
        }
        reasoning = step.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            message["reasoningContent"] = reasoning
        calls = _tool_calls(step.get("tool_calls"))
        if calls:
            message["toolCalls"] = calls
        timestamp = _timestamp_ms(step.get("timestamp"))
        if timestamp is not None:
            message["timestamp"] = timestamp
        step_model = step.get("model_name") or model_name
        if role == "assistant" and step_model:
            message["model"] = step_model
        records.append({"type": "message", "message": message})

        observation = step.get("observation")
        results = observation.get("results", []) if isinstance(observation, dict) else []
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            tool_message: dict[str, Any] = {
                "role": "tool",
                "content": _content_to_text(result.get("content")),
            }
            source_call_id = result.get("source_call_id")
            if isinstance(source_call_id, str) and source_call_id:
                tool_message["toolCallId"] = source_call_id
            records.append({"type": "message", "message": tool_message})

    return records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
            count += 1
    return count


def _default_output(input_path: Path) -> Path:
    return input_path.with_suffix(".hf.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Harbor ATIF trajectory.json to Hugging Face Session Traces JSONL."
    )
    parser.add_argument("input", type=Path, help="Path to an ATIF trajectory.json file")
    parser.add_argument("-o", "--output", type=Path, help="Output path; defaults to <input>.hf.jsonl")
    parser.add_argument("--name", help="Human-readable session name shown by Hugging Face")
    parser.add_argument("--harness", default="harbor", help="Session harness identifier (default: harbor)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or _default_output(input_path)
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("input and output paths must be different")

    try:
        trajectory = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(trajectory, dict):
            raise ValueError("the input JSON root must be an object")
        records = convert_atif(
            trajectory,
            fallback_id=input_path.stem,
            name=args.name,
            harness=args.harness,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = write_jsonl(records, output_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"conversion failed: {error}") from error

    print(f"Wrote {count - 1} messages to {output_path}")


if __name__ == "__main__":
    main()
