"""Convert a training run's SkyRL eval dump into a rollouts file the viewer can open.

SkyRL dumps each eval to outputs/search_agent/exports/<run>/dumped_evals/global_step_<N>_evals/sec_search.jsonl, with
the decoded response, the reward, the stop reason and the dataset row, but not the curated set, turns or F1. This
rebuilds them in the format trajectory.py --output writes: the curated set from the last tool result (every result
ends with curated_ids), turns from the sampled turns' stop tokens, F1 from rewards.score, and trials from the order of
each query's samples. Writes search_agent/data/rollouts/<run>_step<N>.jsonl.

  uv run python -u -m search_agent.scripts.evals_to_rollouts full_lr3e-6 base_format_base_fp_harness

--step picks the eval step; the default is the run's last.
"""

import argparse
import json
import re
from pathlib import Path

from search_agent.rewards import score
from search_agent.viewer import ROLLOUTS_DIR, parse_steps

EXPORTS_DIR = Path("outputs/search_agent/exports")
# A sampled turn ends at a tool call or a final answer; a turn cut off at the token limit ends at neither.
TURN_END = ("<|call|>", "<|return|>")


def eval_steps(run: str) -> dict[int, Path]:
    dumps = EXPORTS_DIR / run / "dumped_evals"
    steps = {}
    for path in dumps.glob("global_step_*_evals"):
        match = re.fullmatch(r"global_step_(\d+)_evals", path.name)
        if match:
            steps[int(match.group(1))] = path / "sec_search.jsonl"
    if not steps:
        raise SystemExit(f"no eval dumps in {dumps}")
    return steps


def convert(dump: Path) -> list[dict]:
    rollouts = []
    trials: dict[str, int] = {}
    with dump.open() as f:
        for line in f:
            row = json.loads(line)
            extras = row["env_extras"]
            transcript = row["output_response"]
            results = [step["result"] for step in parse_steps(transcript) if isinstance(step["result"], dict)]
            curated_ids = next((r["curated_ids"] for r in reversed(results) if "curated_ids" in r), [])
            f1 = score(curated_ids, json.loads(extras["facts"]), beta=1.0)
            trials[extras["query_id"]] = trials.get(extras["query_id"], 0) + 1
            rollouts.append(
                {
                    "query_id": extras["query_id"],
                    "trial": trials[extras["query_id"]],
                    "f1": f1.f_beta,
                    "precision": f1.precision,
                    "recall": f1.recall,
                    "reward": row["score"],
                    "stop_reason": row["stop_reason"],
                    "turns": sum(transcript.count(end) for end in TURN_END) + (not transcript.endswith(TURN_END)),
                    "curated_ids": curated_ids,
                    "transcript": transcript,
                }
            )
    return rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run names, as in outputs/search_agent/exports/<run>")
    parser.add_argument("--step", type=int, help="eval step to convert (default: the run's last)")
    args = parser.parse_args()

    ROLLOUTS_DIR.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        steps = eval_steps(run)
        step = max(steps) if args.step is None else args.step
        if step not in steps:
            raise SystemExit(f"{run} has evals at steps {sorted(steps)}, not {step}")
        rollouts = convert(steps[step])
        out = ROLLOUTS_DIR / f"{run}_step{step}.jsonl"
        with out.open("w") as f:
            for rollout in rollouts:
                f.write(json.dumps(rollout, ensure_ascii=False) + "\n")

        def mean(key: str) -> float:
            return sum(r[key] for r in rollouts) / len(rollouts)

        print(
            f"{run} step {step}: {len(rollouts)} rollouts, f1={mean('f1'):.3f} precision={mean('precision'):.3f} "
            f"recall={mean('recall'):.3f} reward={mean('reward'):.3f} turns={mean('turns'):.1f} -> {out}"
        )


if __name__ == "__main__":
    main()
