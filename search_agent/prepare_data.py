"""Prepare Harness-1 SEC data for search-agent RL.

Adapted from jasper-lu/sec-search-rl (src/sec_rl/sec.py).

Outputs in --out-dir:
  queries.parquet  One row per query. Columns: query_id, split (train/dev), query, answer, facts.
                   facts is a JSON list of {fact, chunk_ids, is_final_answer}; any one chunk in
                   chunk_ids contains the fact. These are the query's gold chunks (correct answers).
  corpus.parquet   The chunks the agent searches: a subset of the 2.1M-chunk SEC corpus.
                   Columns: chunk_id ("<filing id>_<position in filing>"), text.
                   Contains:
                   - every gold chunk of the selected queries, so every query is answerable;
                   - the chunks within --neighbor-radius positions of each gold chunk in the same
                     filing: similar text without the fact (hard distractors);
                   - about --random-distractors random chunks from the full corpus.

Usage:
  uv run python search_agent/prepare_data.py
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO_ID = "pat-jj/harness-1-train-data"
QUERY_FILE = "data/train-00000-of-00001.parquet"
CORPUS_GLOB = "corpora/sec/train/*.parquet"
SEC_CORPUS_SIZE = 2_115_106
# Share of queries with 3, 5, and 7 facts in each split (Jasper's mix).
FACT_MIX = {3: 0.5, 5: 0.3125, 7: 0.1875}


def load_queries(raw_dir: Path) -> list[dict]:
    table = pq.read_table(
        raw_dir / QUERY_FILE,
        columns=["query_id", "query", "answer", "document_ids_json"],
        filters=[("stage", "=", "rl"), ("dataset_name", "=", "sec")],
    )
    return [
        {
            "query_id": row["query_id"],
            # Query ids are "<family>_<variant>"; a family's variants ask the same question with 3, 5, or 7 facts.
            "family_id": row["query_id"].rsplit("_", 1)[0],
            "query": row["query"],
            "answer": row["answer"] or "",
            "facts": json.loads(row["document_ids_json"]),
        }
        for row in table.to_pylist()
    ]


def gold_ids(rows: list[dict]) -> set[str]:
    return {chunk_id for row in rows for fact in row["facts"] for chunk_id in fact["chunk_ids"]}


def family_groups(rows: list[dict]) -> dict[str, str]:
    """Map each family to a group id; families that share any gold chunk are in the same group."""
    parent = {row["family_id"]: row["family_id"] for row in rows}

    def find(family: str) -> str:
        while parent[family] != family:
            family = parent[family]
        return family

    chunk_owner: dict[str, str] = {}
    for row in rows:
        for chunk_id in gold_ids([row]):
            parent[find(row["family_id"])] = find(chunk_owner.setdefault(chunk_id, row["family_id"]))
    return {family: find(family) for family in parent}


def split_queries(rows: list[dict], train_size: int, dev_size: int, seed: int) -> list[dict]:
    """Pick at most one query per group, so train and dev share no family or gold chunk.

    Dev is picked first. Within each split, 7-fact queries are picked first because they are rarest.
    If train runs short of a fact count, the gap is filled with 5-fact, then 3-fact queries.
    """
    group = family_groups(rows)
    rng = random.Random(seed)
    used_groups = set()
    selected = []

    def take(candidates: list[dict], count: int, split: str) -> int:
        taken = 0
        for row in candidates:
            if taken == count:
                break
            if group[row["family_id"]] in used_groups:
                continue
            used_groups.add(group[row["family_id"]])
            selected.append({**row, "split": split})
            taken += 1
        return taken

    for split, size in (("dev", dev_size), ("train", train_size)):
        quotas = {n: round(size * FACT_MIX[n]) for n in (7, 5)}
        quotas[3] = size - quotas[7] - quotas[5]
        candidates = {}
        shortfall = 0
        for fact_count, quota in quotas.items():
            candidates[fact_count] = [row for row in rows if len(row["facts"]) == fact_count]
            rng.shuffle(candidates[fact_count])
            shortfall += quota - take(candidates[fact_count], quota, split)
        if shortfall and split == "dev":
            raise ValueError(f"dev: {shortfall} queries short of the fact mix")
        for fact_count in (5, 3):
            shortfall -= take(candidates[fact_count], shortfall, split)
        if shortfall:
            raise ValueError(f"train: {shortfall} queries short; too few independent query groups")
    return selected


def build_corpus(
    corpus_paths: list[Path], queries: list[dict], neighbor_radius: int, random_distractors: int, seed: int
) -> pa.Table:
    gold = gold_ids(queries)
    # Chunk ids are "<filing>_<chunk index>"; nearby chunks of the same filing are hard distractors.
    keep = set()
    for chunk_id in gold:
        filing, index = chunk_id.rsplit("_", 1)
        keep.update(f"{filing}_{int(index) + d}" for d in range(-neighbor_radius, neighbor_radius + 1))

    rng = random.Random(seed)
    keep_probability = random_distractors / SEC_CORPUS_SIZE
    tables = []
    for path in corpus_paths:
        table = pq.read_table(path, columns=["chunk_id", "document_text"])
        mask = [chunk_id in keep or rng.random() < keep_probability for chunk_id in table["chunk_id"].to_pylist()]
        tables.append(table.filter(pa.array(mask)))
    return pa.concat_tables(tables).rename_columns(["chunk_id", "text"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--dev-size", type=int, default=64)
    parser.add_argument("--neighbor-radius", type=int, default=4)
    parser.add_argument("--random-distractors", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = args.out_dir / "raw"
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=raw_dir, allow_patterns=[QUERY_FILE, CORPUS_GLOB])
    corpus_paths = sorted(raw_dir.glob(CORPUS_GLOB))

    # A few queries cite gold chunks that are not in the published corpus; drop them.
    available = set()
    for path in corpus_paths:
        available.update(pq.read_table(path, columns=["chunk_id"])["chunk_id"].to_pylist())
    rows = [row for row in load_queries(raw_dir) if gold_ids([row]) <= available]

    queries = split_queries(rows, args.train_size, args.dev_size, args.seed)
    corpus = build_corpus(corpus_paths, queries, args.neighbor_radius, args.random_distractors, args.seed)

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": q["query_id"],
                    "split": q["split"],
                    "query": q["query"],
                    "answer": q["answer"],
                    "facts": json.dumps(q["facts"]),
                }
                for q in queries
            ]
        ),
        args.out_dir / "queries.parquet",
    )
    pq.write_table(corpus, args.out_dir / "corpus.parquet")
    print(f"queries: {len(queries)} ({args.dev_size} dev), corpus: {corpus.num_rows} chunks")


if __name__ == "__main__":
    main()
