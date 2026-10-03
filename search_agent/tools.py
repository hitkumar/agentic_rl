"""The six agent tools for one episode, sharing a pool of seen chunks and a curated set.

Adapted from jasper-lu/sec-search-rl (src/sec_rl/tools.py).

Tools, as methods of SearchTools (each returns a JSON-serializable observation):
  bm25_search(query, k)              Search tools from retrieval.py; the chunks they return become seen.
  grep_corpus(pattern, k, case_sensitive)
  read_document(chunk_id)            Read a seen chunk.
  curate(chunk_ids)                  Add seen chunks to the curated set, the episode's output.
  drop_curated(chunk_ids)            Remove chunks from the curated set.
  finish()                           End the episode.

The curated set is scored whether or not finish is called.
"""

from dataclasses import dataclass, field
from typing import TypedDict

from search_agent.retrieval import Index

MAX_CURATED = 30

_K = {"type": "integer", "description": "Number of results to return (1-25).", "default": 10}


class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: dict


# JSON-schema function specs for the SearchTools methods, in the common name/description/parameters shape.
TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "bm25_search",
        "description": "Search the full corpus with BM25 lexical ranking.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A lexical search query; concise keywords usually work best.",
                },
                "k": _K,
            },
            "required": ["query"],
        },
    },
    {
        "name": "grep_corpus",
        "description": "Scan the full corpus with a bounded regular expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "A Python-style regular expression to find exact terms, phrases, or variants. "
                    "Keep it short and targeted.",
                },
                "k": _K,
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether letter case must match exactly.",
                    "default": False,
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_document",
        "description": "Read a candidate document's title and text.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "A chunk ID previously returned by one of the search tools.",
                }
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "curate",
        "description": "Add relevant documents to the curated set that is returned as your output.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk IDs to add. Must have been returned by a search tool earlier in this episode.",
                }
            },
            "required": ["chunk_ids"],
        },
    },
    {
        "name": "drop_curated",
        "description": "Remove documents from the curated set (e.g. redundant or off-topic ones).",
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Chunk IDs to remove from the curated set.",
                }
            },
            "required": ["chunk_ids"],
        },
    },
    {
        "name": "finish",
        "description": "End the search. The curated set is returned as the final evidence.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


@dataclass
class Session:
    seen_ids: set[str] = field(default_factory=set)
    curated_ids: list[str] = field(default_factory=list)
    finished: bool = False
    # curate calls that named an unseen chunk or overflowed MAX_CURATED.
    invalid_curations: int = 0


class SearchTools:
    def __init__(self, index: Index, session: Session):
        self.index = index
        self.session = session

    def bm25_search(self, query: str, k: int = 10) -> dict:
        hits = self.index.bm25_search(query, k)
        return self._results(hits)

    def grep_corpus(self, pattern: str, k: int = 10, case_sensitive: bool = False) -> dict:
        try:
            hits = self.index.grep_corpus(pattern, k, case_sensitive)
        except ValueError as exc:
            return {"error": str(exc)}
        return self._results(hits)

    def read_document(self, chunk_id: str) -> dict:
        if chunk_id not in self.session.seen_ids:
            return {"error": "Search for this chunk before reading it."}
        return self.index.read_document(chunk_id)

    def curate(self, chunk_ids: list[str]) -> dict:
        requested = list(dict.fromkeys(chunk_ids))
        unseen = [i for i in requested if i not in self.session.seen_ids]
        already = [i for i in requested if i in self.session.curated_ids]
        new = [i for i in requested if i in self.session.seen_ids and i not in self.session.curated_ids]
        capacity = max(MAX_CURATED - len(self.session.curated_ids), 0)
        added, over_capacity = new[:capacity], new[capacity:]
        self.session.curated_ids.extend(added)
        if unseen or over_capacity:
            self.session.invalid_curations += 1
        observation = {"added": added, **self._curated()}
        if already:
            observation["already_curated"] = already
        if unseen:
            observation["unseen"] = unseen
            observation["message"] = "Only curate chunk IDs returned by bm25_search or grep_corpus."
        if over_capacity:
            observation["over_capacity"] = over_capacity
            observation["maximum"] = MAX_CURATED
        return observation

    def drop_curated(self, chunk_ids: list[str]) -> dict:
        requested = set(chunk_ids)
        removed = [i for i in self.session.curated_ids if i in requested]
        self.session.curated_ids = [i for i in self.session.curated_ids if i not in requested]
        observation = {"removed": removed, **self._curated()}
        not_curated = [i for i in dict.fromkeys(chunk_ids) if i not in removed]
        if not_curated:
            observation["not_curated"] = not_curated
        return observation

    def finish(self) -> dict:
        self.session.finished = True
        return self._curated()

    def _results(self, hits: list[dict]) -> dict:
        self.session.seen_ids.update(hit["chunk_id"] for hit in hits)
        return {"results": hits, "seen_count": len(self.session.seen_ids), **self._curated()}

    def _curated(self) -> dict:
        return {"curated_count": len(self.session.curated_ids), "curated_ids": list(self.session.curated_ids)}
