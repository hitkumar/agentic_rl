"""BM25 index and the three search tools over corpus.parquet.

Adapted from jasper-lu/sec-search-rl (src/sec_rl/retrieval.py).

build_index writes index.sqlite3 next to corpus.parquet: a chunks table (chunk_id -> text) for lookups
and an FTS5 table (SQLite full-text search, ranked by BM25) for search.

Tools, as methods of Index:
  bm25_search(query, k)    Top-k chunks for the query's words, each with a title and a snippet.
  grep_corpus(pattern, k)  Up to k chunks whose text matches a regex.
  read_document(chunk_id)  The text of one chunk.

Usage:
  uv run python -u search_agent/retrieval.py
"""

import re
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq
import regex

DATA_DIR = Path(__file__).parent / "data"
TOKEN = re.compile(r"[A-Za-z0-9]+")
MAX_K = 25
SNIPPET_CHARS = 220
READ_MAX_CHARS = 4_000
MAX_PATTERN_CHARS = 160
# grep_corpus runs its regex only on the best BM25 matches for the pattern's words.
GREP_CANDIDATES = 5_000
# Seconds a regex may spend on one chunk, and on the whole grep, so a pathological pattern cannot hang an episode.
GREP_TIMEOUT = 0.05
GREP_TOTAL_TIMEOUT = 2.0


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text)]


def build_index(data_dir: Path) -> None:
    corpus = pq.read_table(data_dir / "corpus.parquet")
    path = data_dir / "index.sqlite3"
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL) WITHOUT ROWID;
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED, text, tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )
    rows = list(zip(corpus["chunk_id"].to_pylist(), corpus["text"].to_pylist()))
    db.executemany("INSERT INTO chunks VALUES (?, ?)", rows)
    db.executemany("INSERT INTO chunks_fts VALUES (?, ?)", rows)
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
    db.commit()
    db.close()


class Index:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.db = sqlite3.connect(f"file:{data_dir / 'index.sqlite3'}?mode=ro", uri=True, check_same_thread=False)

    def bm25_search(self, query: str, k: int = 10) -> list[dict]:
        k = clamp_k(k)
        tokens = tokenize(query)
        if not tokens:
            return []
        # OR, not FTS5's default AND: a long query rarely has every word in one chunk.
        match = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
        hits = []
        for chunk_id in self._rank(match, k):
            title, body = split_header(self._text(chunk_id))
            positions = [p for p in (body.lower().find(token) for token in tokens) if p >= 0]
            hits.append({"chunk_id": chunk_id, "title": title, "snippet": window(body, min(positions, default=0))})
        return hits

    def grep_corpus(self, pattern: str, k: int = 10, case_sensitive: bool = False) -> list[dict]:
        k = clamp_k(k)
        if len(pattern) > MAX_PATTERN_CHARS:
            raise ValueError(f"pattern must be at most {MAX_PATTERN_CHARS} characters")
        try:
            compiled = regex.compile(pattern, 0 if case_sensitive else regex.IGNORECASE)
        except regex.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        terms = sorted(set(tokenize(pattern)), key=len, reverse=True)[:8]
        if not terms:
            raise ValueError("pattern must contain at least one letter or number")
        hits = []
        deadline = time.monotonic() + GREP_TOTAL_TIMEOUT
        for chunk_id in self._rank(" OR ".join(f'"{term}"' for term in terms), GREP_CANDIDATES):
            # Match the full text, header included, so company, form, and filing date are searchable.
            text = self._text(chunk_id)
            timeout = min(GREP_TIMEOUT, deadline - time.monotonic())
            try:
                if timeout <= 0:
                    raise TimeoutError
                found = compiled.search(text, timeout=timeout)
            except TimeoutError as exc:
                raise ValueError("regular expression timed out; use a simpler pattern") from exc
            if found:
                title, _ = split_header(text)
                hits.append({"chunk_id": chunk_id, "title": title, "snippet": window(text, found.start())})
                if len(hits) == k:
                    break
        return hits

    def read_document(self, chunk_id: str) -> dict:
        row = self.db.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return {"error": f"unknown chunk_id {chunk_id}"}
        return {"chunk_id": chunk_id, "text": row[0][:READ_MAX_CHARS], "truncated": len(row[0]) > READ_MAX_CHARS}

    def _rank(self, match: str, limit: int) -> list[str]:
        rows = self.db.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?", (match, limit)
        )
        return [row[0] for row in rows]

    def _text(self, chunk_id: str) -> str:
        return self.db.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()[0]


def clamp_k(k: int) -> int:
    # SQLite reads a negative LIMIT as no limit.
    return min(max(k, 1), MAX_K)


def split_header(text: str) -> tuple[str, str]:
    """Split a chunk into a title ("FMC CORP · 2025-03-14 · DEF 14A") built from its header, and its body."""
    header, _, body = text.partition("\n---\n")
    title = " · ".join(line.split(": ", 1)[1] for line in header.splitlines() if ": " in line)
    return title, body


def window(text: str, start: int) -> str:
    """About SNIPPET_CHARS of text around start, whitespace collapsed."""
    begin = max(0, min(start - SNIPPET_CHARS // 3, len(text) - SNIPPET_CHARS))
    end = begin + SNIPPET_CHARS
    snippet = " ".join(text[begin:end].split())
    return f"{'…' if begin else ''}{snippet}{'…' if end < len(text) else ''}"


if __name__ == "__main__":
    build_index(DATA_DIR)
    print(f"wrote {DATA_DIR / 'index.sqlite3'}")
