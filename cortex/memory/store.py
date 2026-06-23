"""Memory: short-term conversation buffer + long-term SQLite store.

* :class:`ShortTermMemory` is a bounded conversation buffer (the working set
  passed to the model each turn).
* :class:`LongTermMemory` persists notes to SQLite and supports two recall
  modes:
    - keyword recall (always available, embedding-free), and
    - optional vector recall using ``sentence-transformers``
      (``all-MiniLM-L6-v2``), guarded so the store works without it.

:class:`Memory` composes both for convenience.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class MemoryRecord:
    """A single recalled long-term memory entry."""

    id: int
    content: str
    kind: str
    score: float
    created_at: float


@dataclass
class ShortTermMemory:
    """A bounded in-memory conversation buffer.

    Keeps at most ``max_items`` recent ``(role, content)`` turns. Older turns
    are dropped (the long-term store is where durable facts belong).
    """

    max_items: int = 50
    _buffer: List[Tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """Append a turn and trim to the configured window."""
        self._buffer.append((role, content))
        if len(self._buffer) > self.max_items:
            self._buffer = self._buffer[-self.max_items :]

    def recent(self, n: Optional[int] = None) -> List[Tuple[str, str]]:
        """Return the most recent ``n`` turns (all turns if ``n`` is None)."""
        if n is None:
            return list(self._buffer)
        return self._buffer[-n:]

    def clear(self) -> None:
        """Forget the buffered conversation."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class LongTermMemory:
    """SQLite-backed long-term memory with keyword and optional vector recall."""

    def __init__(
        self,
        db_path: str = ".cortex/memory.sqlite",
        use_vectors: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the same connection can serve the API.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        self._model = None
        self.use_vectors = False
        if use_vectors:
            self._try_load_model(model_name)

    # -- schema ---------------------------------------------------------- #
    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'note',
                embedding TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def _try_load_model(self, model_name: str) -> None:
        """Attempt to load sentence-transformers; fall back silently if absent."""
        try:  # pragma: no cover - heavy optional dep not installed in CI
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.use_vectors = True
        except Exception:
            self._model = None
            self.use_vectors = False

    # -- writes ---------------------------------------------------------- #
    def add(self, content: str, kind: str = "note") -> int:
        """Persist a memory and return its row id."""
        embedding = None
        if self.use_vectors and self._model is not None:  # pragma: no cover
            vec = self._model.encode(content).tolist()
            embedding = ",".join(f"{x:.6f}" for x in vec)
        cur = self._conn.execute(
            "INSERT INTO memories (content, kind, embedding, created_at) VALUES (?, ?, ?, ?)",
            (content, kind, embedding, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    # -- recall ---------------------------------------------------------- #
    def recall(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        """Recall the most relevant memories for ``query``.

        Uses vector similarity when enabled and available; otherwise keyword
        overlap scoring.
        """
        if self.use_vectors and self._model is not None:  # pragma: no cover
            return self._recall_vector(query, top_k)
        return self._recall_keyword(query, top_k)

    def _recall_keyword(self, query: str, top_k: int) -> List[MemoryRecord]:
        terms = set(_tokenize(query))
        if not terms:
            return []
        rows = self._conn.execute(
            "SELECT id, content, kind, created_at FROM memories"
        ).fetchall()
        scored: List[MemoryRecord] = []
        for row in rows:
            doc_terms = _tokenize(row["content"])
            if not doc_terms:
                continue
            overlap = sum(1 for t in doc_terms if t in terms)
            if overlap == 0:
                continue
            # Normalize by document length to avoid favoring long entries.
            score = overlap / math.sqrt(len(doc_terms))
            scored.append(
                MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    kind=row["kind"],
                    score=score,
                    created_at=row["created_at"],
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _recall_vector(self, query: str, top_k: int) -> List[MemoryRecord]:  # pragma: no cover
        qvec = self._model.encode(query).tolist()
        rows = self._conn.execute(
            "SELECT id, content, kind, embedding, created_at FROM memories"
        ).fetchall()
        scored: List[MemoryRecord] = []
        for row in rows:
            if not row["embedding"]:
                continue
            vec = [float(x) for x in row["embedding"].split(",")]
            score = _cosine(qvec, vec)
            scored.append(
                MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    kind=row["kind"],
                    score=score,
                    created_at=row["created_at"],
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def all(self) -> List[MemoryRecord]:
        """Return every stored memory (newest first)."""
        rows = self._conn.execute(
            "SELECT id, content, kind, created_at FROM memories ORDER BY id DESC"
        ).fetchall()
        return [
            MemoryRecord(
                id=r["id"], content=r["content"], kind=r["kind"], score=0.0, created_at=r["created_at"]
            )
            for r in rows
        ]

    def count(self) -> int:
        """Number of stored memories."""
        return int(self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"])

    def clear(self) -> None:
        """Delete all stored memories."""
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Memory:
    """Convenience wrapper composing short-term and long-term memory."""

    short_term: ShortTermMemory
    long_term: LongTermMemory

    @classmethod
    def create(
        cls,
        db_path: str = ".cortex/memory.sqlite",
        use_vectors: bool = False,
        buffer_size: int = 50,
    ) -> "Memory":
        """Construct a composed memory with sensible defaults."""
        return cls(
            short_term=ShortTermMemory(max_items=buffer_size),
            long_term=LongTermMemory(db_path=db_path, use_vectors=use_vectors),
        )

    def remember(self, content: str, kind: str = "note") -> int:
        """Persist a durable memory in the long-term store."""
        return self.long_term.add(content, kind=kind)

    def recall(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        """Recall relevant long-term memories for a query."""
        return self.long_term.recall(query, top_k=top_k)

    def observe(self, role: str, content: str) -> None:
        """Add a conversation turn to short-term memory."""
        self.short_term.add(role, content)
