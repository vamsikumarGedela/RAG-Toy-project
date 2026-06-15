import logging
import sqlite3
import numpy as np
from pathlib import Path

from .interfaces import VectorStoreProtocol

logger = logging.getLogger(__name__)


class VectorStore:
    """SQLite-backed vector store. Implements VectorStoreProtocol."""
    def __init__(self, db_path: str = "./minrag.db"):
        self.db_path = db_path
        self._cache: tuple | None = None  # (chunks, embeddings) for full-table queries
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")    # concurrent reads don't block writes
            conn.execute("PRAGMA synchronous=NORMAL")  # 3-5x faster writes, still crash-safe
            conn.execute("PRAGMA cache_size=-64000")   # 64 MB in-process page cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    text        TEXT    NOT NULL,
                    source      TEXT,
                    page        INTEGER,
                    embedding   BLOB    NOT NULL,
                    source_hash TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    role    TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
            # Migrate existing DBs that don't have source_hash yet
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN source_hash TEXT")
            except Exception:
                pass

    def add(self, chunks: list, embeddings: np.ndarray) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (text, source, page, embedding, source_hash) VALUES (?, ?, ?, ?, ?)",
                [
                    (c["text"], c["source"], c["page"], e.astype(np.float32).tobytes(), c.get("source_hash"))
                    for c, e in zip(chunks, embeddings)
                ],
            )
        self._cache = None

    def get_all_text(self, source_filter: str = None) -> list:
        # Reuse the in-memory cache when no filter is applied
        if source_filter is None and self._cache is not None:
            return self._cache[0]
        chunks, _ = self.get_all(source_filter)
        return chunks

    def get_all(self, source_filter: str = None) -> tuple:
        # Return cached result for full-table queries
        if source_filter is None and self._cache is not None:
            return self._cache

        with self._connect() as conn:
            if source_filter:
                rows = conn.execute(
                    "SELECT text, source, page, embedding FROM chunks WHERE source = ?",
                    (source_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT text, source, page, embedding FROM chunks"
                ).fetchall()

        if not rows:
            return [], np.array([])

        chunks = [{"text": r[0], "source": r[1], "page": r[2]} for r in rows]
        embeddings = np.stack([
            np.frombuffer(r[3], dtype=np.float32) for r in rows
        ])
        result = (chunks, embeddings)
        if source_filter is None:
            self._cache = result
        return result

    def search(self, query_embedding: np.ndarray, top_k: int = 10, source_filter: str = None) -> list:
        chunks, embeddings = self.get_all(source_filter)
        if not chunks:
            return []
        scores = np.dot(embeddings, query_embedding.astype(np.float32))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{**chunks[i], "score": float(scores[i])} for i in top_indices]

    def source_exists(self, source: str) -> bool:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE source = ?", (source,)
            ).fetchone()[0]
        return count > 0

    def get_source_hash(self, source: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source_hash FROM chunks WHERE source = ? LIMIT 1", (source,)
            ).fetchone()
        return row[0] if row else None

    def get_source_hashes(self, sources: list) -> dict:
        if not sources:
            return {}
        placeholders = ",".join("?" * len(sources))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT source, source_hash FROM chunks WHERE source IN ({placeholders}) GROUP BY source",
                sources,
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_source(self, source: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._cache = None

    def get_sources(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source FROM chunks ORDER BY source"
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
        self._cache = None

    # ------------------------------------------------------------------
    # Chat history persistence
    # ------------------------------------------------------------------

    def load_history(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM history ORDER BY id"
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def save_history(self, messages: list) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
            conn.executemany(
                "INSERT INTO history (role, content) VALUES (?, ?)",
                [(m["role"], m["content"]) for m in messages],
            )

    def clear_history(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
