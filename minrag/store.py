import sqlite3
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self, db_path: str = "./minrag.db"):
        self.db_path = db_path
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

    def get_all_text(self, source_filter: str = None) -> list:
        with self._connect() as conn:
            if source_filter:
                rows = conn.execute(
                    "SELECT text, source, page FROM chunks WHERE source = ?",
                    (source_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT text, source, page FROM chunks"
                ).fetchall()
        return [{"text": r[0], "source": r[1], "page": r[2]} for r in rows]

    def get_all(self, source_filter: str = None) -> tuple:
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
        return chunks, embeddings

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

    def get_sources(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source FROM chunks ORDER BY source"
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
