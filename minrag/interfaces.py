from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Interface every vector store backend must satisfy.
    Swap SQLite for FAISS, Pinecone, etc. without touching retrieval logic.
    """

    def add(self, chunks: list, embeddings: np.ndarray) -> None: ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        source_filter: str | None = None,
    ) -> list: ...

    def get_all(self, source_filter: str | None = None) -> tuple: ...

    def get_all_text(self, source_filter: str | None = None) -> list: ...

    def get_sources(self) -> list: ...

    def get_source_hashes(self, sources: list) -> dict: ...

    def delete_source(self, source: str) -> None: ...

    def clear(self) -> None: ...
