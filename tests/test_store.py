import numpy as np
import pytest
from minrag.store import VectorStore

DIM = 8


@pytest.fixture
def store(tmp_path):
    return VectorStore(db_path=str(tmp_path / "test.db"))


def _chunks(n=3):
    return [
        {
            "text": f"chunk text number {i}",
            "source": f"doc{i % 2}.pdf",
            "page": i + 1,
            "source_hash": f"hash{i}",
        }
        for i in range(n)
    ]


def _embeddings(n=3):
    rng = np.random.default_rng(42)
    embs = rng.random((n, DIM)).astype(np.float32)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def test_add_and_get_all(store):
    store.add(_chunks(3), _embeddings(3))
    chunks, embs = store.get_all()
    assert len(chunks) == 3
    assert embs.shape == (3, DIM)


def test_get_all_text_no_embedding(store):
    store.add(_chunks(3), _embeddings(3))
    texts = store.get_all_text()
    assert len(texts) == 3
    assert all("embedding" not in t for t in texts)
    assert all({"text", "source", "page"} <= t.keys() for t in texts)


def test_get_all_empty(store):
    chunks, embs = store.get_all()
    assert chunks == []
    assert embs.shape == (0,)


def test_search_returns_top_k(store):
    store.add(_chunks(5), _embeddings(5))
    query = _embeddings(1)[0]
    results = store.search(query, top_k=3)
    assert len(results) == 3
    assert all("score" in r for r in results)


def test_search_empty_store(store):
    query = _embeddings(1)[0]
    results = store.search(query, top_k=5)
    assert results == []


def test_source_filter_get_all(store):
    store.add(_chunks(3), _embeddings(3))
    texts = store.get_all_text(source_filter="doc0.pdf")
    assert all(t["source"] == "doc0.pdf" for t in texts)


def test_source_filter_search(store):
    store.add(_chunks(4), _embeddings(4))
    query = _embeddings(1)[0]
    results = store.search(query, top_k=10, source_filter="doc0.pdf")
    assert all(r["source"] == "doc0.pdf" for r in results)


def test_source_exists_true(store):
    store.add(_chunks(2), _embeddings(2))
    assert store.source_exists("doc0.pdf")


def test_source_exists_false(store):
    assert not store.source_exists("missing.pdf")


def test_get_source_hashes(store):
    store.add(_chunks(3), _embeddings(3))
    hashes = store.get_source_hashes(["doc0.pdf", "doc1.pdf"])
    assert "doc0.pdf" in hashes
    assert "doc1.pdf" in hashes


def test_get_source_hashes_empty(store):
    assert store.get_source_hashes([]) == {}


def test_delete_source(store):
    store.add(_chunks(3), _embeddings(3))
    store.delete_source("doc0.pdf")
    assert not store.source_exists("doc0.pdf")
    assert store.source_exists("doc1.pdf")


def test_get_sources(store):
    store.add(_chunks(3), _embeddings(3))
    sources = store.get_sources()
    assert set(sources) == {"doc0.pdf", "doc1.pdf"}


def test_clear(store):
    store.add(_chunks(3), _embeddings(3))
    store.clear()
    chunks, _ = store.get_all()
    assert chunks == []
