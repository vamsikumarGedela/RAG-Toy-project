import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from minrag.retriever import (
    bm25_search,
    reciprocal_rank_fusion,
    _deduplicate,
    rerank,
    clear_cache,
    _bm25_cache,
)

CHUNKS = [
    {"text": "binary trees support fast search and insertion operations", "source": "a.pdf", "page": 1},
    {"text": "hash tables provide O(1) average lookup time", "source": "a.pdf", "page": 2},
    {"text": "bubble sort has quadratic time complexity and is slow", "source": "b.pdf", "page": 1},
    {"text": "linked lists allow dynamic memory allocation efficiently", "source": "b.pdf", "page": 2},
]


@pytest.fixture(autouse=True)
def reset_bm25_cache():
    clear_cache()
    yield
    clear_cache()


# ─── BM25 ────────────────────────────────────────────────────────────────────

def test_bm25_finds_relevant_chunk():
    results = bm25_search("binary tree search", CHUNKS, top_k=2)
    assert results[0]["text"].startswith("binary trees")


def test_bm25_top_k_limit():
    results = bm25_search("binary", CHUNKS, top_k=1)
    assert len(results) <= 1


def test_bm25_empty_chunks():
    assert bm25_search("query", [], top_k=5) == []


def test_bm25_no_matches_returns_empty():
    results = bm25_search("zzzznotaword", CHUNKS, top_k=5)
    assert results == []


def test_bm25_cache_populated():
    bm25_search("binary", CHUNKS, top_k=1, cache_key="mykey")
    assert "mykey" in _bm25_cache


def test_bm25_cache_reused(monkeypatch):
    from minrag import retriever
    build_calls = []
    original_search = bm25_search

    bm25_search("binary", CHUNKS, top_k=1, cache_key="reusekey")
    size_before = len(_bm25_cache)
    bm25_search("trees", CHUNKS, top_k=1, cache_key="reusekey")
    assert len(_bm25_cache) == size_before  # no new entry added


# ─── RRF ─────────────────────────────────────────────────────────────────────

def test_rrf_merges_lists():
    list1 = [{"text": "doc A", "score": 0.9}, {"text": "doc B", "score": 0.5}]
    list2 = [{"text": "doc B", "score": 0.8}, {"text": "doc C", "score": 0.3}]
    fused = reciprocal_rank_fusion([list1, list2])
    texts = [d["text"] for d in fused]
    assert set(texts) == {"doc A", "doc B", "doc C"}


def test_rrf_boosts_doc_in_both_lists():
    list1 = [{"text": "shared doc", "score": 0.5}, {"text": "only in 1", "score": 0.9}]
    list2 = [{"text": "shared doc", "score": 0.5}, {"text": "only in 2", "score": 0.9}]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0]["text"] == "shared doc"


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


# ─── Dedup ───────────────────────────────────────────────────────────────────

def test_deduplicate_removes_near_duplicate():
    docs = [
        {"text": "the quick brown fox jumps over the lazy dog"},
        {"text": "the quick brown fox jumps over the lazy dog today"},
        {"text": "completely unrelated content about databases"},
    ]
    result = _deduplicate(docs, threshold=0.6)
    assert len(result) < 3


def test_deduplicate_keeps_distinct_docs():
    docs = [
        {"text": "binary trees are hierarchical data structures"},
        {"text": "hash tables use key value pairs for fast lookup"},
    ]
    result = _deduplicate(docs, threshold=0.85)
    assert len(result) == 2


def test_deduplicate_empty():
    assert _deduplicate([]) == []


def test_deduplicate_single():
    docs = [{"text": "only one document here"}]
    assert len(_deduplicate(docs)) == 1


# ─── Rerank ──────────────────────────────────────────────────────────────────

def test_rerank_orders_by_score():
    docs = [
        {"text": "bubble sort is slow", "source": "b.pdf", "page": 2},
        {"text": "binary tree search is fast", "source": "a.pdf", "page": 1},
    ]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.1, 0.9]  # second doc scores higher
    with patch("minrag.retriever._get_cross_encoder", return_value=mock_encoder):
        result = rerank("binary search", docs, top_n=2)
    assert result[0]["text"] == "binary tree search is fast"
    assert result[0]["rerank_score"] == pytest.approx(0.9)


def test_rerank_top_n_respected():
    docs = [{"text": f"doc {i}", "source": "x.pdf", "page": i} for i in range(5)]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [float(i) for i in range(5)]
    with patch("minrag.retriever._get_cross_encoder", return_value=mock_encoder):
        result = rerank("query", docs, top_n=3)
    assert len(result) == 3


def test_rerank_empty_docs():
    with patch("minrag.retriever._get_cross_encoder"):
        result = rerank("query", [], top_n=5)
    assert result == []


# ─── clear_cache ─────────────────────────────────────────────────────────────

def test_clear_cache_empties():
    bm25_search("test", CHUNKS, top_k=1, cache_key="toClear")
    assert "toClear" in _bm25_cache
    clear_cache()
    assert "toClear" not in _bm25_cache
