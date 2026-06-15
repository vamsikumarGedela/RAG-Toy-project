import logging
from collections import OrderedDict
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_model_name: str = None
_bm25_cache: OrderedDict = OrderedDict()   # LRU: source_filter → (chunks, BM25Okapi)
_query_cache: OrderedDict = OrderedDict()  # LRU: (query, source_filter, top_k) → ranked docs
_BM25_CACHE_MAX = 10
_QUERY_CACHE_MAX = 128
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


def _get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL):
    global _cross_encoder, _cross_encoder_model_name
    if _cross_encoder is None or _cross_encoder_model_name != model_name:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_model_name = model_name
    return _cross_encoder


def warmup_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> None:
    _get_cross_encoder(model_name)
    logger.info("Cross-encoder warmed up: %s", model_name)


def clear_cache() -> None:
    global _bm25_cache, _query_cache
    _bm25_cache.clear()
    _query_cache.clear()
    logger.info("Retrieval caches cleared.")


def vector_search(query_embedding: np.ndarray, store, top_k: int = 10, source_filter: str = None) -> list:
    return store.search(query_embedding, top_k=top_k, source_filter=source_filter)


def bm25_search(query: str, chunks: list, top_k: int = 10, cache_key: str = "all") -> list:
    global _bm25_cache
    if not chunks:
        return []

    if cache_key in _bm25_cache:
        _bm25_cache.move_to_end(cache_key)
    else:
        tokenized = [c["text"].lower().split() for c in chunks]
        if len(_bm25_cache) >= _BM25_CACHE_MAX:
            _bm25_cache.popitem(last=False)  # evict least-recently-used
        _bm25_cache[cache_key] = (chunks, BM25Okapi(tokenized))

    cached_chunks, bm25 = _bm25_cache[cache_key]
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**cached_chunks[i], "score": float(scores[i])}
        for i in top_indices
        if scores[i] > 0
    ]


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    scores = {}
    doc_map = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc["text"][:120]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def _deduplicate(docs: list, threshold: float = 0.85) -> list:
    """Remove near-duplicate chunks using Jaccard similarity on word sets."""
    kept = []
    kept_word_sets = []
    for doc in docs:
        words = set(doc["text"].lower().split())
        is_dup = any(
            len(words & kws) / len(words | kws) >= threshold
            for kws in kept_word_sets
            if words and kws
        )
        if not is_dup:
            kept.append(doc)
            kept_word_sets.append(words)
    return kept


def rerank(query: str, docs: list, top_n: int = 5, model: str = DEFAULT_RERANK_MODEL) -> list:
    if not docs:
        return []
    encoder = _get_cross_encoder(model)
    pairs = [[query, doc["text"]] for doc in docs]
    scores = encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [{**doc, "rerank_score": float(score)} for score, doc in scored[:top_n]]


def retrieve(query: str, store, embedder, top_k: int = 5, source_filter: str = None, rerank_model: str = DEFAULT_RERANK_MODEL) -> list:
    global _query_cache

    cache_key = (query, source_filter, top_k, rerank_model)
    if cache_key in _query_cache:
        _query_cache.move_to_end(cache_key)
        logger.debug("Query cache hit: %s", query[:60])
        return _query_cache[cache_key]

    query_emb = embedder.encode_one(query)

    # Single store read — cache hit after first query
    all_chunks, all_embeddings = store.get_all(source_filter)
    if not all_chunks:
        return []

    # Vector search using already-loaded embeddings (no second DB read)
    scores = np.dot(all_embeddings, query_emb.astype(np.float32))
    top_indices = np.argsort(scores)[::-1][:top_k * 2]
    vector_results = [{**all_chunks[i], "score": float(scores[i])} for i in top_indices]

    bm25_key = source_filter or "all"
    bm25_results = bm25_search(query, all_chunks, top_k=top_k * 2, cache_key=bm25_key)

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    deduped = _deduplicate(fused)[:top_k * 2]
    result = rerank(query, deduped, top_n=top_k, model=rerank_model)

    if len(_query_cache) >= _QUERY_CACHE_MAX:
        _query_cache.popitem(last=False)
    _query_cache[cache_key] = result
    logger.debug("Query cached: %s", query[:60])

    return result
