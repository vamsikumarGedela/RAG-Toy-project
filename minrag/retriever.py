import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

_cross_encoder = None
_cross_encoder_model_name: str = None
_bm25_cache: dict = {}   # key: source_filter → (chunks, BM25Okapi)
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL) -> CrossEncoder:
    global _cross_encoder, _cross_encoder_model_name
    if _cross_encoder is None or _cross_encoder_model_name != model_name:
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_model_name = model_name
    return _cross_encoder


def clear_cache() -> None:
    global _bm25_cache
    _bm25_cache.clear()


def vector_search(query_embedding: np.ndarray, store, top_k: int = 10, source_filter: str = None) -> list:
    return store.search(query_embedding, top_k=top_k, source_filter=source_filter)


def bm25_search(query: str, chunks: list, top_k: int = 10, cache_key: str = "all") -> list:
    global _bm25_cache
    if not chunks:
        return []

    if cache_key not in _bm25_cache:
        tokenized = [c["text"].lower().split() for c in chunks]
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


def rerank(query: str, docs: list, top_n: int = 5, model: str = DEFAULT_RERANK_MODEL) -> list:
    if not docs:
        return []
    encoder = _get_cross_encoder(model)
    pairs = [[query, doc["text"]] for doc in docs]
    scores = encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [{**doc, "rerank_score": float(score)} for score, doc in scored[:top_n]]


def retrieve(query: str, store, embedder, top_k: int = 5, source_filter: str = None, rerank_model: str = DEFAULT_RERANK_MODEL) -> list:
    query_emb = embedder.encode_one(query)

    vector_results = vector_search(query_emb, store, top_k=top_k * 2, source_filter=source_filter)

    cache_key = source_filter or "all"
    all_chunks, _ = store.get_all(source_filter)
    bm25_results = bm25_search(query, all_chunks, top_k=top_k * 2, cache_key=cache_key)

    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return rerank(query, fused, top_n=top_k, model=rerank_model)
