import math

# Shown to the user whenever retrieval confidence is too low to trust an
# LLM-generated answer, or when nothing relevant was retrieved at all.
# Kept as one canonical string so the code-level refusal (this gate) and the
# LLM's own prompted refusal (llm.build_ask_messages) read identically.
NOT_FOUND_MESSAGE = (
    "I couldn't find this information in the uploaded documents. "
    "The uploaded PDFs do not contain enough information to answer this question."
)

# Below this confidence percentage, we refuse rather than risk an answer that
# isn't actually grounded in the retrieved chunks. Tuned against the existing
# rerank-score-to-confidence curve (minrag.grounding.rerank_confidence_pct).
DEFAULT_MIN_CONFIDENCE = 35.0


def rerank_confidence_pct(scores: list) -> float:
    """Map cross-encoder rerank scores to a 0-100 confidence percentage."""
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    return 100 / (1 + math.exp(-avg / 2))


def confidence_label(pct: float) -> str:
    if pct >= 70:
        return f"{round(pct)}% (High)"
    elif pct >= 40:
        return f"{round(pct)}% (Medium)"
    return f"{round(pct)}% (Low)"


def passes_gate(docs: list, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> bool:
    """
    True if retrieved docs are confident enough to hand to the LLM at all.
    False means: don't call the LLM — return NOT_FOUND_MESSAGE directly.
    """
    if not docs:
        return False
    scores = [d.get("rerank_score", 0) for d in docs]
    return rerank_confidence_pct(scores) >= min_confidence


def build_citations(docs: list) -> list:
    """Chunk-level evidence for a given answer — source, page, and the exact
    supporting text, so a UI can show what the model actually saw."""
    return [
        {"source": d["source"], "page": d["page"], "text": d["text"]}
        for d in docs
    ]
