import pytest
from minrag.grounding import (
    rerank_confidence_pct,
    confidence_label,
    passes_gate,
    build_citations,
    NOT_FOUND_MESSAGE,
)


# ─── rerank_confidence_pct ─────────────────────────────────────────────────

def test_confidence_pct_empty_scores():
    assert rerank_confidence_pct([]) == 0.0


def test_confidence_pct_high_scores():
    assert rerank_confidence_pct([5.0, 6.0]) > 90


def test_confidence_pct_low_scores():
    assert rerank_confidence_pct([-5.0, -6.0]) < 10


def test_confidence_pct_neutral_score():
    assert rerank_confidence_pct([0.0]) == pytest.approx(50.0)


# ─── confidence_label ──────────────────────────────────────────────────────

def test_label_high():
    assert "High" in confidence_label(85)


def test_label_medium():
    assert "Medium" in confidence_label(50)


def test_label_low():
    assert "Low" in confidence_label(10)


# ─── passes_gate ────────────────────────────────────────────────────────────

def test_gate_blocks_empty_docs():
    assert passes_gate([]) is False


def test_gate_blocks_low_confidence():
    docs = [{"rerank_score": -6.0}, {"rerank_score": -5.0}]
    assert passes_gate(docs, min_confidence=35.0) is False


def test_gate_allows_high_confidence():
    docs = [{"rerank_score": 4.0}, {"rerank_score": 3.5}]
    assert passes_gate(docs, min_confidence=35.0) is True


def test_gate_respects_custom_threshold():
    docs = [{"rerank_score": 0.0}]
    assert passes_gate(docs, min_confidence=10.0) is True
    assert passes_gate(docs, min_confidence=60.0) is False


# ─── build_citations ────────────────────────────────────────────────────────

def test_build_citations_shape():
    docs = [{"source": "a.pdf", "page": 3, "text": "some passage", "rerank_score": 1.0}]
    citations = build_citations(docs)
    assert citations == [{"source": "a.pdf", "page": 3, "text": "some passage"}]


def test_build_citations_empty():
    assert build_citations([]) == []


# ─── NOT_FOUND_MESSAGE ───────────────────────────────────────────────────────

def test_not_found_message_matches_brief_wording():
    assert "couldn't find this information in the uploaded documents" in NOT_FOUND_MESSAGE
