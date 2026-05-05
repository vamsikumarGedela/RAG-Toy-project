import pytest
from unittest.mock import MagicMock
from minrag.hypothesis import (
    _choose_n_hypotheses,
    _parse_json_array,
    _confidence,
    generate_hypotheses,
    classify_evidence,
)


# ─── _choose_n_hypotheses ─────────────────────────────────────────────────────

def test_choose_n_short_simple():
    assert _choose_n_hypotheses("What is a tree?") == 2


def test_choose_n_complex_keyword():
    n = _choose_n_hypotheses("Why does bubble sort perform poorly on large datasets?")
    assert n >= 6


def test_choose_n_long_question():
    long = "Describe the detailed algorithmic steps involved in each operation " * 3
    n = _choose_n_hypotheses(long)
    assert n >= 6


def test_choose_n_normal():
    # No complex keywords, 8-25 words → N_HYPOTHESES (4)
    n = _choose_n_hypotheses("What are the main properties of a balanced binary tree?")
    assert n == 4


# ─── _parse_json_array ────────────────────────────────────────────────────────

def test_parse_valid_json_array():
    result = _parse_json_array('Some preamble ["H1", "H2", "H3"] trailing text')
    assert result == ["H1", "H2", "H3"]


def test_parse_no_json():
    assert _parse_json_array("No JSON at all here") == []


def test_parse_malformed_json():
    assert _parse_json_array("[broken json, no quotes]") == []


def test_parse_nested_content():
    raw = 'Reasoning...\n["cause A", "cause B"]'
    result = _parse_json_array(raw)
    assert result == ["cause A", "cause B"]


# ─── _confidence ──────────────────────────────────────────────────────────────

def test_confidence_all_support():
    assert _confidence(5, 0, 5) == 100.0


def test_confidence_all_contradict():
    assert _confidence(0, 5, 5) == 0.0


def test_confidence_mixed():
    c = _confidence(3, 1, 4)
    assert 0.0 < c < 100.0


def test_confidence_zero_total():
    assert _confidence(0, 0, 0) == 0.0


def test_confidence_clamped_at_zero():
    assert _confidence(0, 10, 10) == 0.0


# ─── generate_hypotheses ─────────────────────────────────────────────────────

def test_generate_returns_list():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '["H1: memory issue", "H2: algorithm bug", "H3: config error"]'
    result = generate_hypotheses("Why is the system slow?", mock_llm, n=3)
    assert isinstance(result, list)
    assert len(result) == 3


def test_generate_fallback_line_parsing():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "1. Memory leak\n2. CPU bottleneck\n3. Slow I/O"
    result = generate_hypotheses("Why is it slow?", mock_llm, n=3)
    assert isinstance(result, list)
    assert 1 <= len(result) <= 3


def test_generate_respects_n_limit():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '["H1", "H2", "H3", "H4", "H5"]'
    result = generate_hypotheses("question", mock_llm, n=3)
    assert len(result) == 3


def test_generate_uses_history():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '["hypothesis"]'
    history = [{"role": "user", "content": "prior context"}]
    generate_hypotheses("question", mock_llm, history=history, n=1)
    call_args = mock_llm.chat.call_args[0][0]  # first positional arg (messages)
    contents = [m["content"] for m in call_args]
    assert any("prior context" in c for c in contents)


# ─── classify_evidence ────────────────────────────────────────────────────────

def test_classify_returns_labels():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '["SUPPORTS", "IRRELEVANT", "CONTRADICTS"]'
    chunks = [{"text": f"chunk {i}"} for i in range(3)]
    labels = classify_evidence("Binary trees are efficient", chunks, mock_llm)
    assert labels == ["SUPPORTS", "IRRELEVANT", "CONTRADICTS"]


def test_classify_pads_short_response():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '["SUPPORTS"]'
    chunks = [{"text": f"chunk {i}"} for i in range(3)]
    labels = classify_evidence("hypothesis", chunks, mock_llm)
    assert len(labels) == 3
    assert labels[1] == "IRRELEVANT"
    assert labels[2] == "IRRELEVANT"


def test_classify_normalizes_unknown_label():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '["SUPPORTS", "UNKNOWN_LABEL"]'
    chunks = [{"text": "chunk A"}, {"text": "chunk B"}]
    labels = classify_evidence("hypothesis", chunks, mock_llm)
    assert labels[1] == "IRRELEVANT"


def test_classify_empty_chunks():
    mock_llm = MagicMock()
    result = classify_evidence("hypothesis", [], mock_llm)
    assert result == []
    mock_llm.chat.assert_not_called()
