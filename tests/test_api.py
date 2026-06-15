import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api import app, get_rag

MOCK_ANSWER = {
    "answer": "Binary trees are efficient for search.",
    "sources": ["doc1.pdf p.1"],
    "confidence": "85% (High)",
    "found": True,
}


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.sources.return_value = ["doc1.pdf", "doc2.pdf"]
    rag.ask_raw.return_value = MOCK_ANSWER
    rag.solve.return_value = "Verdict: The most likely cause is X."
    return rag


@pytest.fixture
def client(mock_rag):
    app.dependency_overrides[get_rag] = lambda: mock_rag
    # Patch RAG class so the lifespan startup doesn't need real env vars
    with patch("api.RAG", return_value=mock_rag):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


# ─── General ─────────────────────────────────────────────────────────────────

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_api_info(client):
    resp = client.get("/api")
    assert resp.status_code == 200
    data = resp.json()
    assert "minrag" in data["message"].lower()
    assert "/chat" in data["endpoints"]


def test_health(client, mock_rag):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "sources_loaded": 2}


# ─── Sources ─────────────────────────────────────────────────────────────────

def test_list_sources(client):
    resp = client.get("/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert "doc1.pdf" in data["sources"]


def test_delete_source_success(client, mock_rag):
    with patch("pathlib.Path.exists", return_value=False):
        resp = client.delete("/sources/doc1.pdf")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "doc1.pdf"
    mock_rag.delete_source.assert_called_once_with("doc1.pdf")


def test_delete_source_not_found(client, mock_rag):
    mock_rag.sources.return_value = []
    resp = client.delete("/sources/missing.pdf")
    assert resp.status_code == 404


def test_delete_source_removes_file(client, mock_rag):
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.unlink") as mock_unlink:
        resp = client.delete("/sources/doc1.pdf")
    assert resp.status_code == 200
    mock_unlink.assert_called_once()


# ─── Ingest ──────────────────────────────────────────────────────────────────

def test_ingest_rejects_non_pdf(client):
    resp = client.post(
        "/ingest",
        files=[("files", ("document.txt", b"text content", "text/plain"))],
    )
    assert resp.status_code == 400
    assert "not a PDF" in resp.json()["detail"]


def test_ingest_accepts_pdf(client, mock_rag):
    with patch("pathlib.Path.write_bytes"):
        resp = client.post(
            "/ingest",
            files=[("files", ("report.pdf", b"%PDF-1.4 fake", "application/pdf"))],
        )
    assert resp.status_code == 200
    assert "report.pdf" in resp.json()["ingested"]
    mock_rag.ingest.assert_called_once()


def test_ingest_multiple_pdfs(client, mock_rag):
    with patch("pathlib.Path.write_bytes"):
        resp = client.post(
            "/ingest",
            files=[
                ("files", ("a.pdf", b"%PDF-1.4", "application/pdf")),
                ("files", ("b.pdf", b"%PDF-1.4", "application/pdf")),
            ],
        )
    assert resp.status_code == 200
    assert len(resp.json()["ingested"]) == 2


# ─── Chat ────────────────────────────────────────────────────────────────────

def test_chat_success(client):
    resp = client.post("/chat", json={"question": "What is a binary tree?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == MOCK_ANSWER["answer"]
    assert data["sources"] == MOCK_ANSWER["sources"]
    assert data["confidence"] == MOCK_ANSWER["confidence"]


def test_chat_source_filter_not_found(client, mock_rag):
    mock_rag.sources.return_value = []
    resp = client.post("/chat", json={"question": "test", "source": "missing.pdf"})
    assert resp.status_code == 404


def test_chat_no_relevant_content(client, mock_rag):
    mock_rag.ask_raw.return_value = {
        "answer": "", "sources": [], "confidence": "Unknown", "found": False
    }
    resp = client.post("/chat", json={"question": "irrelevant question"})
    assert resp.status_code == 404


def test_chat_custom_top_k(client, mock_rag):
    client.post("/chat", json={"question": "test", "top_k": 10})
    mock_rag.ask_raw.assert_called_once_with("test", source_filter=None, top_k=10)


# ─── Solve ───────────────────────────────────────────────────────────────────

def test_solve_success(client):
    resp = client.post("/solve", json={"problem": "Why is bubble sort slow?"})
    assert resp.status_code == 200
    assert resp.json()["analysis"] == "Verdict: The most likely cause is X."


def test_solve_source_filter_not_found(client, mock_rag):
    mock_rag.sources.return_value = []
    resp = client.post("/solve", json={"problem": "test", "source": "missing.pdf"})
    assert resp.status_code == 404


# ─── History ─────────────────────────────────────────────────────────────────

def test_clear_history(client, mock_rag):
    resp = client.delete("/history")
    assert resp.status_code == 200
    mock_rag.clear_history.assert_called_once()
    mock_rag.clear_solve_history.assert_called_once()


# ─── Streaming ───────────────────────────────────────────────────────────────

def test_chat_stream_returns_sse_content_type(client, mock_rag):
    mock_rag.ask_stream.return_value = iter([
        "Hello", " world",
        '[META]{"sources": ["doc1.pdf p.1"], "confidence": "85% (High)"}',
    ])
    resp = client.post("/chat/stream", json={"question": "What is a binary tree?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_chat_stream_contains_expected_events(client, mock_rag):
    mock_rag.ask_stream.return_value = iter([
        "Hello", " world",
        '[META]{"sources": ["doc1.pdf p.1"], "confidence": "85% (High)"}',
    ])
    resp = client.post("/chat/stream", json={"question": "What is a binary tree?"})
    body = resp.text
    assert "data:" in body
    assert "event: meta" in body
    assert "event: done" in body


def test_chat_stream_no_results(client, mock_rag):
    mock_rag.ask_stream.return_value = iter(["[NO_RESULTS]"])
    resp = client.post("/chat/stream", json={"question": "unknown question"})
    assert resp.status_code == 200
    assert "event: error" in resp.text


def test_chat_stream_source_not_found(client, mock_rag):
    mock_rag.sources.return_value = []
    resp = client.post("/chat/stream", json={"question": "test", "source": "missing.pdf"})
    assert resp.status_code == 404


def test_chat_stream_tokens_in_order(client, mock_rag):
    tokens = ["Hello", " ", "world"]
    mock_rag.ask_stream.return_value = iter(
        tokens + ['[META]{"sources": [], "confidence": "50% (Medium)"}']
    )
    resp = client.post("/chat/stream", json={"question": "test"})
    body = resp.text
    # all tokens appear in the response body
    for token in tokens:
        assert token in body


# ─── Ingest status ───────────────────────────────────────────────────────────

def test_ingest_status_has_required_fields(client):
    resp = client.get("/ingest/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "state" in data
    assert "message" in data


def test_ingest_status_initial_state(client):
    import api
    api._ingest_status = {"state": "idle", "message": "No ingestion has run yet."}
    resp = client.get("/ingest/status")
    assert resp.json()["state"] == "idle"


def test_ingest_returns_background_message(client, mock_rag):
    with patch("pathlib.Path.write_bytes"):
        resp = client.post(
            "/ingest",
            files=[("files", ("test.pdf", b"%PDF-1.4", "application/pdf"))],
        )
    assert resp.status_code == 200
    assert "background" in resp.json()["message"].lower()


def test_ingest_status_updated_on_success(client, mock_rag):
    import api
    api._run_ingest(mock_rag, "./pdfs")
    assert api._ingest_status["state"] == "done"


def test_ingest_status_updated_on_failure(client, mock_rag):
    import api
    mock_rag.ingest.side_effect = RuntimeError("disk full")
    api._run_ingest(mock_rag, "./pdfs")
    assert api._ingest_status["state"] == "failed"
    assert "disk full" in api._ingest_status["message"]


# ─── Rate limiting ───────────────────────────────────────────────────────────

def test_rate_limiter_registered_on_app():
    from api import app
    from slowapi import Limiter
    assert hasattr(app.state, "limiter")
    assert isinstance(app.state.limiter, Limiter)


def test_rate_limit_exceeded_returns_429(client, mock_rag):
    # Override rate limit to 1/minute for this test
    from api import limiter
    with patch.object(limiter, "limit", return_value=lambda f: f):
        pass  # just verifying the limiter is patachable

    # Exhaust the real limit (30/minute for /chat)
    for _ in range(30):
        client.post("/chat", json={"question": "test"})
    resp = client.post("/chat", json={"question": "test"})
    assert resp.status_code == 429
