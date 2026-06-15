import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from minrag import RAG

limiter = Limiter(key_func=get_remote_address)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress /health and /api polling noise from access logs while models load
class _HealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/health" not in msg and "/api" not in msg

logging.getLogger("uvicorn.access").addFilter(_HealthFilter())

PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
DB_PATH = os.getenv("DB_PATH", "./minrag.db")

_ingest_lock = threading.Lock()
_ingest_status: dict = {"state": "idle", "message": "No ingestion has run yet."}

# ---------------------------------------------------------------------------
# App lifespan — initialize RAG once at startup, reuse across all requests
# ---------------------------------------------------------------------------
_rag: RAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag

    # ── RAG setup ──────────────────────────────────────────────────────
    provider = os.getenv("LLM_PROVIDER", "ollama")
    api_key  = os.getenv("LLM_API_KEY")
    model    = os.getenv("LLM_MODEL")
    timeout  = float(os.getenv("LLM_TIMEOUT", "30"))
    _rag = RAG(llm_provider=provider, api_key=api_key, llm_model=model, db_path=DB_PATH, timeout=timeout)
    Path(PDF_DIR).mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="minrag API",
    description="Lightweight RAG system — chat with your PDFs",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_rag() -> RAG:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    return _rag


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    source: str | None = None
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: str

class SolveRequest(BaseModel):
    problem: str
    source: str | None = None

class SolveResponse(BaseModel):
    analysis: str

class SourcesResponse(BaseModel):
    sources: list[str]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def serve_ui():
    ui_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(ui_path.read_text(encoding="utf-8"))


@app.get("/api", tags=["General"])
def root():
    return {
        "message": "minrag API is running",
        "docs": "/docs",
        "ui": "/",
        "endpoints": ["/ingest", "/ingest/status", "/chat", "/chat/stream", "/solve", "/sources", "/health"],
    }


@app.get("/health", tags=["General"])
def health(rag: RAG = Depends(get_rag)):
    if not rag._ready:
        raise HTTPException(status_code=503, detail="Models are loading, please wait...")
    sources = rag.sources()
    return {"status": "ok", "sources_loaded": len(sources)}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@app.get("/sources", response_model=SourcesResponse, tags=["Documents"])
def list_sources(rag: RAG = Depends(get_rag)):
    sources = rag.sources()
    return SourcesResponse(sources=sources, count=len(sources))


def _run_ingest(rag: RAG, pdf_dir: str) -> None:
    global _ingest_status
    with _ingest_lock:
        _ingest_status = {"state": "running", "message": "Ingestion in progress..."}
        try:
            rag.ingest(pdf_dir)
            _ingest_status = {"state": "done", "message": "Ingestion completed successfully."}
            logger.info("Background ingest finished.")
        except Exception as e:
            _ingest_status = {"state": "failed", "message": str(e)}
            logger.error("Background ingest failed: %s", e)


@app.post("/ingest", tags=["Documents"])
@limiter.limit("5/minute")
def ingest(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more PDF files to ingest"),
    rag: RAG = Depends(get_rag),
):
    pdf_dir = Path(PDF_DIR)
    saved = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF. Only .pdf files are accepted.",
            )
        dest = pdf_dir / file.filename
        dest.write_bytes(file.file.read())
        saved.append(file.filename)

    background_tasks.add_task(_run_ingest, rag, PDF_DIR)

    return {
        "ingested": saved,
        "message": f"Saved {len(saved)} PDF(s). Ingestion running in background — check GET /ingest/status.",
    }


@app.get("/ingest/status", tags=["Documents"])
def ingest_status():
    return _ingest_status


@app.delete("/sources/{name}", tags=["Documents"])
def delete_source(name: str, rag: RAG = Depends(get_rag)):
    if name not in rag.sources():
        raise HTTPException(status_code=404, detail=f"Source '{name}' not found.")

    rag.delete_source(name)

    pdf_file = Path(PDF_DIR) / name
    if pdf_file.exists():
        pdf_file.unlink()

    return {"deleted": name, "message": f"'{name}' removed from the system."}


# ---------------------------------------------------------------------------
# Chat & Solve
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("30/minute")
def chat(request: Request, req: ChatRequest, rag: RAG = Depends(get_rag)):
    if req.source and req.source not in rag.sources():
        raise HTTPException(
            status_code=404,
            detail=f"Source '{req.source}' not found. Check GET /sources.",
        )

    result = rag.ask_raw(req.question, source_filter=req.source, top_k=req.top_k)

    if not result["found"]:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found for this question in the documents.",
        )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )


@app.post("/chat/stream", tags=["Chat"])
@limiter.limit("30/minute")
def chat_stream(request: Request, req: ChatRequest, rag: RAG = Depends(get_rag)):
    if req.source and req.source not in rag.sources():
        raise HTTPException(
            status_code=404,
            detail=f"Source '{req.source}' not found. Check GET /sources.",
        )

    def generate():
        try:
            for token in rag.ask_stream(req.question, source_filter=req.source, top_k=req.top_k):
                if token == "[NO_RESULTS]":
                    yield f"event: error\ndata: {json.dumps('No relevant content found in the documents.')}\n\n"
                    return
                elif token.startswith("[META]"):
                    yield f"event: meta\ndata: {token[6:]}\n\n"
                else:
                    yield f"data: {json.dumps(token)}\n\n"
            yield "event: done\ndata: \n\n"
        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps(f'Server error: {str(e)}')}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/solve", response_model=SolveResponse, tags=["Chat"])
@limiter.limit("10/minute")
def solve(request: Request, req: SolveRequest, rag: RAG = Depends(get_rag)):
    if req.source and req.source not in rag.sources():
        raise HTTPException(
            status_code=404,
            detail=f"Source '{req.source}' not found. Check GET /sources.",
        )

    analysis = rag.solve(req.problem, source_filter=req.source)
    return SolveResponse(analysis=analysis)


@app.delete("/history", tags=["Chat"])
def clear_history(rag: RAG = Depends(get_rag)):
    rag.clear_history()
    rag.clear_solve_history()
    return {"message": "Conversation history cleared."}
