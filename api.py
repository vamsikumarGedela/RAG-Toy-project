import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from minrag import RAG

PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
DB_PATH = os.getenv("DB_PATH", "./minrag.db")

# ---------------------------------------------------------------------------
# App lifespan — initialize RAG once at startup, reuse across all requests
# ---------------------------------------------------------------------------
_rag: RAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag
    provider = os.getenv("LLM_PROVIDER", "ollama")
    api_key  = os.getenv("LLM_API_KEY")
    model    = os.getenv("LLM_MODEL")
    _rag = RAG(llm_provider=provider, api_key=api_key, llm_model=model, db_path=DB_PATH)
    Path(PDF_DIR).mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="minrag API",
    description="Lightweight RAG system — chat with your PDFs",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency
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
    return FileResponse(ui_path)


@app.get("/api", tags=["General"])
def root():
    return {
        "message": "minrag API is running",
        "docs": "/docs",
        "ui": "/",
        "endpoints": ["/ingest", "/chat", "/solve", "/sources", "/health"],
    }


@app.get("/health", tags=["General"])
def health(rag: RAG = Depends(get_rag)):
    sources = rag.sources()
    return {"status": "ok", "sources_loaded": len(sources)}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@app.get("/sources", response_model=SourcesResponse, tags=["Documents"])
def list_sources(rag: RAG = Depends(get_rag)):
    sources = rag.sources()
    return SourcesResponse(sources=sources, count=len(sources))


@app.post("/ingest", tags=["Documents"])
def ingest(
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

    rag.ingest(PDF_DIR)

    return {
        "ingested": saved,
        "message": f"Successfully ingested {len(saved)} PDF(s). Query them at POST /chat.",
    }


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
def chat(req: ChatRequest, rag: RAG = Depends(get_rag)):
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


@app.post("/solve", response_model=SolveResponse, tags=["Chat"])
def solve(req: SolveRequest, rag: RAG = Depends(get_rag)):
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
