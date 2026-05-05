# minrag — Chat with your PDFs. No fluff. Just results.

> Built from scratch. No LangChain. No ChromaDB. No magic wrappers.  
> Just clean Python, deep understanding, and a RAG system that actually works.

---

## The Problem with Every Other RAG Project

You've seen them. 30-line tutorials that call `LangChain.load()` and call it a day.  
They don't teach you anything. They don't give you control. And when something breaks — good luck.

**minrag is different.**

I built every single component by hand — the chunker, the embedder, the vector store, the retriever, the LLM layer, the reranker. Every line of code has a reason. Nothing is hidden behind a framework you can't read.

If you want to understand how RAG *actually* works — this is it.

---

## What minrag Does Differently

### Most RAG projects stop here:
```
Query → Embed → Vector Search → LLM → Answer
```

### minrag goes further:
```
Query → Embed → Vector Search ──┐
                                 ├→ Reciprocal Rank Fusion → Cross-Encoder Rerank → LLM → Answer
Query → BM25 Keyword Search ───┘
```

**Three-stage retrieval.** Dense semantic search *combined* with sparse keyword search, fused together, then re-ranked by a cross-encoder model. This is what production systems use. Not tutorials.

---

## Features That Set It Apart

**Hybrid Retrieval — not just vector search**  
Semantic embeddings catch meaning. BM25 catches exact keywords. Together they catch everything. Reciprocal Rank Fusion merges both rankings intelligently.

**Cross-Encoder Reranking**  
After retrieval, a dedicated reranker model scores every result against your query. The top answers rise to the surface. Less noise. Better answers.

**Hypothesis Mode — beyond Q&A**  
Ask *why* or *how come* and minrag doesn't just retrieve — it thinks. It generates multiple competing hypotheses, retrieves independent evidence for each, scores confidence, and delivers a verdict. Like having a research analyst in your PDF.

```
You: Why is quicksort faster than bubble sort in practice?

minrag:
  ✅ Cache locality advantage          — 78% confidence
  ✅ Fewer total comparisons on avg    — 71% confidence  
  ⚠️  Better pivot selection           — 42% confidence
  ❌ Memory allocation differences     — 12% confidence

  VERDICT: Quicksort's dominance comes primarily from cache-friendly
  access patterns and O(n log n) average complexity...
```

**Smart Ingestion — no duplicate processing**  
Every PDF is fingerprinted with MD5. Re-run ingest as many times as you want — unchanged files are skipped. Only new or modified PDFs are processed. Your time matters.

**Zero Infrastructure**  
SQLite ships with Python. No Docker required. No Pinecone account. No Weaviate setup. No cloud bills.  
`pip install -r requirements.txt` and you're running.

**Works with any LLM**  
Ollama (free, runs locally), OpenAI, OpenRouter, Anthropic — one clean interface, swap providers in your `.env`.

---

## Under the Hood

Every module is small, readable, and purposeful:

```
minrag/
├── chunker.py      — smart PDF text extraction with sentence-aware splitting
├── embedder.py     — lazy-loaded sentence transformers, L2-normalized
├── store.py        — WAL-mode SQLite with 64MB cache, indexed by source
├── retriever.py    — hybrid BM25 + vector fusion with LRU-cached indexes
├── llm.py          — streaming, rate-limit retries, 4 providers
└── hypothesis.py   — parallel evidence retrieval + confidence scoring
```

No file is over 300 lines. Every function does one thing. Read it in an afternoon.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure your LLM (.env file)
LLM_PROVIDER=ollama   # free & local — or openai / openrouter / anthropic
LLM_API_KEY=          # not needed for ollama

# 3. Run
cd ScratchRAG
python run.py
# → opens http://127.0.0.1:8000
```

---

## Web UI

Upload PDFs → ask questions → get answers with source citations and confidence scores.  
Delete individual PDFs without touching the others.  
Switch between PDFs mid-conversation from the navbar.

---

## CLI

```bash
python main.py ingest ./pdfs     # embed your documents
python main.py chat              # start chatting
```

Prefix any question with `solve:` to force hypothesis analysis mode.

---

## API

Full REST API with Swagger docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload & ingest PDFs |
| `GET` | `/sources` | List all documents |
| `DELETE` | `/sources/{name}` | Remove a document |
| `POST` | `/chat` | Ask a question |
| `POST` | `/solve` | Hypothesis analysis |
| `DELETE` | `/history` | Clear chat history |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Keyword search | `rank-bm25` |
| Vector store | SQLite (WAL mode) |
| PDF parsing | `pypdf` |
| Web framework | FastAPI + uvicorn |
| LLM interface | OpenAI SDK + native Anthropic |

---

## Requirements

- Python 3.10+
- For local LLM: [Ollama](https://ollama.com) with any model pulled

---

## Running Tests

```bash
pytest
```

Full test coverage across chunker, store, retriever, hypothesis engine, and all API endpoints.

---

## Docker

```bash
docker-compose up
```

---

*Built with curiosity, not shortcuts.*
