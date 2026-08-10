import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from dotenv import load_dotenv

from .chunker import load_pdfs, chunk_pages, extract_tables
from .embedder import Embedder
from .interfaces import VectorStoreProtocol
from .store import VectorStore
from .retriever import retrieve, clear_cache, warmup_reranker
from .llm import LLM, build_ask_messages
from . import hypothesis as hyp
from . import grounding

load_dotenv()

logger = logging.getLogger(__name__)

_MAX_HISTORY_CHARS = 12_000  # ~3k tokens; trim oldest messages when exceeded


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _trim_history(history: list) -> list:
    total = sum(len(m["content"]) for m in history)
    while len(history) > 2 and total > _MAX_HISTORY_CHARS:
        removed = history.pop(0)
        total -= len(removed["content"])
    return history

__version__ = "0.1.0"
__all__ = ["RAG"]


class RAG:
    """
    minrag — lightweight RAG library built from scratch.
    No LangChain. No ChromaDB. Full control.

    Usage:
        rag = RAG()                                      # free, uses Ollama
        rag = RAG(llm_provider="openai",     api_key="sk-...")
        rag = RAG(llm_provider="anthropic",  api_key="sk-ant-...")
        rag = RAG(llm_provider="openrouter", api_key="sk-or-...")

        rag.ingest("./pdfs")
        rag.ask("What is a binary tree?")
        rag.solve("Why is bubble sort slow?")
    """

    def __init__(
        self,
        db_path: str = "./minrag.db",
        embed_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        llm_provider: str = "ollama",
        llm_model: str = None,
        api_key: str = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        rerank_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        min_confidence: float = grounding.DEFAULT_MIN_CONFIDENCE,
    ):
        self.embedder = Embedder(embed_model)
        self.store = VectorStore(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model
        self.min_confidence = min_confidence
        self.llm = LLM(provider=llm_provider, model=llm_model, api_key=api_key, temperature=temperature, timeout=timeout)
        self._db_path = Path(db_path)
        # Chat history is per-conversation (loaded/saved by conversation_id on
        # every call) rather than instance state — a single RAG instance is
        # shared across every API request, so caching one history in memory
        # here would leak between concurrent conversations.
        self._solve_history: list = self._load_solve_history()

        logger.info("minrag v%s | provider: %s | model: %s", __version__, llm_provider, self.llm.model)
        self._ready = False
        threading.Thread(target=self._warmup_models, daemon=True).start()

    def _warmup_models(self) -> None:
        t1 = threading.Thread(target=self.embedder.warmup, daemon=True)
        t2 = threading.Thread(target=lambda: warmup_reranker(self.rerank_model), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self._ready = True
        logger.info("Models ready — RAG fully warmed up.")

    def _solve_history_path(self) -> Path:
        return self._db_path.with_suffix(".solve_history.json")

    def _load_solve_history(self) -> list:
        p = self._solve_history_path()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _save_solve_history(self) -> None:
        self._solve_history_path().write_text(
            json.dumps(self._solve_history, indent=2), encoding="utf-8"
        )

    def ingest(self, pdf_dir: str = "./pdfs", force: bool = False) -> None:
        logger.info("Ingesting PDFs from '%s'", pdf_dir)

        all_paths = list(Path(pdf_dir).glob("**/*.pdf"))
        if not all_paths:
            logger.warning("No PDFs found in '%s'", pdf_dir)
            return

        if force:
            self.store.clear()
            clear_cache()
            new_paths = all_paths
            file_hashes = {p: _file_hash(p) for p in new_paths}
        else:
            new_paths = []
            file_hashes = {}
            stored_hashes = self.store.get_source_hashes([p.name for p in all_paths])
            for p in all_paths:
                stored_hash = stored_hashes.get(p.name)
                fh = _file_hash(p)
                if stored_hash is None:
                    new_paths.append(p)
                    file_hashes[p] = fh
                elif stored_hash != fh:
                    logger.info("Detected change in %s — re-ingesting.", p.name)
                    self.store.delete_source(p.name)
                    new_paths.append(p)
                    file_hashes[p] = fh
            skipped = len(all_paths) - len(new_paths)
            if skipped:
                logger.info("Skipping %d already-ingested PDF(s).", skipped)
            if not new_paths:
                logger.info("All PDFs already ingested. Nothing to do.")
                return

        names_filter = {p.name for p in new_paths}
        pages = load_pdfs(pdf_dir, names_filter=names_filter)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)

        table_chunks = extract_tables(pdf_dir, names_filter=names_filter)
        if table_chunks:
            logger.info("Extracted %d table chunk(s) across all PDFs", len(table_chunks))
        chunks.extend(table_chunks)

        path_map = {p.name: p for p in new_paths}
        for c in chunks:
            src_path = path_map.get(c["source"])
            if src_path:
                c["source_hash"] = file_hashes[src_path]
        logger.info("Split into %d chunks", len(chunks))

        logger.info("Embedding %d chunks...", len(chunks))
        embeddings = self.embedder.encode(
            [c["text"] for c in chunks],
            batch_size=256,
            show_progress=True,
        )

        self.store.add(chunks, embeddings)
        clear_cache()
        logger.info("Ingestion complete: %d chunks stored.", len(chunks))

    def new_conversation_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def list_conversations(self) -> list:
        return self.store.list_conversations()

    def get_conversation_messages(self, conversation_id: str) -> list:
        return self.store.load_history(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.store.delete_conversation(conversation_id)

    def _ask_core(
        self,
        question: str,
        conversation_id: str,
        source_filter: str = None,
        top_k: int = 5,
        stream: bool = False,
        allow_general_knowledge: bool = False,
    ) -> dict | None:
        docs = retrieve(question, self.store, self.embedder, top_k=top_k, source_filter=source_filter, rerank_model=self.rerank_model)
        if not docs:
            return None

        history = self.store.load_history(conversation_id)
        pct = grounding.rerank_confidence_pct([d.get("rerank_score", 0) for d in docs])
        confidence = grounding.confidence_label(pct)

        if not allow_general_knowledge and pct < self.min_confidence:
            # Retrieval isn't confident enough to trust an LLM-generated answer —
            # refuse deterministically instead of hoping the prompt is obeyed.
            answer = grounding.NOT_FOUND_MESSAGE
            if stream:
                print(f"\nAnswer: {answer}", end="", flush=True)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            history = _trim_history(history)
            self.store.ensure_conversation(conversation_id, question)
            self.store.save_history(conversation_id, history)
            return {"answer": answer, "sources": [], "citations": [], "confidence": confidence, "grounded": False}

        messages = build_ask_messages(question, docs, history=history, allow_general_knowledge=allow_general_knowledge)
        if stream:
            print("\nAnswer: ", end="", flush=True)
        answer = self.llm.chat(messages, stream=stream)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        history = _trim_history(history)
        self.store.ensure_conversation(conversation_id, question)
        self.store.save_history(conversation_id, history)
        sources = sorted({f"{d['source']} p.{d['page']}" for d in docs})
        citations = grounding.build_citations(docs)
        return {"answer": answer, "sources": sources, "citations": citations, "confidence": confidence, "grounded": True}

    def ask(self, question: str, conversation_id: str, source_filter: str = None, top_k: int = 5, allow_general_knowledge: bool = False) -> str:
        result = self._ask_core(question, conversation_id, source_filter, top_k, stream=True, allow_general_knowledge=allow_general_knowledge)
        if result is None:
            print("No relevant documents found.")
            return ""
        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources: {', '.join(result['sources']) or '(none — answer not grounded in documents)'}\n")
        return result["answer"]

    def ask_stream(self, question: str, conversation_id: str, source_filter: str = None, top_k: int = 5, allow_general_knowledge: bool = False):
        """
        Yield text tokens one by one, then a final [META] event with sources + confidence.
        Designed for Server-Sent Events — caller never needs to call ask() separately.
        """
        docs = retrieve(question, self.store, self.embedder, top_k=top_k,
                        source_filter=source_filter, rerank_model=self.rerank_model)
        if not docs:
            yield "[NO_RESULTS]"
            return

        history = self.store.load_history(conversation_id)
        pct = grounding.rerank_confidence_pct([d.get("rerank_score", 0) for d in docs])
        confidence = grounding.confidence_label(pct)

        if not allow_general_knowledge and pct < self.min_confidence:
            answer = grounding.NOT_FOUND_MESSAGE
            yield answer
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            history = _trim_history(history)
            self.store.ensure_conversation(conversation_id, question)
            self.store.save_history(conversation_id, history)
            yield f"[META]{json.dumps({'sources': [], 'citations': [], 'confidence': confidence, 'grounded': False})}"
            return

        messages = build_ask_messages(question, docs, history=history, allow_general_knowledge=allow_general_knowledge)
        full_answer = ""
        for token in self.llm.stream(messages):
            full_answer += token
            yield token

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})
        history = _trim_history(history)
        self.store.ensure_conversation(conversation_id, question)
        self.store.save_history(conversation_id, history)

        sources = sorted({f"{d['source']} p.{d['page']}" for d in docs})
        citations = grounding.build_citations(docs)
        yield f"[META]{json.dumps({'sources': sources, 'citations': citations, 'confidence': confidence, 'grounded': True})}"

    def ask_raw(self, question: str, conversation_id: str, source_filter: str = None, top_k: int = 5, allow_general_knowledge: bool = False) -> dict:
        result = self._ask_core(question, conversation_id, source_filter, top_k, stream=False, allow_general_knowledge=allow_general_knowledge)
        if result is None:
            return {"answer": "", "sources": [], "citations": [], "confidence": "Unknown", "grounded": False, "found": False}
        return {**result, "found": True}

    def solve(self, problem: str, source_filter: str = None) -> str:
        analysis, self._solve_history = hyp.solve(
            problem,
            self.store,
            self.embedder,
            self.llm,
            source_filter=source_filter,
            history=self._solve_history,
            rerank_model=self.rerank_model,
        )
        self._solve_history = _trim_history(self._solve_history)
        self._save_solve_history()
        return analysis

    def sources(self) -> list:
        return self.store.get_sources()

    def delete_source(self, name: str) -> None:
        self.store.delete_source(name)

    def clear_solve_history(self) -> None:
        self._solve_history = []
        p = self._solve_history_path()
        if p.exists():
            p.unlink()
