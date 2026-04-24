import math
from pathlib import Path
from dotenv import load_dotenv

from .chunker import load_pdfs, chunk_pages
from .embedder import Embedder
from .store import VectorStore
from .retriever import retrieve, clear_cache
from .llm import LLM, build_ask_messages
from . import hypothesis as hyp

load_dotenv()

__version__ = "0.1.0"
__all__ = ["RAG"]


def _rerank_to_confidence(scores: list) -> str:
    if not scores:
        return "Unknown"
    avg = sum(scores) / len(scores)
    pct = round(100 / (1 + math.exp(-avg / 2)))
    if pct >= 70:
        return f"{pct}% (High)"
    elif pct >= 40:
        return f"{pct}% (Medium)"
    else:
        return f"{pct}% (Low)"


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
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedder = Embedder(embed_model)
        self.store = VectorStore(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model = rerank_model
        self.llm = LLM(provider=llm_provider, model=llm_model, api_key=api_key, temperature=temperature)
        self._query_history: list = []
        self._solve_history: list = []

        print(f"  minrag v{__version__} | provider: {llm_provider} | model: {self.llm.model}")

    def ingest(self, pdf_dir: str = "./pdfs", force: bool = False) -> None:
        print(f"\n=== minrag: Ingesting PDFs from '{pdf_dir}' ===")

        all_paths = list(Path(pdf_dir).glob("**/*.pdf"))
        if not all_paths:
            print("  No PDFs found.")
            return

        if force:
            self.store.clear()
            clear_cache()
            new_paths = all_paths
        else:
            new_paths = [p for p in all_paths if not self.store.source_exists(p.name)]
            skipped = len(all_paths) - len(new_paths)
            if skipped:
                print(f"  Skipping {skipped} already-ingested PDF(s).")
            if not new_paths:
                print("  All PDFs already ingested. Nothing to do.")
                return

        names_filter = {p.name for p in new_paths}
        pages = load_pdfs(pdf_dir, names_filter=names_filter)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        print(f"  Split into {len(chunks)} chunks")

        print(f"  Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.encode(
            [c["text"] for c in chunks],
            batch_size=256,
            show_progress=True,
        )

        self.store.add(chunks, embeddings)
        clear_cache()
        print(f"=== Ingestion complete: {len(chunks)} chunks stored ===\n")

    def ask(self, question: str, source_filter: str = None, top_k: int = 5) -> str:
        docs = retrieve(question, self.store, self.embedder, top_k=top_k, source_filter=source_filter, rerank_model=self.rerank_model)
        if not docs:
            print("No relevant documents found.")
            return ""

        messages = build_ask_messages(question, docs, history=self._query_history)
        print("\nAnswer: ", end="", flush=True)
        answer = self.llm.chat(messages, stream=True)

        self._query_history.append({"role": "user", "content": question})
        self._query_history.append({"role": "assistant", "content": answer})
        if len(self._query_history) > 12:
            self._query_history = self._query_history[-12:]

        # Show sources + confidence
        sources = list({f"{d['source']} p.{d['page']}" for d in docs})
        rerank_scores = [d.get("rerank_score", 0) for d in docs]
        confidence = _rerank_to_confidence(rerank_scores)
        print(f"\nConfidence: {confidence}")
        print(f"Sources: {', '.join(sorted(sources))}\n")
        return answer

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
        return analysis

    def sources(self) -> list:
        return self.store.get_sources()

    def clear_history(self) -> None:
        self._query_history = []

    def clear_solve_history(self) -> None:
        self._solve_history = []
