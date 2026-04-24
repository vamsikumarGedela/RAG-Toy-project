import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def select_provider() -> tuple:
    """Let user pick LLM provider. Returns (provider, api_key)."""
    providers = {
        "1": ("ollama",     "Free — runs locally, no API key needed (requires Ollama installed)"),
        "2": ("openai",     "OpenAI — needs OPENAI_API_KEY"),
        "3": ("anthropic",  "Anthropic Claude — needs ANTHROPIC_API_KEY"),
        "4": ("openrouter", "OpenRouter — needs OPENROUTER_API_KEY (access to many models)"),
    }

    print("\nSelect LLM provider:")
    for num, (name, desc) in providers.items():
        print(f"  [{num}] {name:12} — {desc}")

    while True:
        choice = input("\nProvider number (default 1 = Ollama): ").strip() or "1"
        if choice in providers:
            provider = providers[choice][0]
            break
        print("  Invalid choice.")

    api_key = None
    env_map = {
        "openai":     "OPENAI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    if provider in env_map:
        env_var = env_map[provider]
        api_key = os.getenv(env_var)
        if not api_key:
            api_key = input(f"Enter your {env_var}: ").strip()

    return provider, api_key


def select_pdf(rag) -> str | None:
    sources = rag.sources()
    if not sources:
        print("  No PDFs ingested yet. Run: python main.py ingest")
        sys.exit(1)

    print("\nAvailable PDFs:")
    print("  [0] All PDFs")
    for i, name in enumerate(sources, 1):
        print(f"  [{i}] {name}")

    while True:
        choice = input("\nSelect PDF number: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(sources):
            return sources[int(choice) - 1]
        print("  Invalid choice.")


def cmd_ingest(pdf_dir: str) -> None:
    from minrag import RAG
    # Ingest doesn't need LLM — use ollama as placeholder (won't be called)
    rag = RAG(llm_provider="ollama")
    rag.ingest(pdf_dir)


def cmd_query() -> None:
    from minrag import RAG
    provider, api_key = select_provider()
    rag = RAG(llm_provider=provider, api_key=api_key)

    pdf_filter = select_pdf(rag)
    scope = pdf_filter or "All PDFs"
    print(f"\nQuerying: {scope}")
    print("Commands: 'switch' = change PDF  |  'clear' = reset history  |  'exit' = quit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if question.lower() == "switch":
            pdf_filter = select_pdf(rag)
            scope = pdf_filter or "All PDFs"
            print(f"Switched to: {scope}\n")
            continue
        if question.lower() == "clear":
            rag.clear_history()
            print("History cleared.\n")
            continue

        rag.ask(question, source_filter=pdf_filter)


def _save_analysis(analysis: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"analysis_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"  Saved to: {filename}\n")


def cmd_solve() -> None:
    from minrag import RAG
    provider, api_key = select_provider()
    rag = RAG(llm_provider=provider, api_key=api_key)

    pdf_filter = select_pdf(rag)
    scope = pdf_filter or "All PDFs"
    print(f"\nHypothesis Solver — Scope: {scope}")
    print("Commands: 'switch' = change PDF  |  'clear' = reset history  |  'exit' = quit\n")

    while True:
        try:
            problem = input("Problem: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not problem:
            continue
        if problem.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if problem.lower() == "switch":
            pdf_filter = select_pdf(rag)
            scope = pdf_filter or "All PDFs"
            print(f"Switched to: {scope}\n")
            continue
        if problem.lower() == "clear":
            rag.clear_solve_history()
            print("Solve history cleared.\n")
            continue

        analysis = rag.solve(problem, source_filter=pdf_filter)

        # Offer to export
        try:
            save = input("\nSave this analysis? (y/n): ").strip().lower()
            if save == "y":
                _save_analysis(analysis)
        except (EOFError, KeyboardInterrupt):
            pass


def print_usage() -> None:
    print(
        "\nminrag — lightweight RAG built from scratch\n"
        "\nUsage:\n"
        "  python main.py ingest [pdf_dir]   — embed PDFs (default: ./pdfs)\n"
        "  python main.py query              — interactive Q&A\n"
        "  python main.py solve              — hypothesis-driven problem solver (with history + export)\n"
        "\nLLM providers (selected at runtime):\n"
        "  ollama      — FREE, runs locally, no API key\n"
        "  openai      — needs OPENAI_API_KEY\n"
        "  anthropic   — needs ANTHROPIC_API_KEY\n"
        "  openrouter  — needs OPENROUTER_API_KEY\n"
    )


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print_usage()
        sys.exit(0)

    command = args[0].lower()

    if command == "ingest":
        cmd_ingest(args[1] if len(args) > 1 else "./pdfs")
    elif command == "query":
        cmd_query()
    elif command == "solve":
        cmd_solve()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
