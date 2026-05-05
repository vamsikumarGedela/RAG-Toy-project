import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# First words/phrases that trigger hypothesis (solve) mode automatically
_SOLVE_FIRST_WORDS = {"why"}
_SOLVE_TWO_WORDS   = {"how come", "what causes", "what caused", "explain why", "how could", "what makes"}


def _should_solve(text: str) -> bool:
    t = text.lower().strip()
    if t.startswith("solve:"):
        return True
    words = t.split()
    if not words:
        return False
    if words[0] in _SOLVE_FIRST_WORDS:
        return True
    return " ".join(words[:2]) in _SOLVE_TWO_WORDS


def select_provider() -> tuple:
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
        choice = input("\nSelect PDF number (default 0 = All): ").strip() or "0"
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(sources):
            return sources[int(choice) - 1]
        print("  Invalid choice.")


def _save_analysis(analysis: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"analysis_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(analysis)
    print(f"  Saved to: {filename}\n")


def cmd_ingest(pdf_dir: str) -> None:
    from minrag import RAG
    rag = RAG(llm_provider="ollama")
    rag.ingest(pdf_dir)


def cmd_chat() -> None:
    from minrag import RAG
    provider, api_key = select_provider()
    rag = RAG(llm_provider=provider, api_key=api_key)

    pdf_filter = select_pdf(rag)
    scope = pdf_filter or "All PDFs"

    print(f"""
Scope: {scope}
Just ask anything — query and hypothesis mode are automatic.

  Regular question  →  direct answer        e.g. "What is a linked list?"
  Starts with why / how come / what causes  →  hypothesis analysis
  solve: <question> →  force hypothesis mode on any question

Commands: switch | clear | exit
""")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            continue

        if text.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if text.lower() == "switch":
            pdf_filter = select_pdf(rag)
            scope = pdf_filter or "All PDFs"
            print(f"Switched to: {scope}\n")
            continue

        if text.lower() == "clear":
            rag.clear_history()
            rag.clear_solve_history()
            print("History cleared.\n")
            continue

        # Strip explicit "solve:" prefix before passing the question
        question = text[len("solve:"):].strip() if text.lower().startswith("solve:") else text

        if _should_solve(text):
            print("  [hypothesis mode — testing competing theories against your PDFs]\n")
            analysis = rag.solve(question, source_filter=pdf_filter)
            try:
                save = input("\nSave this analysis? (y/n): ").strip().lower()
                if save == "y":
                    _save_analysis(analysis)
                print()
            except (EOFError, KeyboardInterrupt):
                print()
        else:
            print("  [query mode]\n")
            rag.ask(question, source_filter=pdf_filter)


def print_usage() -> None:
    print(
        "\nminrag — lightweight RAG built from scratch\n"
        "\nUsage:\n"
        "  python main.py ingest [pdf_dir]   — embed PDFs (default: ./pdfs)\n"
        "  python main.py chat               — unified chat (auto query + hypothesis)\n"
        "\nInside chat:\n"
        "  Ask anything normally             → direct answer\n"
        "  Start with 'why / how come / what causes / ...' → hypothesis mode\n"
        "  solve: <question>                 → force hypothesis mode\n"
        "  switch                            → change PDF scope\n"
        "  clear                             → reset all history\n"
        "  exit                              → quit\n"
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
    elif command in {"chat", "query", "solve"}:
        cmd_chat()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
