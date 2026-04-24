# minrag

A lightweight RAG (Retrieval-Augmented Generation) library built completely from scratch.

**No LangChain. No ChromaDB. Full control.**

The unique feature is the **Hypothesis Engine** — instead of just answering questions, it generates multiple possible explanations, tests each one against your documents, and tells you which is most likely and why.

---

## What Makes It Different

| Others | minrag |
|---|---|
| Uses LangChain (black box) | Built from scratch (every line is transparent) |
| ChromaDB for vector storage | SQLite + NumPy (no extra database) |
| One answer, no reasoning | Hypothesis Engine — tests all possible explanations |
| Needs paid API key | Works free with Ollama (local LLM) |
| 14+ packages | Only 6 packages |

---

## Requirements

- Python 3.10 or higher
- One of the following LLMs:
  - **Ollama** (free, local) — recommended for beginners
  - OpenAI API key
  - Anthropic API key
  - OpenRouter API key

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/vamsikumarGedela/RAG-Toy-project.git
cd RAG-Toy-project
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up your API key

Copy the example env file:
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and fill in your key based on which provider you want:

```env
# Only fill in the one you want to use

OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

> **Want to use it for free?** Install [Ollama](https://ollama.com) and run:
> ```bash
> ollama pull llama3.2
> ```
> No API key needed.

### Step 5 — Add your PDFs

Create a `pdfs` folder and put your PDF files inside:
```bash
mkdir pdfs
# copy your PDF files into the pdfs folder
```

---

## Usage

### Ingest PDFs (run once)

```bash
python main.py ingest
```

This reads your PDFs, splits them into chunks, creates embeddings, and stores them in a local database (`minrag.db`).

> **Note:** If you add new PDFs later, run `ingest` again — it will only process the new ones.

### Ask Questions (Q&A Mode)

```bash
python main.py query
```

Select your LLM provider and PDF scope, then ask questions:

```
You: What is a linked list?
Answer: A linked list is a linear data structure...
Confidence: 85% (High)
Sources: Unit1.pdf p.3, Unit3.pdf p.7
```

**Available commands inside query mode:**
| Command | Action |
|---|---|
| `switch` | Change PDF scope |
| `clear` | Reset conversation history |
| `exit` | Quit |

### Solve Problems — Hypothesis Engine (unique feature)

```bash
python main.py solve
```

Type a "Why" or "Reason" problem:

```
Problem: Why is bubble sort slower than merge sort?
```

Output:
```
Step 1/4 — Generating hypotheses...
  H1: Bubble sort makes O(n²) comparisons
  H2: Bubble sort does excessive swaps
  H3: Merge sort uses divide-and-conquer
  H4: Bubble sort has no early termination

Step 2/4 — Retrieving evidence (parallel)...
Step 3/4 — Classifying evidence...
Step 4/4 — Generating verdict...

================================================================
HYPOTHESIS ANALYSIS
================================================================

✅ 91%  [LIKELY]    Bubble sort makes O(n²) comparisons...
     + "Each pass compares n elements..." (Unit1.pdf p.23)

⚠  55%  [POSSIBLE]  Bubble sort does excessive swaps...

❌  5%   [UNLIKELY]  Memory usage is the bottleneck...
     - "Bubble sort is in-place O(1) space..." (Unit1.pdf p.24)

================================================================
VERDICT
================================================================
Verdict: The primary reason bubble sort is slower...

Save this analysis? (y/n): y
→ Saved to: analysis_2026-04-22_14-30-45.txt
```

**Available commands inside solve mode:**
| Command | Action |
|---|---|
| `switch` | Change PDF scope |
| `clear` | Reset hypothesis history |
| `exit` | Quit |

---

## Use as a Python Library

You can also use minrag directly in your own Python code:

```python
from minrag import RAG

# Free — uses Ollama locally
rag = RAG()

# Or with an API key
rag = RAG(llm_provider="openrouter", api_key="your_key")
rag = RAG(llm_provider="openai",     api_key="your_key")
rag = RAG(llm_provider="anthropic",  api_key="your_key")

# Ingest PDFs
rag.ingest("./pdfs")

# Ask a question
rag.ask("What is a binary tree?")

# Solve a problem using hypothesis engine
rag.solve("Why is quicksort faster than bubble sort?")

# See ingested PDFs
print(rag.sources())

# Clear conversation history
rag.clear_history()       # clears query history
rag.clear_solve_history() # clears solve history
```

---

## Supported LLM Providers

| Provider | Free? | Setup |
|---|---|---|
| **Ollama** | Yes | Install from [ollama.com](https://ollama.com), run `ollama pull llama3.2` |
| **OpenRouter** | Free tier available | Get key from [openrouter.ai](https://openrouter.ai) |
| **OpenAI** | Paid | Get key from [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | Paid | Get key from [console.anthropic.com](https://console.anthropic.com), then `pip install anthropic` |

---

## Project Structure

```
minrag/
├── minrag/
│   ├── __init__.py      ← RAG class
│   ├── chunker.py       ← PDF reading + text splitting
│   ├── embedder.py      ← SentenceTransformer embeddings
│   ├── store.py         ← Vector database (SQLite + NumPy)
│   ├── retriever.py     ← BM25 + vector search + RRF + reranking
│   ├── llm.py           ← Multi-provider LLM interface
│   └── hypothesis.py    ← Hypothesis engine
├── main.py              ← CLI
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Built With

- [pypdf](https://pypdf.readthedocs.io) — PDF reading
- [sentence-transformers](https://www.sbert.net) — embeddings + reranking
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — keyword search
- [numpy](https://numpy.org) — vector math
- [openai](https://github.com/openai/openai-python) — OpenAI/Ollama/OpenRouter API
- [python-dotenv](https://github.com/theskumar/python-dotenv) — environment variables

---

## Author

Built by [vamsikumarGedela](https://github.com/vamsikumarGedela)
