import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from .retriever import retrieve

N_HYPOTHESES = 4
W = 62
LOW_CONFIDENCE_THRESHOLD = 30.0


def _parse_json_array(text: str) -> list:
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return []


def generate_hypotheses(problem: str, llm, history: list = None, n: int = N_HYPOTHESES) -> list:
    messages = [{
        "role": "system",
        "content": (
            f"Generate {n} distinct hypotheses that could explain or solve the given problem. "
            "Each must be a different angle or root cause. One sentence each.\n"
            "If there is prior conversation context, use it to generate more relevant hypotheses.\n"
            "Return ONLY a JSON array of strings: [\"H1\", \"H2\", ...]"
        ),
    }]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": problem})

    raw = llm.chat(messages, stream=False)
    result = _parse_json_array(raw)
    if result:
        return result[:n]
    lines = [l.strip().lstrip("0123456789.-) ") for l in raw.splitlines() if l.strip()]
    return lines[:n]


def classify_evidence(hypothesis: str, chunks: list, llm) -> list:
    if not chunks:
        return []
    chunks_text = "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Classify each document chunk as SUPPORTS, CONTRADICTS, or IRRELEVANT "
                "to the given hypothesis.\n"
                "Return ONLY a JSON array with one label per chunk: [\"SUPPORTS\", ...]"
            ),
        },
        {
            "role": "user",
            "content": f"Hypothesis: {hypothesis}\n\nChunks:\n{chunks_text}",
        },
    ]
    raw = llm.chat(messages, stream=False)
    try:
        labels = _parse_json_array(raw)
        valid = {"SUPPORTS", "CONTRADICTS", "IRRELEVANT"}
        normalized = [
            l.upper() if l.upper() in valid else "IRRELEVANT"
            for l in labels
        ]
        while len(normalized) < len(chunks):
            normalized.append("IRRELEVANT")
        return normalized[:len(chunks)]
    except Exception:
        return ["IRRELEVANT"] * len(chunks)


def _confidence(n_support: int, n_contradict: int, total: int) -> float:
    if total == 0:
        return 0.0
    raw = (n_support - 0.5 * n_contradict) / total
    return round(max(0.0, min(1.0, raw)) * 100, 1)


def _build_analysis_lines(problem: str, results: list) -> list:
    lines = []
    lines.append("=" * W)
    lines.append(f"PROBLEM: {problem}")
    lines.append("=" * W)
    lines.append("HYPOTHESIS ANALYSIS")
    lines.append("=" * W)

    for r in results:
        conf = r["confidence"]
        if conf >= 70:
            icon, tag = "✅", "LIKELY"
        elif conf >= 35:
            icon, tag = "⚠ ", "POSSIBLE"
        else:
            icon, tag = "❌", "UNLIKELY"

        lines.append(f"\n{icon} {conf}%  [{tag}]  {r['hypothesis']}")

        for doc, _ in r["supporting"][:2]:
            snippet = doc["text"][:160].replace("\n", " ")
            lines.append(f'     + "{snippet}..."')
            lines.append(f"       ({doc['source']} p.{doc['page']})")

        for doc, _ in r["contradicting"][:2]:
            snippet = doc["text"][:160].replace("\n", " ")
            lines.append(f'     - "{snippet}..."')
            lines.append(f"       ({doc['source']} p.{doc['page']})")

        if not r["supporting"] and not r["contradicting"]:
            lines.append("     (no relevant evidence found in documents)")

    return lines


def solve(
    problem: str,
    store,
    embedder,
    llm,
    source_filter: str = None,
    history: list = None,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> tuple:
    """
    Returns (verdict, updated_history) so RAG class can maintain solve history.
    """
    if history is None:
        history = []

    print("\n" + "=" * W)
    print(f"PROBLEM: {problem}")
    print("=" * W)

    # Step 1 — generate hypotheses (aware of history)
    print("\nStep 1/4 — Generating hypotheses...")
    hypotheses = generate_hypotheses(problem, llm, history=history)
    for i, h in enumerate(hypotheses, 1):
        print(f"  H{i}: {h}")

    # Step 2 — retrieve evidence per hypothesis in parallel
    print("\nStep 2/4 — Retrieving evidence (parallel)...")

    def _retrieve(h):
        try:
            return retrieve(f"{problem} {h}", store, embedder, source_filter=source_filter, rerank_model=rerank_model)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=len(hypotheses)) as ex:
        evidence_docs = list(ex.map(_retrieve, hypotheses))

    # Step 3 — classify evidence (parallel across hypotheses)
    print("Step 3/4 — Classifying evidence (parallel)...")

    def _classify(args):
        hypothesis, docs = args
        labels = classify_evidence(hypothesis, docs, llm)
        supporting    = [(d, l) for d, l in zip(docs, labels) if l == "SUPPORTS"]
        contradicting = [(d, l) for d, l in zip(docs, labels) if l == "CONTRADICTS"]
        confidence    = _confidence(len(supporting), len(contradicting), len(docs))
        return {
            "hypothesis":    hypothesis,
            "supporting":    supporting,
            "contradicting": contradicting,
            "confidence":    confidence,
        }

    with ThreadPoolExecutor(max_workers=len(hypotheses)) as ex:
        results = list(ex.map(_classify, zip(hypotheses, evidence_docs)))

    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Step 4 — print analysis
    print("Step 4/4 — Generating verdict...\n")
    analysis_lines = _build_analysis_lines(problem, results)
    for line in analysis_lines:
        print(line)

    # Confidence threshold warning
    max_conf = max(r["confidence"] for r in results)
    if max_conf < LOW_CONFIDENCE_THRESHOLD:
        print(f"\n{'⚠ ' * 10}")
        print("⚠  WARNING: Low confidence across all hypotheses.")
        print("   Your documents may not cover this topic well.")
        print("   Consider adding more relevant PDFs.")
        print(f"{'⚠ ' * 10}\n")

    # Verdict
    print(f"\n{'=' * W}")
    print("VERDICT")
    print(f"{'=' * W}\n")

    evidence_summary = "\n".join(
        f"H{i+1} [{r['confidence']}%]: {r['hypothesis']} "
        f"| support: {len(r['supporting'])} | contradict: {len(r['contradicting'])}"
        for i, r in enumerate(results)
    )

    verdict_messages = [
        {
            "role": "system",
            "content": (
                "Given a problem and ranked hypotheses with evidence scores, write:\n"
                "1. Most likely explanation (grounded in evidence)\n"
                "2. Why other hypotheses are less likely\n"
                "3. Clear recommended action or conclusion\n"
                "Be direct. Base verdict ONLY on provided evidence."
            ),
        },
    ]
    if history:
        verdict_messages.extend(history)
    verdict_messages.append({
        "role": "user",
        "content": f"Problem: {problem}\n\nHypotheses:\n{evidence_summary}",
    })

    print("Verdict: ", end="", flush=True)
    verdict = llm.chat(verdict_messages, stream=True)
    print("\n" + "=" * W)

    # Build full text for export
    full_analysis = "\n".join(analysis_lines)
    full_analysis += f"\n\nVERDICT\n{'=' * W}\n{verdict}\n"

    # Update history with this exchange
    updated_history = history + [
        {"role": "user", "content": f"Problem: {problem}"},
        {"role": "assistant", "content": f"Verdict: {verdict}"},
    ]
    if len(updated_history) > 12:
        updated_history = updated_history[-12:]

    return full_analysis, updated_history
