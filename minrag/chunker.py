import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pypdf

logger = logging.getLogger(__name__)

# Split on sentence endings followed by whitespace + capital/digit/quote.
# Avoids splitting on "Fig. 1", "e.g.", "1.5", etc.
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\d"\'])')

MIN_CHUNK_LEN = 50


def _split_sentences(text: str) -> list:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _load_one(path: Path) -> list:
    pages = []
    try:
        reader = pypdf.PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text.strip(),
                    "source": path.name,
                    "page": i + 1,
                })
    except Exception as e:
        logger.warning("Could not read %s — %s", path.name, e)
    return pages


def load_pdfs(pdf_dir: str, names_filter: set = None) -> list:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: '{pdf_dir}'")
    if not pdf_path.is_dir():
        raise NotADirectoryError(f"'{pdf_dir}' is not a directory")
    paths = list(pdf_path.glob("**/*.pdf"))
    if names_filter:
        paths = [p for p in paths if p.name in names_filter]
    if not paths:
        return []
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(_load_one, paths))
    pages = []
    for path, result in zip(paths, results):
        pages.extend(result)
        logger.info("Loaded %s (%d pages)", path.name, len(result))
    return pages


def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber-style table (list of rows, each a list of cells)
    into a markdown table string. Returns '' if there's nothing worth keeping —
    a single-row table isn't a table, it's noise from a false-positive detection."""
    rows = [[("" if cell is None else str(cell).strip()) for cell in row] for row in table]
    rows = [r for r in rows if any(c for c in r)]  # drop fully-empty rows
    if len(rows) < 2:
        return ""

    header, *body = rows
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        row = (row + [""] * width)[:width]  # pad/truncate ragged rows to header width
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_tables_one(path: Path) -> list:
    tables_out = []
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                for table in page.extract_tables():
                    md = _table_to_markdown(table)
                    if md:
                        tables_out.append({"text": md, "source": path.name, "page": i + 1})
    except Exception as e:
        logger.warning("Could not extract tables from %s — %s", path.name, e)
    return tables_out


def extract_tables(pdf_dir: str, names_filter: set = None) -> list:
    """Find every table in the given PDFs and return each as its own chunk,
    formatted as markdown so rows/columns survive instead of being flattened
    into a jumbled sentence. Kept fully separate from chunk_pages() — tables
    must never be cut mid-row by sentence-based chunking."""
    pdf_path = Path(pdf_dir)
    paths = list(pdf_path.glob("**/*.pdf"))
    if names_filter:
        paths = [p for p in paths if p.name in names_filter]
    if not paths:
        return []
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(_extract_tables_one, paths))
    tables = []
    for path, result in zip(paths, results):
        tables.extend(result)
        if result:
            logger.info("Extracted %d table(s) from %s", len(result), path.name)
    return tables


def chunk_pages(pages: list, chunk_size: int = 800, overlap: int = 100) -> list:
    chunks = []
    for page in pages:
        sentences = _split_sentences(page["text"])
        current: list = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent) + 1  # +1 for the space between sentences
            if current and current_len + sent_len > chunk_size:
                chunk_text = " ".join(current)
                if len(chunk_text) >= MIN_CHUNK_LEN:
                    chunks.append({"text": chunk_text, "source": page["source"], "page": page["page"]})

                # Build overlap: keep trailing sentences that fit within `overlap` chars
                overlap_sents: list = []
                overlap_len = 0
                for s in reversed(current):
                    s_len = len(s) + 1
                    if overlap_len + s_len <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += s_len
                    else:
                        break
                current = overlap_sents
                current_len = overlap_len

            current.append(sent)
            current_len += sent_len

        if current:
            chunk_text = " ".join(current)
            if len(chunk_text) >= MIN_CHUNK_LEN:
                chunks.append({"text": chunk_text, "source": page["source"], "page": page["page"]})

    return chunks
