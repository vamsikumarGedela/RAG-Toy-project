from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pypdf


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
        print(f"  Warning: could not read {path.name} — {e}")
    return pages


MIN_CHUNK_LEN = 50


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
        print(f"  Loaded: {path.name} ({len(result)} pages)")
    return pages


def chunk_pages(pages: list, chunk_size: int = 800, overlap: int = 100) -> list:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text and len(chunk_text) >= MIN_CHUNK_LEN:
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                })
            if end >= len(text):
                break
            start = end - overlap
    return chunks
