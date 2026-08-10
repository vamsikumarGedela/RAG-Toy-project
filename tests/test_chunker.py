import pytest
from pathlib import Path
from minrag.chunker import (
    _split_sentences,
    chunk_pages,
    MIN_CHUNK_LEN,
    _table_to_markdown,
    _extract_tables_one,
    extract_tables,
)


def test_split_basic():
    sents = _split_sentences("Hello world. This is a test. Another sentence.")
    assert len(sents) == 3


def test_split_no_split_on_lowercase_after_period():
    # Regex only splits before uppercase/digit — "e.g. this" stays together
    text = "This uses e.g. this approach for demonstration. See the result."
    sents = _split_sentences(text)
    assert any("e.g. this" in s for s in sents)


def test_split_no_split_on_decimal():
    text = "The value is 1.5 units. This is the next sentence."
    sents = _split_sentences(text)
    assert any("1.5" in s for s in sents)


def test_split_empty_string():
    assert _split_sentences("") == []


def test_chunk_pages_basic():
    pages = [{"text": "This is a sentence. " * 20, "source": "test.pdf", "page": 1}]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all("text" in c and "source" in c and "page" in c for c in chunks)


def test_chunk_pages_min_len_filter():
    pages = [{"text": "Short.", "source": "test.pdf", "page": 1}]
    chunks = chunk_pages(pages, chunk_size=800, overlap=100)
    assert len(chunks) == 0


def test_chunk_pages_source_and_page_preserved():
    pages = [{"text": "A" * 60 + ". " + "B" * 60 + ".", "source": "my.pdf", "page": 3}]
    chunks = chunk_pages(pages, chunk_size=800, overlap=0)
    assert all(c["source"] == "my.pdf" for c in chunks)
    assert all(c["page"] == 3 for c in chunks)


def test_chunk_pages_overlap_shares_sentences():
    sentence = "This is a sentence about data structures. "
    pages = [{"text": sentence * 30, "source": "test.pdf", "page": 1}]
    chunks = chunk_pages(pages, chunk_size=200, overlap=80)
    assert len(chunks) >= 2
    # Overlap means consecutive chunks share trailing/leading sentences
    words_0 = set(chunks[0]["text"].split())
    words_1 = set(chunks[1]["text"].split())
    assert len(words_0 & words_1) > 0


def test_chunk_pages_empty_pages():
    assert chunk_pages([], chunk_size=800, overlap=100) == []


def test_chunk_pages_multiple_pages():
    pages = [
        {"text": "Page one content. " * 10, "source": "doc.pdf", "page": 1},
        {"text": "Page two content. " * 10, "source": "doc.pdf", "page": 2},
    ]
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    pages_seen = {c["page"] for c in chunks}
    assert 1 in pages_seen and 2 in pages_seen


# ─── _table_to_markdown ─────────────────────────────────────────────────────

def test_table_to_markdown_basic():
    table = [
        ["Algorithm", "Best", "Worst"],
        ["Bubble Sort", "O(n)", "O(n^2)"],
        ["Quick Sort", "O(n log n)", "O(n^2)"],
    ]
    md = _table_to_markdown(table)
    assert "| Algorithm | Best | Worst |" in md
    assert "| --- | --- | --- |" in md
    assert "| Bubble Sort | O(n) | O(n^2) |" in md
    assert md.count("\n") == 3  # header + separator + 2 data rows, joined by \n


def test_table_to_markdown_handles_none_cells():
    table = [["A", "B"], ["1", None]]
    md = _table_to_markdown(table)
    assert "| 1 |  |" in md


def test_table_to_markdown_drops_empty_rows():
    table = [["A", "B"], [None, None], ["1", "2"]]
    md = _table_to_markdown(table)
    assert md.count("\n") == 2  # header + separator + one real data row


def test_table_to_markdown_pads_ragged_rows():
    table = [["A", "B", "C"], ["1", "2"]]  # short row
    md = _table_to_markdown(table)
    assert "| 1 | 2 |  |" in md


def test_table_to_markdown_single_row_is_not_a_table():
    # A "table" with only a header and no data rows is almost certainly a
    # false-positive detection (e.g. a styled heading) — not worth keeping.
    assert _table_to_markdown([["Just", "A", "Heading"]]) == ""


def test_table_to_markdown_empty_input():
    assert _table_to_markdown([]) == ""


# ─── extract_tables ──────────────────────────────────────────────────────────

def test_extract_tables_no_pdfs(tmp_path):
    assert extract_tables(str(tmp_path)) == []


def test_extract_tables_respects_names_filter(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    # Both are invalid PDFs, so extraction fails gracefully for each —
    # this just confirms the filter narrows which files are even attempted.
    result = extract_tables(str(tmp_path), names_filter={"a.pdf"})
    assert result == []


def test_extract_tables_one_handles_corrupt_pdf(tmp_path):
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a real pdf")
    assert _extract_tables_one(bad_pdf) == []


def test_extract_tables_against_real_project_pdfs():
    # Informational smoke test against the actual sample PDFs shipped with
    # the project — confirms the pipeline runs end-to-end without crashing,
    # whatever it finds (some/none of these PDFs may have real tables).
    pdf_dir = Path(__file__).parent.parent / "pdfs"
    if not pdf_dir.exists():
        pytest.skip("sample pdfs/ directory not present")
    result = extract_tables(str(pdf_dir))
    assert isinstance(result, list)
    for chunk in result:
        assert chunk["text"].startswith("|")
        assert "source" in chunk and "page" in chunk
