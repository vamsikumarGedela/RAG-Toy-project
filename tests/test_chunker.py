import pytest
from minrag.chunker import _split_sentences, chunk_pages, MIN_CHUNK_LEN


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
