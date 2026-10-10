import pytest

from src.pdf_processor import chunk_pages, extract_pages


def test_chunk_pages_preserves_page_number():
    pages = [{"page_number": 3, "text": "A" * 1200}]

    chunks = chunk_pages(pages, chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert all(chunk["page_number"] == 3 for chunk in chunks)
    assert all(len(chunk["text"]) <= 500 for chunk in chunks)


def test_chunks_have_unique_ids():
    pages = [{"page_number": 1, "text": "B" * 1200}]

    chunks = chunk_pages(pages, chunk_size=500, overlap=100)
    ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([{"page_number": 1, "text": "  "}]) == []


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=0)


def test_extract_pages_rejects_invalid_pdf():
    with pytest.raises(Exception):
        extract_pages(b"this is not a PDF")
