from __future__ import annotations

import pytest

from documind.errors import IngestionError
from documind.ingestion import chunk_pages, compute_doc_id, load_pdf_pages, pdf_to_chunks
from tests.conftest import make_pdf


def test_load_pdf_pages_keeps_page_numbers():
    data = make_pdf(["First page about apples.", "Second page about bananas."])
    pages = load_pdf_pages(data, "fruit.pdf")

    assert [p.metadata["page"] for p in pages] == [1, 2]
    assert all(p.metadata["source"] == "fruit.pdf" for p in pages)
    assert "bananas" in pages[1].page_content


def test_blank_pages_are_skipped_but_numbering_is_preserved():
    data = make_pdf(["Intro text.", "", "Text on page three."])
    pages = load_pdf_pages(data, "gaps.pdf")
    assert [p.metadata["page"] for p in pages] == [1, 3]


def test_invalid_pdf_raises_ingestion_error():
    with pytest.raises(IngestionError):
        load_pdf_pages(b"%PDF-1.4 this is not really a pdf", "broken.pdf")


def test_pdf_without_text_raises_ingestion_error():
    with pytest.raises(IngestionError, match="No extractable text"):
        load_pdf_pages(make_pdf(["", ""]), "empty.pdf")


def test_chunks_respect_size_and_carry_metadata(handbook_bytes):
    doc_id, chunks = pdf_to_chunks(handbook_bytes, "handbook.pdf", chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 6  # more chunks than pages at this size
    assert all(len(c.page_content) <= 200 for c in chunks)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == doc_id
        assert chunk.metadata["source"] == "handbook.pdf"
        assert 1 <= chunk.metadata["page"] <= 6


def test_smaller_chunk_size_gives_more_chunks(handbook_bytes):
    pages = load_pdf_pages(handbook_bytes, "handbook.pdf")
    small = chunk_pages(pages, "x", chunk_size=128, chunk_overlap=16)
    large = chunk_pages(pages, "x", chunk_size=1024, chunk_overlap=128)
    assert len(small) > len(large)


def test_overlap_repeats_text_between_neighbouring_chunks():
    sentence = " ".join(f"word{i}" for i in range(200))
    pages = load_pdf_pages(make_pdf([sentence[i : i + 90] for i in range(0, 900, 90)]), "x.pdf")
    joined = pages[0].model_copy(update={"page_content": sentence})
    chunks = chunk_pages([joined], "x", chunk_size=100, chunk_overlap=30)

    first, second = chunks[0].page_content, chunks[1].page_content
    assert first.split()[-1] in second.split()  # tail of chunk 1 reappears in chunk 2


def test_doc_id_is_content_based():
    a, b = make_pdf(["alpha"]), make_pdf(["beta"])
    assert compute_doc_id(a) == compute_doc_id(a)
    assert compute_doc_id(a) != compute_doc_id(b)
