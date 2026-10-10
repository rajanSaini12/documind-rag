"""Shared fixtures. Everything here is offline: fake embeddings, fake LLM, temp Chroma dirs."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from documind.fakes import ExtractiveFakeChatModel, HashingEmbeddings
from documind.rag import RAGPipeline
from documind.vectorstore import DocumentStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "eval" / "sample_docs" / "handbook.pdf"


def make_pdf(pages: list[str]) -> bytes:
    """Build a small in-memory PDF with one string of text per page."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture(scope="session")
def handbook_bytes() -> bytes:
    return SAMPLE_PDF.read_bytes()


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings(dim=256)


@pytest.fixture
def store(tmp_path: Path, embeddings: HashingEmbeddings) -> DocumentStore:
    return DocumentStore(tmp_path / "chroma", "test-collection", embeddings)


@pytest.fixture
def loaded_store(store: DocumentStore, handbook_bytes: bytes) -> DocumentStore:
    store.add_pdf(handbook_bytes, "handbook.pdf", chunk_size=512, chunk_overlap=64)
    return store


@pytest.fixture
def pipeline(loaded_store: DocumentStore) -> RAGPipeline:
    return RAGPipeline(loaded_store, ExtractiveFakeChatModel())
