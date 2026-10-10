"""PDF loading and chunking.

Pipeline: PDF bytes -> one `Document` per page (with page number) -> overlapping chunks.
We split page by page so every chunk belongs to exactly one page, which keeps
citations like [handbook.pdf p.3] accurate.
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from documind.errors import IngestionError
from documind.text_utils import normalize_whitespace


def compute_doc_id(data: bytes) -> str:
    """Stable document id derived from the file content (so re-uploads are detected)."""
    return hashlib.sha256(data).hexdigest()[:16]


def load_pdf_pages(data: bytes, filename: str) -> list[Document]:
    """Extract text from each page. Pages with no extractable text (e.g. scans) are skipped."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IngestionError(f"{filename} is encrypted; please upload an unlocked PDF.")
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise IngestionError(f"Could not read {filename} as a PDF: {exc}") from exc

    pages = [
        Document(page_content=text, metadata={"source": filename, "page": number})
        for number, raw in enumerate(raw_pages, start=1)
        if (text := normalize_whitespace(raw))
    ]
    if not pages:
        raise IngestionError(
            f"No extractable text found in {filename}. Scanned PDFs need OCR, "
            "which is not supported yet."
        )
    return pages


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Character-based recursive splitter.

    It tries to split on paragraph breaks first, then lines, sentences and words, so
    chunks end at natural boundaries where possible. Sizes are measured in characters.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def chunk_pages(
    pages: list[Document], doc_id: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split pages into chunks and attach the metadata needed for citations and deletion."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(pages)
    uploaded_at = datetime.now(UTC).isoformat(timespec="seconds")
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            doc_id=doc_id,
            chunk_index=index,
            chunk_id=f"{doc_id}-{index:05d}",
            num_pages=len(pages),
            uploaded_at=uploaded_at,
        )
    return chunks


def pdf_to_chunks(
    data: bytes, filename: str, chunk_size: int, chunk_overlap: int
) -> tuple[str, list[Document]]:
    """Convenience wrapper: bytes -> (doc_id, chunks)."""
    doc_id = compute_doc_id(data)
    pages = load_pdf_pages(data, filename)
    return doc_id, chunk_pages(pages, doc_id, chunk_size, chunk_overlap)
