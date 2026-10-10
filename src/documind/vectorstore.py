"""A thin wrapper around a persistent Chroma collection that knows about *documents*.

Chroma stores chunks. Each chunk carries `doc_id`, `source`, `page` and `chunk_id`
metadata, which is enough to list documents, delete one, and build citations without a
separate database.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from documind.errors import DocumentExistsError, DocumentNotFoundError
from documind.ingestion import pdf_to_chunks
from documind.models import DocumentInfo

_ADD_BATCH_SIZE = 256  # keep embedding requests and Chroma inserts reasonably sized


class DocumentStore:
    def __init__(self, persist_dir: str | Path, collection_name: str, embeddings: Embeddings):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._chroma = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            # Cosine distance -> relevance score = 1 - distance, easy to interpret.
            collection_configuration={"hnsw": {"space": "cosine"}},
        )

    # ------------------------------------------------------------------ writes
    def add_pdf(
        self, data: bytes, filename: str, chunk_size: int, chunk_overlap: int
    ) -> DocumentInfo:
        doc_id, chunks = pdf_to_chunks(data, filename, chunk_size, chunk_overlap)
        if self._chunk_ids(doc_id):
            raise DocumentExistsError(f"{filename} has already been uploaded (id {doc_id}).")
        self.add_chunks(chunks)
        meta = chunks[0].metadata
        return DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            num_pages=meta["num_pages"],
            num_chunks=len(chunks),
            uploaded_at=meta["uploaded_at"],
        )

    def add_chunks(self, chunks: list[Document]) -> None:
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start : start + _ADD_BATCH_SIZE]
            self._chroma.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of a document. Returns the number of chunks removed."""
        ids = self._chunk_ids(doc_id)
        if not ids:
            raise DocumentNotFoundError(f"No document with id {doc_id}")
        self._chroma.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------- reads
    def list_documents(self) -> list[DocumentInfo]:
        # Fine for a personal/portfolio-scale corpus; a large deployment would keep a
        # separate document table instead of scanning chunk metadata.
        metadatas = self._chroma.get(include=["metadatas"])["metadatas"]
        docs: dict[str, DocumentInfo] = {}
        for meta in metadatas:
            info = docs.get(meta["doc_id"])
            if info is None:
                docs[meta["doc_id"]] = DocumentInfo(
                    doc_id=meta["doc_id"],
                    filename=meta["source"],
                    num_pages=meta["num_pages"],
                    num_chunks=1,
                    uploaded_at=meta["uploaded_at"],
                )
            else:
                info.num_chunks += 1
        return sorted(docs.values(), key=lambda d: d.uploaded_at)

    def count_chunks(self) -> int:
        return len(self._chroma.get(include=[])["ids"])

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Semantic search. Returns (chunk, relevance score in [0, 1]) best-first."""
        if self.count_chunks() == 0:
            return []
        # Chroma returns cosine *distance* in [0, 2]; cosine similarity = 1 - distance.
        # Negative similarity (opposite direction) is clamped to 0 for a clean [0, 1] score.
        results = self._chroma.similarity_search_with_score(query, k=k)
        return [(doc, max(0.0, min(1.0, 1.0 - distance))) for doc, distance in results]

    def _chunk_ids(self, doc_id: str) -> list[str]:
        return self._chroma.get(where={"doc_id": doc_id}, include=[])["ids"]
