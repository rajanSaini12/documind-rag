"""Request/response bodies for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from documind.models import Citation, DocumentInfo


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["How many leave days do I get?"])
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Chunks to retrieve (defaults to TOP_K)."
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Citation]


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int


class HealthResponse(BaseModel):
    status: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    num_documents: int
    num_chunks: int
