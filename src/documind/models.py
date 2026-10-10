"""Pydantic data models shared by the pipeline and the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str = Field(description="Content hash of the uploaded file.")
    filename: str
    num_pages: int
    num_chunks: int
    uploaded_at: str = Field(description="ISO-8601 UTC timestamp.")


class Citation(BaseModel):
    source: str = Field(description="File name of the source document.")
    page: int = Field(description="1-based page number.")
    doc_id: str
    chunk_id: str
    snippet: str = Field(description="Beginning of the retrieved chunk.")
    score: float = Field(description="Relevance score in [0, 1]; higher is more similar.")
    cited: bool = Field(
        default=False, description="True if the answer explicitly cites this source/page."
    )


class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool = Field(description="True if the model said it could not answer.")
