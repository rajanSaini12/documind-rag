"""FastAPI backend.

Run locally with:
    uvicorn documind.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePath

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from documind import __version__
from documind.api.schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentListResponse,
    HealthResponse,
)
from documind.config import Settings, get_settings
from documind.errors import DocumentExistsError, DocumentNotFoundError, IngestionError
from documind.models import DocumentInfo
from documind.providers import get_embeddings, get_llm
from documind.rag import RAGPipeline
from documind.vectorstore import DocumentStore


@dataclass
class AppState:
    settings: Settings
    store: DocumentStore
    pipeline: RAGPipeline


def build_state(settings: Settings) -> AppState:
    """Wire up embeddings -> store -> LLM -> pipeline from configuration."""
    embeddings = get_embeddings(
        settings.embedding_provider, settings.embedding_model, settings.openai_api_key
    )
    store = DocumentStore(settings.chroma_dir, settings.collection_name, embeddings)
    llm = get_llm(
        settings.llm_provider, settings.llm_model, settings.llm_temperature, settings.openai_api_key
    )
    return AppState(settings=settings, store=store, pipeline=RAGPipeline(store, llm))


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory. Tests pass their own `Settings` (fake models, temp Chroma dir)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Loading an embedding model can take a few seconds, so do it once at startup.
        app.state.documind = build_state(settings or get_settings())
        yield

    app = FastAPI(
        title="DocuMind API",
        version=__version__,
        description="Upload PDFs and ask questions answered with source citations.",
        lifespan=lifespan,
    )

    def get_state(request: Request) -> AppState:
        return request.app.state.documind

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(state: AppState = Depends(get_state)) -> HealthResponse:
        s = state.settings
        return HealthResponse(
            status="ok",
            embedding_provider=s.embedding_provider,
            embedding_model=s.embedding_model,
            llm_provider=s.llm_provider,
            llm_model=s.llm_model,
            num_documents=len(state.store.list_documents()),
            num_chunks=state.store.count_chunks(),
        )

    @app.post(
        "/documents",
        response_model=DocumentInfo,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def upload_document(
        file: UploadFile = File(..., description="A PDF file"),
        state: AppState = Depends(get_state),
    ) -> DocumentInfo:
        # Strip any directory part a client might send, e.g. "../../etc/passwd.pdf".
        filename = PurePath(file.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported."
            )

        max_bytes = state.settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"File exceeds the {state.settings.max_upload_mb} MB limit.",
            )
        if not data.startswith(b"%PDF"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File is not a valid PDF.")

        try:
            # Parsing + embedding is CPU/network bound; keep the event loop free.
            return await run_in_threadpool(
                state.store.add_pdf,
                data,
                filename,
                state.settings.chunk_size,
                state.settings.chunk_overlap,
            )
        except DocumentExistsError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except IngestionError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    @app.get("/documents", response_model=DocumentListResponse, tags=["documents"])
    def list_documents(state: AppState = Depends(get_state)) -> DocumentListResponse:
        return DocumentListResponse(documents=state.store.list_documents())

    @app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["documents"])
    def delete_document(doc_id: str, state: AppState = Depends(get_state)) -> DeleteResponse:
        try:
            removed = state.store.delete_document(doc_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        return DeleteResponse(doc_id=doc_id, chunks_deleted=removed)

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    def ask(body: AskRequest, state: AppState = Depends(get_state)) -> AskResponse:
        question = body.question.strip()
        if not question:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "Question must not be empty."
            )
        top_k = body.top_k or state.settings.top_k
        try:
            result = state.pipeline.ask(question, top_k=top_k)
        except Exception as exc:  # e.g. OpenAI timeout / auth error
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, f"Answer generation failed: {exc}"
            ) from exc
        return AskResponse(question=question, **result.model_dump())

    return app


app = create_app()
