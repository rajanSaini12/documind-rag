"""Application settings, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["huggingface", "openai", "fake"]
LLMProvider = Literal["openai", "fake"]


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case env vars (e.g. `TOP_K`)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    embedding_provider: EmbeddingProvider = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: LLMProvider = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_api_key: str | None = None

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    top_k: int = Field(default=4, ge=1, le=20)

    chroma_dir: str = "./chroma_data"
    chroma_collection: str = "documind"
    max_upload_mb: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def collection_name(self) -> str:
        """Collection name that is unique per embedding model.

        Vectors from different embedding models live in different spaces (and often
        have different dimensions), so they must never share a collection.
        """
        return make_collection_name(
            self.chroma_collection, self.embedding_provider, self.embedding_model
        )


def make_collection_name(base: str, provider: str, model: str) -> str:
    """Build a Chroma-safe collection name such as `documind-huggingface-all-minilm-l6-v2`."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{provider}-{model.split('/')[-1]}").strip("-")
    name = f"{base}-{slug}".lower()
    return name[:63].strip("-")


@lru_cache
def get_settings() -> Settings:
    return Settings()
