"""Factories that turn configuration into concrete LangChain embedding / chat models.

Heavy or optional dependencies (sentence-transformers, OpenAI) are imported lazily so the
offline 'fake' mode works without them.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from documind.errors import DocuMindError
from documind.fakes import ExtractiveFakeChatModel, HashingEmbeddings


class ConfigurationError(DocuMindError):
    """Configuration is invalid or a required credential/dependency is missing."""


def get_embeddings(provider: str, model: str, openai_api_key: str | None = None) -> Embeddings:
    """Create an embedding model.

    provider: "huggingface" (local sentence-transformers model), "openai", or "fake".
    For "fake", `model` may be e.g. "hashing-384" to choose the vector size.
    """
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ConfigurationError("Install the 'hf' extra: pip install '.[hf]'") from exc
        # Normalised vectors make cosine similarity a plain dot product.
        return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
    if provider == "openai":
        if not openai_api_key:
            raise ConfigurationError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=openai_api_key)
    if provider == "fake":
        dim = int(model.rsplit("-", 1)[-1]) if model.rsplit("-", 1)[-1].isdigit() else 384
        return HashingEmbeddings(dim=dim)
    raise ConfigurationError(f"Unknown embedding provider: {provider!r}")


def get_llm(
    provider: str, model: str, temperature: float = 0.0, openai_api_key: str | None = None
) -> BaseChatModel:
    """Create the chat model used to generate answers (or to judge them during eval)."""
    if provider == "openai":
        if not openai_api_key:
            raise ConfigurationError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=openai_api_key, timeout=60)
    if provider == "fake":
        return ExtractiveFakeChatModel()
    raise ConfigurationError(f"Unknown LLM provider: {provider!r}")
