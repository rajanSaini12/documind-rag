from __future__ import annotations

import pytest
from pydantic import ValidationError

from documind.config import Settings, make_collection_name
from documind.fakes import ExtractiveFakeChatModel, HashingEmbeddings
from documind.providers import ConfigurationError, get_embeddings, get_llm


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    settings = Settings(_env_file=None)
    assert settings.top_k == 7
    assert settings.embedding_provider == "fake"


def test_collection_name_differs_per_embedding_model():
    a = make_collection_name("documind", "huggingface", "sentence-transformers/all-MiniLM-L6-v2")
    b = make_collection_name("documind", "huggingface", "BAAI/bge-small-en-v1.5")
    assert a == "documind-huggingface-all-minilm-l6-v2"
    assert a != b


def test_fake_providers():
    assert isinstance(get_embeddings("fake", "hashing-128"), HashingEmbeddings)
    assert get_embeddings("fake", "hashing-128").dim == 128
    assert isinstance(get_llm("fake", "anything"), ExtractiveFakeChatModel)


def test_openai_providers_require_api_key():
    with pytest.raises(ConfigurationError):
        get_llm("openai", "gpt-4o-mini", openai_api_key=None)
    with pytest.raises(ConfigurationError):
        get_embeddings("openai", "text-embedding-3-small", openai_api_key=None)


def test_openai_llm_is_constructed_without_network():
    llm = get_llm("openai", "gpt-4o-mini", openai_api_key="sk-test-not-real")
    assert llm.model_name == "gpt-4o-mini"


def test_unknown_provider():
    with pytest.raises(ConfigurationError):
        get_embeddings("nope", "x")
