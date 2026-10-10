from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from documind.api.main import create_app
from documind.config import Settings
from tests.conftest import make_pdf


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None,  # ignore any developer .env
        embedding_provider="fake",
        embedding_model="hashing-256",
        llm_provider="fake",
        chroma_dir=str(tmp_path / "chroma"),
        max_upload_mb=1,
    )
    with TestClient(create_app(settings)) as test_client:  # `with` runs the lifespan
        yield test_client


def upload(client, data: bytes, name: str = "handbook.pdf"):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["embedding_provider"] == "fake"
    assert body["num_documents"] == 0


def test_upload_list_ask_delete_flow(client, handbook_bytes):
    resp = upload(client, handbook_bytes)
    assert resp.status_code == 201
    doc = resp.json()
    assert doc["filename"] == "handbook.pdf" and doc["num_pages"] == 6

    listed = client.get("/documents").json()["documents"]
    assert [d["doc_id"] for d in listed] == [doc["doc_id"]]

    resp = client.post("/ask", json={"question": "How long must passwords be?"})
    assert resp.status_code == 200
    answer = resp.json()
    assert not answer["refused"]
    assert "14 characters" in answer["answer"]
    assert "[handbook.pdf p.5]" in answer["answer"]
    citation = answer["citations"][0]
    assert {"source", "page", "snippet", "score", "cited"} <= citation.keys()

    resp = client.delete(f"/documents/{doc['doc_id']}")
    assert resp.status_code == 200
    assert resp.json()["chunks_deleted"] == doc["num_chunks"]
    assert client.get("/documents").json()["documents"] == []


def test_ask_respects_top_k(client, handbook_bytes):
    upload(client, handbook_bytes)
    resp = client.post("/ask", json={"question": "hotel cost per night", "top_k": 2})
    assert len(resp.json()["citations"]) == 2


def test_unanswerable_question_returns_refusal(client, handbook_bytes):
    upload(client, handbook_bytes)
    body = client.post("/ask", json={"question": "How large is the annual bonus?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_duplicate_upload_returns_409(client, handbook_bytes):
    upload(client, handbook_bytes)
    assert upload(client, handbook_bytes).status_code == 409


@pytest.mark.parametrize(
    ("data", "name", "status"),
    [
        (b"plain text", "notes.txt", 415),
        (b"not a pdf at all", "fake.pdf", 415),
        (b"%PDF-1.4 garbage", "broken.pdf", 422),
    ],
)
def test_bad_uploads_are_rejected(client, data, name, status):
    assert upload(client, data, name).status_code == status


def test_oversized_upload_returns_413(client):
    big = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10)
    assert upload(client, big).status_code == 413


def test_path_in_filename_is_stripped(client):
    resp = upload(client, make_pdf(["Hello world."]), "../../secret/evil.pdf")
    assert resp.status_code == 201
    assert resp.json()["filename"] == "evil.pdf"


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/documents/unknown").status_code == 404


@pytest.mark.parametrize(
    "payload", [{"question": ""}, {"question": "   "}, {"question": "hi", "top_k": 0}, {}]
)
def test_invalid_ask_requests_return_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
