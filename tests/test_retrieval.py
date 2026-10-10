from __future__ import annotations

import numpy as np
import pytest

from documind.errors import DocumentExistsError, DocumentNotFoundError
from documind.fakes import HashingEmbeddings
from documind.vectorstore import DocumentStore
from tests.conftest import make_pdf


def test_hashing_embeddings_are_deterministic_and_normalised():
    emb = HashingEmbeddings(dim=128)
    a = emb.embed_query("annual leave policy")
    assert a == HashingEmbeddings(dim=128).embed_query("annual leave policy")
    assert len(a) == 128
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_similar_texts_are_closer_than_unrelated_ones():
    emb = HashingEmbeddings(dim=512)
    q = np.array(emb.embed_query("paid annual leave days"))
    related = np.array(emb.embed_query("Employees receive 24 days of paid annual leave."))
    unrelated = np.array(emb.embed_query("Laptops must use full-disk encryption."))
    assert q @ related > q @ unrelated


@pytest.mark.parametrize(
    ("question", "expected_page"),
    [
        ("How many days of paid annual leave do employees get?", 3),
        ("How long must passwords be?", 5),
        ("How much are hotel costs reimbursed per night?", 4),
    ],
)
def test_search_returns_gold_page_first(loaded_store, question, expected_page):
    results = loaded_store.search(question, k=3)
    top_doc, top_score = results[0]
    assert top_doc.metadata["page"] == expected_page
    assert 0.0 <= top_score <= 1.0
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_on_empty_store_returns_nothing(store):
    assert store.search("anything", k=4) == []


def test_multiple_documents_list_and_delete(store, handbook_bytes):
    store.add_pdf(handbook_bytes, "handbook.pdf", 512, 64)
    other = store.add_pdf(make_pdf(["The cafeteria serves lunch at noon."]), "cafe.pdf", 512, 64)

    assert {d.filename for d in store.list_documents()} == {"handbook.pdf", "cafe.pdf"}
    top_doc, _ = store.search("When is lunch served in the cafeteria?", k=1)[0]
    assert top_doc.metadata["source"] == "cafe.pdf"

    removed = store.delete_document(other.doc_id)
    assert removed == other.num_chunks
    assert [d.filename for d in store.list_documents()] == ["handbook.pdf"]
    assert all(d.metadata["source"] == "handbook.pdf" for d, _ in store.search("lunch", k=10))


def test_duplicate_upload_is_rejected(loaded_store, handbook_bytes):
    with pytest.raises(DocumentExistsError):
        loaded_store.add_pdf(handbook_bytes, "copy.pdf", 512, 64)


def test_delete_unknown_document_raises(store):
    with pytest.raises(DocumentNotFoundError):
        store.delete_document("does-not-exist")


def test_store_persists_across_instances(tmp_path, handbook_bytes):
    emb = HashingEmbeddings(dim=64)
    DocumentStore(tmp_path, "persist-test", emb).add_pdf(handbook_bytes, "handbook.pdf", 512, 64)
    reopened = DocumentStore(tmp_path, "persist-test", emb)
    assert [d.filename for d in reopened.list_documents()] == ["handbook.pdf"]
