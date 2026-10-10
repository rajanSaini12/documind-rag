"""DocuMind chat UI. It only talks to the FastAPI backend over HTTP (see BACKEND_URL).

Run locally with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = 120  # embedding a large PDF or a slow LLM call can take a while


# --------------------------------------------------------------------------- API helpers
def api(method: str, path: str, **kwargs: Any) -> requests.Response:
    return requests.request(method, f"{BACKEND_URL}{path}", timeout=TIMEOUT_S, **kwargs)


def error_detail(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text


def fetch_documents() -> list[dict[str, Any]]:
    resp = api("GET", "/documents")
    resp.raise_for_status()
    return resp.json()["documents"]


# --------------------------------------------------------------------------- rendering
def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            marker = "cited" if c.get("cited") else "retrieved"
            st.markdown(
                f"**{c['source']} - page {c['page']}** · score {c['score']:.2f} · _{marker}_"
            )
            st.caption(c["snippet"])


def render_sidebar() -> None:
    st.sidebar.header("Documents")
    uploaded = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.sidebar.button("Ingest", type="primary"):
        with st.sidebar.status(f"Ingesting {uploaded.name}..."):
            resp = api(
                "POST",
                "/documents",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
        if resp.status_code == 201:
            info = resp.json()
            st.sidebar.success(f"Added {info['filename']}: {info['num_chunks']} chunks.")
        else:
            st.sidebar.error(error_detail(resp))

    try:
        documents = fetch_documents()
    except requests.RequestException as exc:
        st.sidebar.error(f"Backend unreachable at {BACKEND_URL}: {exc}")
        return

    if not documents:
        st.sidebar.info("No documents yet. Upload a PDF to start.")
    for doc in documents:
        col_name, col_delete = st.sidebar.columns([4, 1])
        col_name.markdown(
            f"**{doc['filename']}**  \n{doc['num_pages']} pages · {doc['num_chunks']} chunks"
        )
        if col_delete.button("✕", key=f"del-{doc['doc_id']}", help="Delete this document"):
            resp = api("DELETE", f"/documents/{doc['doc_id']}")
            if resp.ok:
                st.rerun()
            st.sidebar.error(error_detail(resp))

    st.sidebar.divider()
    st.session_state.top_k = st.sidebar.slider("Chunks to retrieve (top-k)", 1, 10, 4)
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")
    st.title("DocuMind")
    st.caption("Ask questions about your PDFs. Answers cite the pages they come from.")

    st.session_state.setdefault("messages", [])
    render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_citations(message.get("citations", []))

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Searching your documents..."):
        try:
            resp = api("POST", "/ask", json={"question": question, "top_k": st.session_state.top_k})
        except requests.RequestException as exc:
            resp = None
            answer, citations = f"Could not reach the backend: {exc}", []
        if resp is not None:
            if resp.ok:
                body = resp.json()
                answer, citations = body["answer"], body["citations"]
            else:
                answer, citations = f"Error: {error_detail(resp)}", []
        st.markdown(answer)
        render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


main()
