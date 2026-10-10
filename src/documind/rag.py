"""The question-answering pipeline: retrieve -> build grounded prompt -> LLM -> cited answer."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from documind.models import Citation, RAGAnswer
from documind.prompts import (
    REFUSAL_MESSAGE,
    build_rag_prompt,
    extract_citations,
    format_context,
    is_refusal,
)
from documind.vectorstore import DocumentStore

SNIPPET_CHARS = 300


class RAGPipeline:
    def __init__(self, store: DocumentStore, llm: BaseChatModel, min_score: float = 0.0):
        """
        min_score: chunks scoring below this relevance are dropped before prompting.
        Leaving weak matches out makes it easier for the model to refuse correctly.
        """
        self.store = store
        self.min_score = min_score
        # LCEL chain: the prompt template fills {context}/{question}, the model answers,
        # and the parser turns the chat message into a plain string.
        self.chain = build_rag_prompt() | llm | StrOutputParser()

    def retrieve(self, question: str, top_k: int) -> list[tuple[Document, float]]:
        hits = self.store.search(question, k=top_k)
        return [(doc, score) for doc, score in hits if score >= self.min_score]

    def ask(self, question: str, top_k: int = 4) -> RAGAnswer:
        return self.generate(question, self.retrieve(question, top_k))

    def generate(self, question: str, hits: list[tuple[Document, float]]) -> RAGAnswer:
        """Answer from already-retrieved chunks (the evaluator calls this directly)."""
        if not hits:
            # Nothing to ground on: refuse without spending an LLM call.
            return RAGAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True)

        docs = [doc for doc, _ in hits]
        answer = self.chain.invoke({"context": format_context(docs), "question": question})
        answer = answer.strip()
        refused = is_refusal(answer)
        return RAGAnswer(
            answer=answer,
            citations=[] if refused else self._citations(hits, answer),
            refused=refused,
        )

    @staticmethod
    def _citations(hits: list[tuple[Document, float]], answer: str) -> list[Citation]:
        """All retrieved chunks, flagged with whether the answer actually cited their page."""
        cited_pages = extract_citations(answer)
        citations = []
        for doc, score in hits:
            meta = doc.metadata
            text = " ".join(doc.page_content.split())
            snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "..."
            citations.append(
                Citation(
                    source=meta["source"],
                    page=int(meta["page"]),
                    doc_id=meta["doc_id"],
                    chunk_id=meta["chunk_id"],
                    snippet=snippet,
                    score=round(score, 4),
                    cited=(meta["source"], int(meta["page"])) in cited_pages,
                )
            )
        return citations
