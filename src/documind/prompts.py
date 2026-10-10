"""The grounding prompt: the main lever we have against hallucinated answers."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

REFUSAL_MESSAGE = "I don't know based on the documents."
CONTEXT_SEPARATOR = "\n\n---\n\n"
QUESTION_PREFIX = "Question:"

# Matches citations such as "[handbook.pdf p.3]" (the file name may contain spaces).
CITATION_RE = re.compile(r"\[([^\[\]]+?)\s+p\.\s?(\d+)\]")

SYSTEM_PROMPT = f"""You are DocuMind, an assistant that answers questions about the user's \
documents.

Rules:
1. Answer ONLY using the information in the provided context. Do not use outside knowledge.
2. After every sentence that uses the context, cite its source exactly as it appears in the \
context header, in the form [file p.N], for example [handbook.pdf p.3].
3. If the context does not contain the answer, reply exactly: "{REFUSAL_MESSAGE}" \
Do not guess and do not add anything else.
4. Be concise: answer in at most a few sentences."""

HUMAN_TEMPLATE = (
    """Context:
{context}

"""
    + QUESTION_PREFIX
    + " {question}"
)


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)])


def format_citation(source: str, page: int) -> str:
    return f"[{source} p.{page}]"


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with a citation header the model can copy verbatim."""
    blocks = [
        f"Source: {format_citation(d.metadata['source'], int(d.metadata['page']))}\n"
        f"{d.page_content.strip()}"
        for d in docs
    ]
    return CONTEXT_SEPARATOR.join(blocks)


def extract_citations(answer: str) -> set[tuple[str, int]]:
    """Return the (source, page) pairs the answer cites."""
    return {(src.strip(), int(page)) for src, page in CITATION_RE.findall(answer)}


def strip_citations(answer: str) -> str:
    return CITATION_RE.sub("", answer).strip()


def is_refusal(answer: str) -> bool:
    """True if the answer is (or contains) the refusal phrase, ignoring case and quote style."""
    normalized = answer.lower().replace("’", "'")
    return "i don't know based on the documents" in normalized
