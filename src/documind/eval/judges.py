"""Answer-quality judges: an LLM-as-judge and an offline lexical fallback.

Both expose the same two methods so the runner does not care which one it gets:
    faithfulness(answer, contexts) -> float in [0, 1]
    relevance(question, answer, reference) -> float in [0, 1]
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from documind.eval.metrics import lexical_faithfulness, lexical_relevance


class Judge(Protocol):
    name: str

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float: ...

    def relevance(self, question: str, answer: str, reference: str) -> float: ...


class LexicalJudge:
    """Word-overlap heuristics. Free, fast, deterministic, but blind to paraphrase."""

    name = "lexical"

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        return lexical_faithfulness(answer, contexts)

    def relevance(self, question: str, answer: str, reference: str) -> float:
        return lexical_relevance(question, answer, reference)


_JUDGE_SYSTEM = (
    "You are a strict evaluator of a question-answering system. "
    'Reply with JSON only, in the form {{"score": <number between 0 and 1>, '
    '"reason": "<one sentence>"}}.'
)

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _JUDGE_SYSTEM),
        (
            "human",
            "Break the ANSWER into its factual claims and check each one against the CONTEXT.\n"
            "score = (claims fully supported by the CONTEXT) / (total claims).\n"
            "Ignore citation markers like [file p.N]. Outside knowledge does not count as "
            "support.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}",
        ),
    ]
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _JUDGE_SYSTEM),
        (
            "human",
            "Rate how well the ANSWER answers the QUESTION, using the REFERENCE answer as the "
            "ground truth.\n1.0 = complete and correct, 0.5 = partially correct or "
            "incomplete, 0.0 = wrong, off-topic or a refusal.\n\n"
            "QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}",
        ),
    ]
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_score(text: str) -> float:
    """Extract a [0, 1] score from the judge's reply, tolerating extra text around the JSON."""
    match = _JSON_RE.search(text)
    if match is None:
        raise ValueError(f"Judge reply contained no JSON: {text!r}")
    score = float(json.loads(match.group(0))["score"])
    return max(0.0, min(1.0, score))


class LLMJudge:
    """LLM-as-judge. Use a capable model at temperature 0 for stable scores."""

    def __init__(self, llm: BaseChatModel, name: str = "llm"):
        self.name = name
        self._faithfulness = FAITHFULNESS_PROMPT | llm | StrOutputParser()
        self._relevance = RELEVANCE_PROMPT | llm | StrOutputParser()

    def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        reply = self._faithfulness.invoke({"context": "\n\n".join(contexts), "answer": answer})
        return parse_score(reply)

    def relevance(self, question: str, answer: str, reference: str) -> float:
        reply = self._relevance.invoke(
            {"question": question, "answer": answer, "reference": reference}
        )
        return parse_score(reply)
