"""Deterministic, offline stand-ins for the embedding model and the LLM.

They let the whole pipeline (API, UI, eval, tests) run with no network and no API key.
They are *not* meant to be good models, but they are honest ones:

* `HashingEmbeddings` is a bag-of-words "hashing trick" embedding. Texts that share
  words get similar vectors, so retrieval behaves sensibly on lexical matches.
* `ExtractiveFakeChatModel` reads the grounded prompt, picks the context sentence that
  best overlaps with the question, and cites it. If nothing overlaps enough it refuses,
  mimicking the behaviour the real prompt asks of a real LLM.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from documind.prompts import CONTEXT_SEPARATOR, QUESTION_PREFIX, REFUSAL_MESSAGE
from documind.text_utils import content_tokens, split_sentences

_SOURCE_HEADER_RE = re.compile(r"^Source: (\[.+? p\.\d+\])\s*\n", re.MULTILINE)


class HashingEmbeddings(Embeddings):
    """Feature-hashed bag of unigrams + bigrams, L2-normalised.

    Uses md5 rather than Python's `hash()` so vectors are stable across processes
    (Python salts string hashes per process).
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, feature: str) -> int:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def _embed(self, text: str) -> list[float]:
        tokens = content_tokens(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = np.zeros(self.dim, dtype=np.float32)
        # Unsigned counts: a hash collision can only add spurious similarity, never cancel
        # a genuine word match (which signed hashing can do with small dimensions).
        for feature in features:
            vec[self._bucket(feature)] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ExtractiveFakeChatModel(BaseChatModel):
    """A deterministic 'LLM' that answers by extracting the best-matching context sentence.

    It relies on the prompt layout produced by `documind.prompts.build_rag_prompt`:
    context blocks separated by `CONTEXT_SEPARATOR`, each starting with
    `Source: [file p.N]`, followed by a final `Question: ...` line.
    """

    min_overlap: float = 0.5
    """Fraction of the question's content words that must appear in a sentence to answer."""

    @property
    def _llm_type(self) -> str:
        return "extractive-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = str(messages[-1].content)
        answer = self._answer(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    def _answer(self, prompt: str) -> str:
        context, _, question = prompt.rpartition(QUESTION_PREFIX)
        question_tokens = set(content_tokens(question))
        if not question_tokens:
            return REFUSAL_MESSAGE

        best_score, best_sentence, best_citation = 0.0, "", ""
        for block in context.split(CONTEXT_SEPARATOR):
            header = _SOURCE_HEADER_RE.search(block)
            if header is None:
                continue
            citation = header.group(1)
            body = block[header.end() :]
            for sentence in split_sentences(body.replace("\n", " ")):
                overlap = len(question_tokens & set(content_tokens(sentence)))
                score = overlap / len(question_tokens)
                if score > best_score:
                    best_score, best_sentence, best_citation = score, sentence, citation

        if best_score < self.min_overlap:
            return REFUSAL_MESSAGE
        return f"{best_sentence} {best_citation}"
