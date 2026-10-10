"""Metric functions. All are pure functions so they are easy to unit test.

Retrieval metrics compare retrieved (source, page) pairs against the gold page(s).
The lexical answer metrics are cheap, offline fallbacks for the LLM judge: they are
crude (they reward word overlap, not meaning) and are reported as such.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from documind.prompts import strip_citations
from documind.text_utils import content_tokens, split_sentences

Page = tuple[str, int]  # (source file name, 1-based page number)


# ------------------------------------------------------------------ retrieval metrics
def hit_at_k(retrieved: Sequence[Page], gold: Iterable[Page]) -> float:
    """1.0 if any gold page appears among the retrieved chunks, else 0.0."""
    gold_set = set(gold)
    return float(any(page in gold_set for page in retrieved))


def reciprocal_rank(retrieved: Sequence[Page], gold: Iterable[Page]) -> float:
    """1 / rank of the first retrieved chunk from a gold page (0.0 if none).

    Averaged over questions this is MRR: 1.0 means the right page is always ranked first.
    """
    gold_set = set(gold)
    for rank, page in enumerate(retrieved, start=1):
        if page in gold_set:
            return 1.0 / rank
    return 0.0


# ------------------------------------------------------------- lexical answer metrics
def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between two texts (stopwords removed, citations ignored)."""
    pred = content_tokens(strip_citations(prediction))
    ref = content_tokens(reference)
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def lexical_relevance(question: str, answer: str, reference: str) -> float:
    """Fallback relevance: how well the answer's words match the human reference answer."""
    return token_f1(answer, reference)


def lexical_faithfulness(answer: str, contexts: Sequence[str], threshold: float = 0.8) -> float:
    """Fallback faithfulness: share of answer sentences that are 'supported' by the context.

    A sentence counts as supported if at least `threshold` of its content words occur
    somewhere in the retrieved context. Unsupported words are a cheap hallucination signal.
    """
    context_vocab = set(content_tokens(" ".join(contexts)))
    sentences = [s for s in split_sentences(strip_citations(answer)) if content_tokens(s)]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        tokens = content_tokens(sentence)
        coverage = sum(t in context_vocab for t in tokens) / len(tokens)
        supported += coverage >= threshold
    return supported / len(sentences)


def mean_or_nan(values: Iterable[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")
