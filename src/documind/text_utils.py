"""Tiny text helpers shared by the offline embeddings, the fake LLM and the lexical metrics."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# A deliberately small stopword list: enough to stop "the"/"is" dominating overlap scores.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have how
    i if in into is it its may me might must my no not of on or our should so than that
    the their them then there these they this to was we were what when where which who
    whom why will with would you your about after before any all also more most such
    """.split()
)


def _normalize(token: str) -> str:
    """Very light stemming: strip a plural 's' so 'days' matches 'day'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens (letters and digits only)."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed and light normalisation applied."""
    return [_normalize(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and excess blank lines that PDF extraction produces."""
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
