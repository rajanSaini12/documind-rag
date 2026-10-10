"""Grid evaluation over (embedding model x chunk size).

For every configuration we build a fresh, temporary Chroma collection, ingest the
documents, then for every dataset question we:
  1. retrieve top-k chunks  -> hit@k and reciprocal rank against the gold page(s)
  2. generate an answer     -> relevance, faithfulness, refusal behaviour
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from documind.config import make_collection_name
from documind.eval.dataset import QAExample
from documind.eval.judges import Judge
from documind.eval.metrics import hit_at_k, reciprocal_rank
from documind.rag import RAGPipeline
from documind.vectorstore import DocumentStore


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> EmbeddingSpec:
        """'huggingface:BAAI/bge-small-en-v1.5' -> EmbeddingSpec('huggingface', 'BAAI/...')."""
        provider, sep, model = spec.partition(":")
        if not sep or not model:
            raise ValueError(f"Expected 'provider:model', got {spec!r}")
        return cls(provider, model)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def evaluate_config(
    pdf_paths: Sequence[Path],
    examples: Sequence[QAExample],
    embeddings: Embeddings,
    embedding_name: str,
    llm: BaseChatModel,
    judge: Judge,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> list[dict]:
    """Evaluate one configuration and return one row per question."""
    rows = []
    with tempfile.TemporaryDirectory(prefix="documind-eval-") as tmp:
        store = DocumentStore(tmp, make_collection_name("eval", "grid", embedding_name), embeddings)
        for path in pdf_paths:
            store.add_pdf(path.read_bytes(), path.name, chunk_size, chunk_overlap)
        pipeline = RAGPipeline(store, llm)

        for ex in examples:
            hits = pipeline.retrieve(ex.question, top_k)
            retrieved = [(d.metadata["source"], int(d.metadata["page"])) for d, _ in hits]
            result = pipeline.generate(ex.question, hits)
            contexts = [d.page_content for d, _ in hits]
            gold = [(ex.source, p) for p in ex.gold_pages] if ex.answerable else []

            row: dict = {
                "embedding": embedding_name,
                "chunk_size": chunk_size,
                "question_id": ex.id,
                "answerable": ex.answerable,
                "question": ex.question,
                "answer": result.answer,
                "refused": result.refused,
                "retrieved_pages": ";".join(f"{s} p.{p}" for s, p in retrieved),
                # NaN = "not applicable" so pandas' mean() skips it.
                "hit_at_k": hit_at_k(retrieved, gold) if ex.answerable else math.nan,
                "reciprocal_rank": reciprocal_rank(retrieved, gold) if ex.answerable else math.nan,
                "relevance": math.nan,
                "faithfulness": math.nan,
            }
            if ex.answerable:
                # A refusal to an answerable question is simply an irrelevant answer.
                row["relevance"] = (
                    0.0
                    if result.refused
                    else judge.relevance(ex.question, result.answer, ex.reference_answer or "")
                )
            if not result.refused:
                # Refusals make no claims, so faithfulness is only scored for real answers.
                row["faithfulness"] = judge.faithfulness(result.answer, contexts)
            rows.append(row)
    return rows


def run_grid(
    pdf_paths: Sequence[Path],
    examples: Sequence[QAExample],
    embedding_specs: Sequence[EmbeddingSpec],
    make_embeddings: Callable[[EmbeddingSpec], Embeddings],
    llm: BaseChatModel,
    judge: Judge,
    chunk_sizes: Sequence[int],
    top_k: int = 4,
    overlap_ratio: float = 0.125,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Run every (embedding, chunk size) combination; returns the per-question results."""
    rows: list[dict] = []
    for spec in embedding_specs:
        embeddings = make_embeddings(spec)  # load each model once, reuse across chunk sizes
        for chunk_size in chunk_sizes:
            progress(f"Evaluating embedding={spec} chunk_size={chunk_size} ...")
            rows += evaluate_config(
                pdf_paths,
                examples,
                embeddings,
                str(spec),
                llm,
                judge,
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * overlap_ratio),
                top_k=top_k,
            )
    return pd.DataFrame(rows)


def summarize(per_question: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-question rows into one row per configuration."""

    def agg(group: pd.DataFrame) -> pd.Series:
        answerable = group[group["answerable"]]
        unanswerable = group[~group["answerable"]]
        return pd.Series(
            {
                "hit_rate@k": answerable["hit_at_k"].mean(),
                "mrr": answerable["reciprocal_rank"].mean(),
                "relevance": answerable["relevance"].mean(),
                "faithfulness": group["faithfulness"].mean(),  # mean over non-refused answers
                "false_refusal_rate": answerable["refused"].mean(),
                "correct_refusal_rate": unanswerable["refused"].mean(),
                "n_questions": len(group),
            }
        )

    summary = (
        per_question.groupby(["embedding", "chunk_size"], sort=False)[per_question.columns.tolist()]
        .apply(agg)
        .reset_index()
    )
    summary["n_questions"] = summary["n_questions"].astype(int)
    return summary


def write_reports(
    per_question: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, header: str
) -> dict[str, Path]:
    """Write per-question CSV, summary CSV and a markdown report. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_question_csv": out_dir / "per_question.csv",
        "summary_csv": out_dir / "summary.csv",
        "summary_md": out_dir / "summary.md",
    }
    per_question.to_csv(paths["per_question_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False, float_format="%.3f")
    table = summary.to_markdown(index=False, floatfmt=".3f")
    paths["summary_md"].write_text(f"{header}\n\n{table}\n", encoding="utf-8")
    return paths
