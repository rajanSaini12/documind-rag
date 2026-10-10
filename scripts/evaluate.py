"""Compare embedding models and chunk sizes on a small Q&A dataset.

Examples:
    # Real models (needs sentence-transformers and OPENAI_API_KEY):
    python scripts/evaluate.py \\
        --embeddings huggingface:sentence-transformers/all-MiniLM-L6-v2 \\
                     huggingface:BAAI/bge-small-en-v1.5 \\
        --chunk-sizes 256 512 1024 --llm-provider openai --llm-model gpt-4o-mini

    # Fully offline smoke run (hashing embeddings + extractive fake LLM + lexical judge):
    python scripts/evaluate.py --offline

The judge is an LLM (LLM-as-judge) when OPENAI_API_KEY is set, unless --judge lexical is
given; otherwise the lexical-overlap fallback is used. The report header records exactly
which models were used, because the numbers mean nothing without that context.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from documind.eval.dataset import load_dataset
from documind.eval.judges import Judge, LexicalJudge, LLMJudge
from documind.eval.runner import EmbeddingSpec, run_grid, summarize, write_reports
from documind.providers import get_embeddings, get_llm

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_EMBEDDINGS = ["fake:hashing-256", "fake:hashing-1024"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", type=Path, default=ROOT / "eval" / "qa_dataset.jsonl")
    p.add_argument(
        "--docs", type=Path, nargs="+", default=[ROOT / "eval" / "sample_docs" / "handbook.pdf"]
    )
    p.add_argument(
        "--embeddings",
        nargs="+",
        default=["huggingface:sentence-transformers/all-MiniLM-L6-v2"],
        help="One or more provider:model specs (providers: huggingface, openai, fake).",
    )
    p.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--overlap-ratio", type=float, default=0.125, help="overlap = size * ratio")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--llm-provider", choices=["openai", "fake"], default="openai")
    p.add_argument("--llm-model", default="gpt-4o-mini")
    p.add_argument("--judge", choices=["auto", "llm", "lexical"], default="auto")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use hashing embeddings, the fake LLM and the lexical judge (no network).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") or None
    if args.offline:
        args.embeddings, args.llm_provider, args.judge = OFFLINE_EMBEDDINGS, "fake", "lexical"

    specs = [EmbeddingSpec.parse(s) for s in args.embeddings]
    llm = get_llm(args.llm_provider, args.llm_model, 0.0, api_key)

    judge_kind = args.judge
    if judge_kind == "auto":
        judge_kind = "llm" if api_key else "lexical"
    judge: Judge = (
        LLMJudge(get_llm("openai", args.judge_model, 0.0, api_key), name=args.judge_model)
        if judge_kind == "llm"
        else LexicalJudge()
    )

    examples = load_dataset(args.dataset)
    per_question = run_grid(
        pdf_paths=args.docs,
        examples=examples,
        embedding_specs=specs,
        make_embeddings=lambda s: get_embeddings(s.provider, s.model, api_key),
        llm=llm,
        judge=judge,
        chunk_sizes=args.chunk_sizes,
        top_k=args.top_k,
        overlap_ratio=args.overlap_ratio,
    )
    summary = summarize(per_question)

    llm_label = "fake extractive LLM" if args.llm_provider == "fake" else args.llm_model
    header = (
        "# DocuMind evaluation\n\n"
        f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"- Dataset: `{args.dataset.name}` ({len(examples)} questions, "
        f"{sum(not e.answerable for e in examples)} unanswerable)\n"
        f"- Documents: {', '.join(f'`{d.name}`' for d in args.docs)}\n"
        f"- Answer LLM: {llm_label}\n"
        f"- Judge: {judge.name}\n"
        f"- top_k={args.top_k}, chunk_overlap = chunk_size * {args.overlap_ratio}\n"
    )
    if args.offline:
        header += (
            "\n> Offline run with hashing embeddings, a fake extractive LLM and lexical "
            "metrics. Illustrative only: it shows the pipeline works, not real model quality.\n"
        )
    paths = write_reports(per_question, summary, args.out_dir, header)

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
