from __future__ import annotations

import math

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from documind.eval.dataset import QAExample, load_dataset
from documind.eval.judges import LexicalJudge, LLMJudge, parse_score
from documind.eval.metrics import (
    hit_at_k,
    lexical_faithfulness,
    reciprocal_rank,
    token_f1,
)
from documind.eval.runner import EmbeddingSpec, run_grid, summarize, write_reports
from documind.fakes import ExtractiveFakeChatModel, HashingEmbeddings
from tests.conftest import ROOT, SAMPLE_PDF

GOLD = [("hb.pdf", 3)]


def test_hit_at_k():
    assert hit_at_k([("hb.pdf", 1), ("hb.pdf", 3)], GOLD) == 1.0
    assert hit_at_k([("hb.pdf", 1), ("other.pdf", 3)], GOLD) == 0.0
    assert hit_at_k([], GOLD) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([("hb.pdf", 3)], GOLD) == 1.0
    assert reciprocal_rank([("hb.pdf", 1), ("hb.pdf", 2), ("hb.pdf", 3)], GOLD) == pytest.approx(
        1 / 3
    )
    assert reciprocal_rank([("hb.pdf", 1)], GOLD) == 0.0


def test_token_f1():
    assert token_f1("24 days of leave", "24 days of leave") == pytest.approx(1.0)
    assert token_f1("completely unrelated words", "24 days of leave") == 0.0
    assert 0.0 < token_f1("You get 24 days.", "Employees get 24 days of paid leave.") < 1.0
    # Citation markers must not count as answer words.
    assert token_f1("24 days [hb.pdf p.3]", "24 days") == pytest.approx(1.0)


def test_lexical_faithfulness_flags_unsupported_sentences():
    context = ["Employees receive 24 days of paid annual leave."]
    assert (
        lexical_faithfulness("Employees receive 24 days of paid leave. [hb.pdf p.3]", context) == 1
    )
    mixed = "Employees receive 24 days of leave. Unicorns deliver payroll every Tuesday."
    assert lexical_faithfulness(mixed, context) == pytest.approx(0.5)


def test_parse_score_tolerates_surrounding_text_and_clamps():
    assert parse_score('Sure! {"score": 0.75, "reason": "ok"}') == 0.75
    assert parse_score('{"score": 3}') == 1.0
    with pytest.raises(ValueError):
        parse_score("no json here")


def test_llm_judge_uses_model_reply():
    llm = FakeListChatModel(responses=['{"score": 0.9, "reason": "supported"}'] * 2)
    judge = LLMJudge(llm)
    assert judge.faithfulness("answer", ["context"]) == 0.9
    assert judge.relevance("q", "a", "ref") == 0.9


def test_dataset_validation():
    examples = load_dataset(ROOT / "eval" / "qa_dataset.jsonl")
    assert len(examples) >= 10
    assert sum(not e.answerable for e in examples) >= 2
    with pytest.raises(ValueError):
        QAExample(id="bad", question="?", answerable=True)  # missing gold fields


def test_embedding_spec_parse():
    spec = EmbeddingSpec.parse("huggingface:BAAI/bge-small-en-v1.5")
    assert (spec.provider, spec.model) == ("huggingface", "BAAI/bge-small-en-v1.5")
    with pytest.raises(ValueError):
        EmbeddingSpec.parse("no-provider")


def test_run_grid_end_to_end_offline(tmp_path):
    examples = load_dataset(ROOT / "eval" / "qa_dataset.jsonl")
    per_question = run_grid(
        pdf_paths=[SAMPLE_PDF],
        examples=examples,
        embedding_specs=[EmbeddingSpec("fake", "hashing-256")],
        make_embeddings=lambda spec: HashingEmbeddings(dim=256),
        llm=ExtractiveFakeChatModel(),
        judge=LexicalJudge(),
        chunk_sizes=[256, 512],
        progress=lambda _: None,
    )
    assert len(per_question) == 2 * len(examples)

    summary = summarize(per_question)
    assert list(summary["chunk_size"]) == [256, 512]
    for col in ["hit_rate@k", "mrr", "relevance", "correct_refusal_rate"]:
        assert summary[col].between(0, 1).all()
    # The fake LLM refuses every unanswerable question in the sample dataset.
    assert (summary["correct_refusal_rate"] == 1.0).all()
    assert not math.isnan(summary["faithfulness"].iloc[0])

    paths = write_reports(per_question, summary, tmp_path, "# test")
    assert all(p.exists() for p in paths.values())
    assert "| embedding" in paths["summary_md"].read_text()
