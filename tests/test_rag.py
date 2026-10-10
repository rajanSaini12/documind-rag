from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from documind.prompts import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    build_rag_prompt,
    extract_citations,
    format_context,
    is_refusal,
)
from documind.rag import RAGPipeline


def test_system_prompt_enforces_grounding_citations_and_refusal():
    prompt = SYSTEM_PROMPT.lower()
    assert "only" in prompt and "context" in prompt
    assert "[file p.n]" in prompt
    assert REFUSAL_MESSAGE.lower() in prompt


def test_rendered_prompt_contains_context_headers_and_question():
    docs = [Document(page_content="Leave is 24 days.", metadata={"source": "hb.pdf", "page": 3})]
    messages = build_rag_prompt().format_messages(
        context=format_context(docs), question="How much leave?"
    )
    assert messages[0].type == "system"
    human = messages[1].content
    assert "Source: [hb.pdf p.3]" in human
    assert "Leave is 24 days." in human
    assert human.rstrip().endswith("Question: How much leave?")


def test_extract_citations_handles_spaces_in_file_names():
    answer = "It is 24 days [Employee Handbook.pdf p.3] and 5 carry over [hb.pdf p.12]."
    assert extract_citations(answer) == {("Employee Handbook.pdf", 3), ("hb.pdf", 12)}


def test_is_refusal_tolerates_curly_quotes():
    assert is_refusal("I don’t know based on the documents.")
    assert not is_refusal("You get 24 days.")


def test_answerable_question_is_answered_with_citation(pipeline):
    result = pipeline.ask("How many days of paid annual leave do full-time employees get?", 4)

    assert not result.refused
    assert "24 days" in result.answer
    assert ("handbook.pdf", 3) in extract_citations(result.answer)
    cited = [c for c in result.citations if c.cited]
    assert cited and all(c.page == 3 for c in cited)
    assert all(c.snippet and 0.0 <= c.score <= 1.0 for c in result.citations)


def test_unanswerable_question_is_refused(pipeline):
    result = pipeline.ask("Are employees allowed to bring pets to the office?", 4)
    assert result.refused
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


def test_empty_store_refuses_without_calling_llm(store):
    llm = FakeListChatModel(responses=["this must not be used"])
    result = RAGPipeline(store, llm).ask("Anything?", 4)
    assert result.refused
    assert result.answer == REFUSAL_MESSAGE


def test_llm_refusal_is_detected_and_citations_dropped(loaded_store):
    llm = FakeListChatModel(responses=["I don't know based on the documents."])
    result = RAGPipeline(loaded_store, llm).ask("What is the CEO's salary?", 4)
    assert result.refused
    assert result.citations == []


def test_min_score_filters_weak_matches(loaded_store):
    strict = RAGPipeline(loaded_store, FakeListChatModel(responses=["unused"]), min_score=0.99)
    assert strict.retrieve("password length", top_k=4) == []
