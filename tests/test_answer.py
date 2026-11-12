from dataclasses import asdict
from types import SimpleNamespace

import core.generator as generator
from core.answer import (
    AnswerResult,
    Source,
    normalize_citations,
    parse_citations,
)


def test_parse_citations_extracts_unique_source_ids():
    assert parse_citations("答案 [S1] 和 [S3]，重复 [S1]") == {"S1", "S3"}


def test_source_serializes_to_dict():
    source = Source(
        id="S1",
        kind="local",
        snippet="text",
        filename="a.pdf",
        chunk_id="doc:1",
        chunk_index=0,
    )

    assert asdict(source)["filename"] == "a.pdf"
    assert asdict(source)["chunk_id"] == "doc:1"


def test_normalize_citations_handles_plain_and_prefixed_ids():
    text = "根据S1，[来源：S2]，（S3），以及 [S9]。"

    normalized = normalize_citations(text, {"S1", "S2", "S3"})

    assert "[S1]" in normalized
    assert "[S2]" in normalized
    assert "[S3]" in normalized
    assert "S9" not in normalized
    assert "来源" not in normalized


def test_build_prompt_instructs_concise_citations_and_refusal():
    prompt = generator._build_prompt(
        "问题", "参考内容", False, True, False, False
    )

    assert "先给出 1-3 句结论" in prompt
    assert "根据提供的资料无法确认" in prompt
    assert "[S1]" in prompt
    assert "示例" in prompt
    assert "只输出最终回答" in prompt
    assert "关键数值" in prompt


def test_answer_question_normalizes_citations(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "根据S1，需要更换喷油器。"}

    class FakeSession:
        def post(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(generator, "get_session", lambda: FakeSession())
    monkeypatch.setattr(generator, "vector_store", SimpleNamespace(is_ready=True))
    monkeypatch.setattr(
        generator,
        "recursive_retrieval",
        lambda **kwargs: (
            ["chunk text"],
            ["doc:1"],
            [{"source": "a.pdf", "chunk_index": 0}],
        ),
    )

    result = generator.answer_question("问题", model_choice="ollama")

    assert "[S1]" in result.answer
    assert "根据S1" not in result.answer


def test_build_context_labels_sources():
    context, sources = generator._build_context(
        ["chunk text"],
        ["doc:1"],
        [{"source": "a.pdf", "chunk_index": 0}],
        False,
    )

    assert "[S1]" in context
    assert sources == [
        Source(
            id="S1",
            kind="local",
            snippet="chunk text",
            filename="a.pdf",
            chunk_id="doc:1",
            chunk_index=0,
        )
    ]


def test_answer_question_returns_structured_sources(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "答案 [S1]"}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return FakeResponse()

    session = FakeSession()
    monkeypatch.setattr(generator, "get_session", lambda: session)
    monkeypatch.setattr(generator, "vector_store", SimpleNamespace(is_ready=True))
    monkeypatch.setattr(
        generator,
        "recursive_retrieval",
        lambda **kwargs: (
            ["chunk text"],
            ["doc:1"],
            [{"source": "a.pdf", "chunk_index": 0}],
        ),
    )

    result = generator.answer_question("问题", model_choice="ollama")

    assert isinstance(result, AnswerResult)
    assert result.answer == "答案 [S1]"
    assert result.sources == [
        Source(
            id="S1",
            kind="local",
            snippet="chunk text",
            filename="a.pdf",
            chunk_id="doc:1",
            chunk_index=0,
        )
    ]
