from dataclasses import asdict
from types import SimpleNamespace

import core.generator as generator
from core.answer import AnswerResult, Source, parse_citations


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
