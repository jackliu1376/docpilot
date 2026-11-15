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
    assert "不要复用示例" in prompt
    assert "油品问题" not in prompt
    assert "缸压不足" not in prompt
    assert "只输出最终回答" in prompt
    assert "关键数值" in prompt


def test_build_prompt_includes_history_as_non_evidence():
    prompt = generator._build_prompt(
        "它的寿命呢？",
        "参考内容",
        False,
        True,
        False,
        False,
        history_text="用户：量子计算机是什么？\n助手：量子计算机利用量子力学。",
    )

    assert "对话历史" in prompt
    assert "不能作为回答依据" in prompt
    assert "它的寿命呢？" in prompt
    assert "量子计算机是什么" in prompt


def test_rewrite_followup_query_uses_history():
    captured = {}

    def fake_call(prompt, model_choice, max_tokens=64):
        captured["prompt"] = prompt
        captured["model_choice"] = model_choice
        captured["max_tokens"] = max_tokens
        return "量子计算机的使用寿命"

    rewritten = generator.rewrite_followup_query(
        "它的寿命呢？",
        [
            {"role": "user", "content": "量子计算机是什么？"},
            {"role": "assistant", "content": "量子计算机利用量子力学。"},
        ],
        "deepseek",
        call_fn=fake_call,
    )

    assert rewritten == "量子计算机的使用寿命"
    assert captured["model_choice"] == "deepseek"
    assert captured["max_tokens"] == 64
    assert "量子计算机是什么" in captured["prompt"]
    assert "它的寿命呢？" in captured["prompt"]


def test_rewrite_followup_query_falls_back_without_history_or_valid_output():
    def fail_call(*args, **kwargs):
        raise AssertionError("no LLM call without history")

    assert generator.rewrite_followup_query(
        "量子计算机是什么？", [], "deepseek", call_fn=fail_call
    ) == "量子计算机是什么？"

    assert generator.rewrite_followup_query(
        "它的寿命呢？",
        [{"role": "user", "content": "量子计算机是什么？"}],
        "deepseek",
        call_fn=lambda *args, **kwargs: "",
    ) == "它的寿命呢？"


def test_rewrite_followup_query_skips_complete_standalone_question():
    calls = []

    def fail_call(*args, **kwargs):
        calls.append(1)
        raise AssertionError("standalone question should not be rewritten")

    question = "请详细说明量子计算机的使用寿命、主要风险和适用场景"
    assert generator.rewrite_followup_query(
        question,
        [{"role": "user", "content": "量子计算机是什么？"}],
        "deepseek",
        call_fn=fail_call,
    ) == question
    assert calls == []


def test_answer_question_passes_rewritten_query_to_retrieval(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "答案 [S1]"}

    class FakeSession:
        def post(self, url, **kwargs):
            return FakeResponse()

    captured = {}

    def fake_retrieval(**kwargs):
        captured.update(kwargs)
        return (
            ["chunk text"],
            ["doc:1"],
            [{"source": "a.pdf", "chunk_index": 0}],
        )

    monkeypatch.setattr(generator, "get_session", lambda: FakeSession())
    monkeypatch.setattr(generator, "vector_store", SimpleNamespace(is_ready=True))
    monkeypatch.setattr(generator, "recursive_retrieval", fake_retrieval)
    monkeypatch.setattr(
        generator,
        "rewrite_followup_query",
        lambda question, history, model_choice: "量子计算机的使用寿命",
    )

    result = generator.answer_question(
        "它的寿命呢？",
        model_choice="ollama",
        history=[
            {"role": "user", "content": "量子计算机是什么？"},
            {"role": "assistant", "content": "量子计算机利用量子力学。"},
        ],
    )

    assert captured["initial_query"] == "量子计算机的使用寿命"
    assert result.metadata["retrieval_query"] == "量子计算机的使用寿命"
    assert result.metadata["history_turns"] == 1


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
