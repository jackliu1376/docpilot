from core.answer import Source
from eval.answer_judge import (
    build_judge_prompt,
    choose_judge_model,
    fallback_judge_model,
    judge_answer,
    parse_judge_response,
)


def test_parse_judge_response_handles_markdown_fence():
    text = """```json
{"correct": true, "correctness_score": 5, "faithful": true,
 "unsupported_claims": [], "citation_support": [{"citation": "S1", "supported": true}],
 "reason": "ok"}
```"""

    data = parse_judge_response(text)

    assert data["correct"] is True
    assert data["correctness_score"] == 5.0
    assert data["faithful"] is True
    assert data["citation_support"][0]["supported"] is True


def test_build_judge_prompt_includes_context_answer_and_reference():
    prompt = build_judge_prompt(
        "问题",
        "参考答案",
        "候选答案 [S1]",
        [Source(id="S1", kind="local", snippet="证据片段")],
    )

    assert "参考答案" in prompt
    assert "候选答案" in prompt
    assert "[S1]" in prompt
    assert "证据片段" in prompt


def test_judge_answer_uses_injected_model_call():
    calls = []

    def fake_call(prompt, model_choice, **kwargs):
        calls.append((prompt, model_choice, kwargs))
        return (
            '{"correct": true, "correctness_score": 4, "faithful": false, '
            '"unsupported_claims": ["x"], "citation_support": [], '
            '"reason": "r"}'
        )

    result = judge_answer(
        "q", "ref", "ans", [], judge_model="deepseek", call_fn=fake_call
    )

    assert result["correct"] is True
    assert result["faithful"] is False
    assert result["unsupported_claims"] == ["x"]
    assert calls[0][1] == "deepseek"
    assert calls[0][2]["temperature"] == 0.0
    assert calls[0][2]["max_tokens"] >= 2000


def test_judge_answer_falls_back_when_primary_is_not_json():
    calls = []

    def fake_call(prompt, model_choice, **kwargs):
        calls.append(model_choice)
        if model_choice == "deepseek":
            return "not json at all"
        return (
            '{"correct": true, "correctness_score": 5, "faithful": true, '
            '"unsupported_claims": [], "citation_support": [], "reason": "ok"}'
        )

    result = judge_answer(
        "q", "ref", "ans", [],
        judge_model="deepseek",
        fallback_model="glm",
        call_fn=fake_call,
    )

    assert calls == ["deepseek", "glm"]
    assert result["judge_model"] == "glm"
    assert result["judge_fallback_used"] is True
    assert result["correct"] is True


def test_choose_judge_model_avoids_self_judging():
    assert choose_judge_model("deepseek") == "glm"
    assert choose_judge_model("glm") == "deepseek"
    assert choose_judge_model("ollama") == "deepseek"
    assert choose_judge_model("deepseek", "deepseek") == "deepseek"


def test_fallback_judge_model_uses_the_other_provider():
    assert fallback_judge_model("deepseek") == "glm"
    assert fallback_judge_model("glm") == "deepseek"
