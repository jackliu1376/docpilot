import core.generator as generator


def test_call_cloud_api_dispatches_deepseek(monkeypatch):
    calls = []
    monkeypatch.setattr(
        generator,
        "_call_openai_compatible_api",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "ok",
    )

    assert generator.call_cloud_api("prompt", "deepseek") == "ok"

    args, _ = calls[0]
    assert args[0] == "DeepSeek"
    assert args[2] == generator.DEEPSEEK_API_URL
    assert args[3] == generator.DEEPSEEK_MODEL_NAME


def test_call_cloud_api_dispatches_glm(monkeypatch):
    calls = []
    monkeypatch.setattr(
        generator,
        "_call_openai_compatible_api",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "ok",
    )

    assert generator.call_cloud_api("prompt", "glm") == "ok"

    args, _ = calls[0]
    assert args[0] == "GLM"
    assert args[2] == generator.GLM_API_URL
    assert args[3] == generator.GLM_MODEL_NAME


def test_extract_content_handles_reasoning_field():
    result = {
        "choices": [
            {
                "message": {
                    "content": "答案",
                    "reasoning": "推理",
                }
            }
        ]
    }

    assert (
        generator._extract_openai_compatible_content(result)
        == "答案<think>推理</think>"
    )
