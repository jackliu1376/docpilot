import inspect
import os
import subprocess
import sys
from pathlib import Path

import config
from core import generator, retriever


def test_api_key_validation_rejects_placeholders():
    assert config.is_configured_api_key(None) is False
    assert config.is_configured_api_key("") is False
    assert config.is_configured_api_key("Your_SILICONFLOW_API_KEY") is False
    assert config.is_configured_api_key("sk-real-value") is True


def test_default_model_selection_order():
    assert config.choose_default_model("sk-deepseek", "sk-glm", True) == "deepseek"
    assert config.choose_default_model(None, "sk-glm", True) == "glm"
    assert config.choose_default_model(None, None, True) == "ollama"
    assert config.choose_default_model(None, None, False) == "deepseek"


def test_public_entry_points_use_configured_default_model():
    for function in (
        generator.answer_question,
        generator.query_answer,
        generator.stream_answer,
        generator.call_llm_simple,
    ):
        assert (
            inspect.signature(function).parameters["model_choice"].default
            == config.DEFAULT_MODEL_CHOICE
        )

    assert (
        inspect.signature(retriever.recursive_retrieval)
        .parameters["model_choice"]
        .default
        == config.DEFAULT_MODEL_CHOICE
    )


def test_hf_endpoint_does_not_override_existing_value():
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://example.test"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, config; print(os.environ['HF_ENDPOINT'])",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "https://example.test"
