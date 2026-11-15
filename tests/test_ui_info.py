from types import SimpleNamespace

from config import RERANK_MODEL_NAME

import rag_demo


def test_system_models_info_uses_configured_reranker():
    info = rag_demo.get_system_models_info()

    assert RERANK_MODEL_NAME in info["重排序模型"]


def test_check_environment_does_not_call_cloud_api(monkeypatch):
    calls = []
    monkeypatch.setattr(rag_demo, "is_configured_api_key", lambda key: True)
    monkeypatch.setattr(
        rag_demo,
        "call_deepseek_api",
        lambda *args, **kwargs: calls.append(1) or "ok",
        raising=False,
    )

    assert rag_demo.check_environment() is True
    assert calls == []


def test_process_chat_forwards_history(monkeypatch):
    captured = {}

    def fake_answer(question, enable_web_search, model_choice, history=None):
        captured["history"] = list(history) if history else history
        return SimpleNamespace(answer="答案", sources=[])

    monkeypatch.setattr(rag_demo, "answer_question", fake_answer)

    history = [
        {"role": "user", "content": "量子计算机是什么？"},
        {"role": "assistant", "content": "量子计算机利用量子力学。"},
    ]
    updated, _, _ = rag_demo.process_chat("它的寿命呢？", history, False, "ollama")

    assert captured["history"] == [
        {"role": "user", "content": "量子计算机是什么？"},
        {"role": "assistant", "content": "量子计算机利用量子力学。"},
    ]
    assert updated[-2] == {"role": "user", "content": "它的寿命呢？"}
    assert updated[-1] == {"role": "assistant", "content": "答案"}
