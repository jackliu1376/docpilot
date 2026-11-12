from fastapi.testclient import TestClient

import api_router
from core.answer import AnswerResult, Source
from tests.helpers import FakeFile, make_kb


def test_documents_endpoint_lists_persisted_documents(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["filename"] == "notes.txt"


def test_upload_persists_document(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"abcdefghij", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["file_info"]["chunks"] == 1
    assert kb.stats().documents == 1


def test_delete_document_endpoint(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.delete("/api/documents/notes.txt")

    assert response.status_code == 200
    assert kb.stats().documents == 0


def test_clear_endpoint(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.post("/api/clear")

    assert response.status_code == 200
    assert kb.list_documents() == []


class FailingKnowledgeBase:
    def load(self):
        pass

    def add_files(self, files):
        raise RuntimeError("C:/secret/temp/path")


def test_upload_rejects_invalid_filename(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("..", b"data", "text/plain")},
        )

    assert response.status_code == 400
    assert "无效" in response.json()["detail"]


def test_upload_failure_does_not_leak_temp_path(monkeypatch):
    monkeypatch.setattr(
        api_router, "knowledge_base", FailingKnowledgeBase(), raising=False
    )

    with TestClient(api_router.app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"data", "text/plain")},
        )

    assert response.status_code == 500
    assert "secret" not in response.text
    assert "文档处理失败" in response.json()["detail"]


def test_ask_returns_structured_sources(monkeypatch):
    result = AnswerResult(
        answer="答案 [S1]",
        sources=[
            Source(
                id="S1",
                kind="local",
                snippet="text",
                filename="notes.txt",
                chunk_id="doc:1",
                chunk_index=0,
            )
        ],
        metadata={"model": "ollama"},
    )
    monkeypatch.setattr(
        api_router, "answer_question", lambda *args, **kwargs: result
    )

    with TestClient(api_router.app) as client:
        response = client.post(
            "/api/ask",
            json={"question": "q", "model_choice": "ollama"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "答案 [S1]"
    assert body["sources"][0]["id"] == "S1"
    assert body["sources"][0]["filename"] == "notes.txt"
    assert body["metadata"]["model"] == "ollama"
