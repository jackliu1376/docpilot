import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.bm25_index import bm25_manager
from core.vector_store import vector_store
from tests.helpers import FakeFile, make_kb


def test_new_knowledge_base_creates_empty_schema(tmp_path):
    kb = make_kb(tmp_path)

    assert kb.list_documents() == []
    stats = kb.stats()
    assert stats.documents == 0
    assert stats.chunks == 0


def test_add_files_persists_document_and_chunks(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    kb = make_kb(tmp_path)

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "added"
    assert results[0].chunks == 3
    assert kb.stats().documents == 1
    assert kb.stats().chunks == 3
    assert vector_store.total_chunks == 3
    assert bm25_manager.bm25_index is not None


def test_add_same_file_twice_is_unchanged(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "unchanged"
    assert kb.stats().documents == 1
    assert kb.stats().chunks == 3


def test_changed_file_replaces_old_chunks(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])
    path.write_text("0123456789", encoding="utf-8")

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "replaced"
    assert kb.stats().documents == 1
    assert kb.stats().chunks == 1


def test_empty_document_is_reported_and_not_persisted(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    kb = make_kb(tmp_path)

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "error"
    assert kb.stats().documents == 0
    assert kb.stats().chunks == 0


def test_concurrent_adds_do_not_lose_documents(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("aaaaaaaaaa", encoding="utf-8")
    second.write_text("bbbbbbbbbb", encoding="utf-8")
    kb = make_kb(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda p: kb.add_files([FakeFile(p)]), [first, second]))

    assert kb.stats().documents == 2
    assert kb.stats().chunks == 2


def test_load_rebuilds_indexes_from_sqlite(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    make_kb(tmp_path).add_files([FakeFile(path)])

    stats = make_kb(tmp_path).load()

    assert stats.documents == 1
    assert stats.chunks == 3
    assert vector_store.total_chunks == 3
    assert bm25_manager.bm25_index is not None


def test_delete_document_removes_chunks_and_updates_indexes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    assert kb.delete_document("a.txt") is True
    assert kb.stats().documents == 0
    assert kb.stats().chunks == 0
    assert vector_store.total_chunks == 0


def test_clear_removes_all_documents_and_indexes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    kb.clear()

    assert kb.list_documents() == []
    assert vector_store.total_chunks == 0


def test_load_reembeds_when_embedding_model_changes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    first = make_kb(
        tmp_path,
        embed_model_name="model-a",
        embedder=lambda texts, show_progress=False: np.array(
            [[1.0, 0.0] for _ in texts], dtype="float32"
        ),
    )
    first.add_files([FakeFile(path)])

    second = make_kb(
        tmp_path,
        embed_model_name="model-b",
        embedder=lambda texts, show_progress=False: np.array(
            [[2.0, 0.0] for _ in texts], dtype="float32"
        ),
    )
    stats = second.load()

    assert stats.embedding_model == "model-b"
    with sqlite3.connect(tmp_path / "rag.db") as conn:
        blob = conn.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()[0]
    assert np.frombuffer(blob, dtype="float32")[0] == 2.0


def test_changed_file_duplicating_another_document_replaces_stale_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("AAAAAAAAAA", encoding="utf-8")
    b.write_text("BBBBBBBBBB", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(a), FakeFile(b)])
    a.write_text("BBBBBBBBBB", encoding="utf-8")

    results = kb.add_files([FakeFile(a)])

    assert results[0].status == "replaced"
    assert kb.stats().documents == 2
    with sqlite3.connect(tmp_path / "rag.db") as conn:
        content = conn.execute(
            "SELECT c.content FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE d.filename = 'a.txt'"
        ).fetchone()[0]
    assert content == "BBBBBBBBBB"


def test_load_repairs_mixed_embedding_dimensions(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrst", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    with sqlite3.connect(tmp_path / "rag.db") as conn:
        chunk_id = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
        conn.execute(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            (np.array([1.0, 2.0, 3.0], dtype="float32").tobytes(), chunk_id),
        )

    stats = kb.load()

    assert stats.chunks == 2
    assert vector_store.total_chunks == 2
    with sqlite3.connect(tmp_path / "rag.db") as conn:
        sizes = {
            len(row[0]) for row in conn.execute("SELECT embedding FROM chunks")
        }
    assert sizes == {8}


def test_add_files_accepts_path_objects(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb = make_kb(tmp_path)

    results = kb.add_files([path])

    assert results[0].status == "added"
    assert results[0].chunks == 1


def test_two_instances_can_add_documents_to_same_db(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("AAAAAAAAAA", encoding="utf-8")
    b.write_text("BBBBBBBBBB", encoding="utf-8")
    kb1 = make_kb(tmp_path)
    kb2 = make_kb(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(kb1.add_files, [FakeFile(a)]),
            pool.submit(kb2.add_files, [FakeFile(b)]),
        ]
        results = [future.result() for future in futures]

    assert all(result[0].status == "added" for result in results)
    assert kb1.stats().documents == 2
    assert kb1.stats().chunks == 2


def test_index_rebuild_is_serialized_across_instances(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb1 = make_kb(tmp_path)
    kb1.add_files([FakeFile(path)])
    kb2 = make_kb(tmp_path)

    active = {"count": 0, "max": 0}
    guard = threading.Lock()
    original_build = vector_store.build_index

    def guarded_build(*args, **kwargs):
        with guard:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        try:
            return original_build(*args, **kwargs)
        finally:
            with guard:
                active["count"] -= 1

    monkeypatch.setattr(vector_store, "build_index", guarded_build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda kb: kb.load(), [kb1, kb2]))

    assert active["max"] == 1


def test_unchanged_only_batch_does_not_reload(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    calls = {"count": 0}
    original_load = kb.load

    def counted_load():
        calls["count"] += 1
        return original_load()

    monkeypatch.setattr(kb, "load", counted_load)
    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "unchanged"
    assert calls["count"] == 0


def test_delete_missing_document_does_not_reload(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    calls = {"count": 0}
    original_load = kb.load

    def counted_load():
        calls["count"] += 1
        return original_load()

    monkeypatch.setattr(kb, "load", counted_load)

    assert kb.delete_document("missing.txt") is False
    assert calls["count"] == 0
