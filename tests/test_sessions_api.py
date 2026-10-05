"""Session store + HTTP API. The RAG core is mocked so tests need no models or keys."""

import pytest
from fastapi.testclient import TestClient

from turbo_rag.sessions import SessionStore


def test_store_appends_in_order(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    assert store.append("s1", "user", "hi") == 0
    assert store.append("s1", "assistant", "hello", {"grounded": True}) == 1
    store.append("s2", "user", "other session")
    history = store.history("s1")
    assert [(m["turn"], m["role"], m["content"]) for m in history] == [(0, "user", "hi"), (1, "assistant", "hello")]
    assert history[1]["meta"] == {"grounded": True}


def test_store_last_n_and_delete(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    for i in range(5):
        store.append("s", "user", str(i))
    assert [m["content"] for m in store.history("s", last_n=2)] == ["3", "4"]
    assert store.history("s", last_n=0) == []
    store.delete("s")
    assert not store.exists("s")


def test_store_persists_across_connections(tmp_path):
    SessionStore(tmp_path / "s.db").append("s", "user", "remember me")
    assert SessionStore(tmp_path / "s.db").history("s")[0]["content"] == "remember me"


@pytest.fixture
def client(tmp_path, monkeypatch):
    from turbo_rag import api, pipeline
    from turbo_rag.chunking import Chunk
    from turbo_rag.generate import Answer, Citation
    from turbo_rag.retrieval.retriever import Hit

    chunk = Chunk("b1:0000", "b1", 1, 1, "Title: Gilead", {"title": "Gilead", "authors": "Marilynne Robinson"})
    seen = {}

    def fake_condense(question, history, model):
        seen.setdefault("histories", []).append(len(history))
        return question if not history else f"{question} (about Gilead)"

    def fake_rag_answer(question, mode=None):
        return Answer("Gilead is by Marilynne Robinson [1].", [Citation(1, "b1:0000", "Gilead")]), [Hit(chunk, 0.9)]

    monkeypatch.setattr(pipeline, "condense_question", fake_condense)
    monkeypatch.setattr(pipeline, "rag_answer", fake_rag_answer)
    monkeypatch.setattr(api, "build_retriever", lambda use_reranker: None)
    store = SessionStore(tmp_path / "api.db")
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "service", pipeline.ChatService(store))
    with TestClient(api.app) as c:
        c.seen = seen
        yield c


def test_chat_multi_turn_flow(client):
    first = client.post("/chat", json={"message": "a novel about a preacher in iowa"}).json()
    sid = first["session_id"]
    assert first["citations"][0]["title"] == "Gilead"
    assert first["sources"][0]["authors"] == "Marilynne Robinson"

    second = client.post("/chat", json={"session_id": sid, "message": "who wrote it?"}).json()
    assert second["standalone_query"] == "who wrote it? (about Gilead)"
    assert client.seen["histories"] == [0, 2]  # turn 2 saw turn 1's user+assistant messages

    messages = client.get(f"/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["meta"]["citations"][0]["chunk_id"] == "b1:0000"


def test_unknown_session_404(client):
    assert client.get("/sessions/nope").status_code == 404


def test_empty_message_rejected(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
