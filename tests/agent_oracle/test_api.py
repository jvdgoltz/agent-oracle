"""Tests for the FastAPI REST API."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agent_oracle.api import create_app


def _client(store: MagicMock | None = None, embedder: MagicMock | None = None):
    """Return a TestClient plus the injected store and embedder mocks."""
    store = store or MagicMock()
    embedder = embedder or MagicMock()
    app = create_app(store, embedder)
    return TestClient(app), store, embedder


def test_health_returns_ok() -> None:
    """The health endpoint reports a live service."""
    client, _store, _embedder = _client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_sessions_returns_paginated_results() -> None:
    """List sessions returns the pages and a total count."""
    store = MagicMock()
    sessions = [
        {
            "id": "s2",
            "agent": "claude",
            "cwd": "/p2",
            "started_at": "2026-01-02T00:00:00+00:00",
            "summary": None,
            "enriched": 0,
        },
        {
            "id": "s1",
            "agent": "codex",
            "cwd": "/p1",
            "started_at": "2026-01-01T00:00:00+00:00",
            "summary": None,
            "enriched": 0,
        },
    ]
    store.list_sessions.return_value = sessions
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/sessions?limit=10&offset=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sessions"] == sessions
    assert body["total"] == 2
    store.list_sessions.assert_called_once_with(limit=10, offset=5)


def test_get_session_returns_detail() -> None:
    """Getting a session returns it with all its messages."""
    store = MagicMock()
    session = {
        "id": "s1",
        "agent": "codex",
        "cwd": "/p1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "summary": "A summary.",
        "enriched": 1,
        "messages": [
            {
                "id": 1,
                "session_id": "s1",
                "role": "user",
                "content": "hi",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "seq": 0,
            }
        ],
    }
    store.get_session.return_value = session
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/sessions/s1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "s1"
    assert body["agent"] == "codex"
    assert body["messages"][0]["content"] == "hi"
    store.get_session.assert_called_once_with("s1")


def test_get_session_missing_returns_404() -> None:
    """Getting an unknown session returns a 404."""
    store = MagicMock()
    store.get_session.return_value = None
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/sessions/missing")
    assert resp.status_code == 404


def test_get_entities_for_session() -> None:
    """Entities for a session are returned as a list."""
    store = MagicMock()
    store.get_entities.return_value = [{"entity_type": "product", "entity_value": "SQLite"}]
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/entities?session_id=s1")
    assert resp.status_code == 200
    assert resp.json() == {"entities": [{"entity_type": "product", "entity_value": "SQLite"}]}
    store.get_entities.assert_called_once_with("s1")


def test_search_text_mode_does_not_embed() -> None:
    """Text search maps store rows to snippet/score without embedding."""
    store = MagicMock()
    embedder = MagicMock()
    store.search_text.return_value = [
        {"session_id": "s1", "snippet": "fix [the] bug", "rank": -1.0}
    ]
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=bug&mode=text")
    assert resp.status_code == 200
    assert resp.json()["results"] == [
        {"session_id": "s1", "snippet": "fix [the] bug", "score": -1.0}
    ]
    store.search_text.assert_called_once_with("bug", limit=20)
    embedder.embed_query.assert_not_called()


def test_search_hybrid_embeds_query() -> None:
    """Hybrid search embeds the query then runs RRF search by default."""
    store = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    store.search_hybrid.return_value = [{"session_id": "s1", "score": 0.9}]
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=bug")
    assert resp.status_code == 200
    assert resp.json()["results"] == [{"session_id": "s1", "snippet": "", "score": 0.9}]
    embedder.embed_query.assert_called_once_with("bug")
    store.search_hybrid.assert_called_once_with("bug", [0.1, 0.2], limit=20)


def test_search_vector_embeds_query() -> None:
    """Vector search embeds the query and maps distance to score."""
    store = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.3, 0.4]
    store.search_vector.return_value = [{"session_id": "s2", "distance": 0.2}]
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=fix&mode=vector")
    assert resp.status_code == 200
    assert resp.json()["results"] == [{"session_id": "s2", "snippet": "", "score": 0.2}]
    embedder.embed_query.assert_called_once_with("fix")
    store.search_vector.assert_called_once_with([0.3, 0.4], limit=20)


def test_search_filters_by_agent() -> None:
    """Results are dropped when their session agent does not match."""
    store = MagicMock()
    store.search_text.return_value = [
        {"session_id": "s1", "snippet": "a", "rank": -1.0},
        {"session_id": "s2", "snippet": "b", "rank": -2.0},
    ]
    store.get_session.side_effect = [
        {"id": "s1", "agent": "codex"},
        {"id": "s2", "agent": "claude"},
    ]
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/search?q=bug&mode=text&agent=codex")
    assert resp.status_code == 200
    assert [r["session_id"] for r in resp.json()["results"]] == ["s1"]


def test_search_filters_by_entity() -> None:
    """Results are dropped when their session lacks the matching entity."""
    store = MagicMock()
    store.search_text.return_value = [
        {"session_id": "s1", "snippet": "a", "rank": -1.0},
        {"session_id": "s2", "snippet": "b", "rank": -2.0},
    ]
    store.get_entities.side_effect = [
        [{"entity_type": "product", "entity_value": "SQLite"}],
        [{"entity_type": "person", "entity_value": "Ada"}],
    ]
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/search?q=bug&mode=text&entity=SQLite")
    assert resp.status_code == 200
    assert [r["session_id"] for r in resp.json()["results"]] == ["s1"]
