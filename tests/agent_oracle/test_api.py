"""Tests for the FastAPI REST API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_oracle.api import create_app


def _client(
    store: MagicMock | None = None,
    embedder: MagicMock | None = None,
    summarizer: MagicMock | None = None,
):
    """Return a TestClient plus the injected store and embedder mocks."""
    store = store or MagicMock()
    embedder = embedder or MagicMock()
    app = create_app(store, embedder, summarizer)
    return TestClient(app), store, embedder


def test_health_returns_ok() -> None:
    """The health endpoint reports a live service."""
    client, _store, _embedder = _client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_behavior_stats_passes_bounded_filters_to_store() -> None:
    """Behavior statistics accept agent and inclusive calendar-date filters."""
    store = MagicMock()
    store.list_behavior_messages.return_value = [
        {
            "content": "Wrong, you missed it",
            "timestamp": "2026-08-01T12:00:00+00:00",
            "agent": "codex",
            "cwd": "/work/a",
            "is_injected": 0,
        }
    ]
    client, store, _embedder = _client(store=store)

    response = client.get("/api/stats/behavior?agent=codex&start=2026-08-01&end=2026-08-01")

    assert response.status_code == 200
    assert response.json()["totals"]["negation"] == 1
    assert response.json()["totals"]["blame"] == 1
    assert response.json()["totals"]["detected_messages"] == 1
    assert response.json()["totals"]["detection_rate"] == 100.0
    assert response.json()["models"][0]["model"] == "unknown"
    store.list_behavior_messages.assert_called_once()


def test_behavior_stats_rejects_reversed_dates() -> None:
    """Behavior statistics reject an invalid date range."""
    client, _store, _embedder = _client()

    response = client.get("/api/stats/behavior?start=2026-08-02&end=2026-08-01")

    assert response.status_code == 422


def test_list_sessions_returns_paginated_results() -> None:
    """List sessions returns the pages and a total count."""
    store = MagicMock()
    sessions = [
        {
            "id": "s2",
            "agent": "claude",
            "cwd": "/p2",
            "title": "Claude title",
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
    store.count_sessions.return_value = 12
    store.list_entities.return_value = {
        "s2": [{"entity_type": "product", "entity_value": "SQLite"}]
    }
    store.list_review_sessions.return_value = {}
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/sessions?limit=10&offset=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sessions"][0]["entities"] == [{"type": "product", "value": "SQLite"}]
    assert body["sessions"][1]["entities"] == []
    assert body["total"] == 12
    store.list_sessions.assert_called_once_with(
        limit=10, offset=5, include_review_agents=False, agent=None
    )

    for key in ("agent", "cwd", "title", "summary"):
        assert body["sessions"][0][key] == sessions[0][key]


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
    store.get_entities.return_value = [{"entity_type": "product", "entity_value": "SQLite"}]
    store.list_review_sessions.return_value = {}
    client, store, _embedder = _client(store=store)
    resp = client.get("/api/sessions/s1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "s1"
    assert body["agent"] == "codex"
    assert body["entities"] == [{"type": "product", "value": "SQLite"}]
    assert body["messages"][0]["content"] == "hi"
    store.get_session.assert_called_once_with("s1")
    store.get_entities.assert_called_once_with("s1")


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
    store.list_entities.return_value = {}
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=bug&mode=text")
    assert resp.status_code == 200
    assert resp.json()["results"] == [
        {
            "session_id": "s1",
            "agent": None,
            "cwd": None,
            "title": None,
            "started_at": None,
            "summary": None,
            "entities": [],
            "snippet": "fix [the] bug",
            "message_snippets": ["fix [the] bug"],
            "score": -1.0,
        }
    ]
    store.search_text.assert_called_once_with("bug", limit=20, agent=None, entity=None)
    embedder.embed_query.assert_not_called()


def test_search_groups_multiple_messages_same_session() -> None:
    """Multiple message hits for the same session are merged into one result."""
    store = MagicMock()
    embedder = MagicMock()
    store.search_text.return_value = [
        {"session_id": "s1", "snippet": "first match", "rank": -1.0},
        {"session_id": "s1", "snippet": "second match", "rank": -2.0},
        {"session_id": "s2", "snippet": "other session", "rank": -3.0},
    ]
    store.list_entities.return_value = {}
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=bug&mode=text")
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["session_id"] == "s1"
    assert results[0]["message_snippets"] == ["first match", "second match"]
    assert results[1]["session_id"] == "s2"
    assert results[1]["message_snippets"] == ["other session"]


def test_search_hybrid_embeds_query() -> None:
    """Hybrid search embeds the query then runs RRF search by default."""
    store = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    store.search_hybrid.return_value = [{"session_id": "s1", "score": 0.9}]
    store.list_entities.return_value = {
        "s1": [{"entity_type": "product", "entity_value": "SQLite"}]
    }
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=bug")
    assert resp.status_code == 200
    assert resp.json()["results"] == [
        {
            "session_id": "s1",
            "agent": None,
            "cwd": None,
            "title": None,
            "started_at": None,
            "summary": None,
            "entities": [{"type": "product", "value": "SQLite"}],
            "snippet": "",
            "message_snippets": [],
            "score": 0.9,
        }
    ]
    embedder.embed_query.assert_called_once_with("bug")
    store.search_hybrid.assert_called_once_with(
        "bug", [0.1, 0.2], limit=20, agent=None, entity=None
    )


def test_search_vector_embeds_query() -> None:
    """Vector search embeds the query and maps distance to score."""
    store = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.3, 0.4]
    store.search_vector.return_value = [{"session_id": "s2", "distance": 0.2}]
    store.list_entities.return_value = {}
    client, store, embedder = _client(store=store, embedder=embedder)
    resp = client.get("/api/search?q=fix&mode=vector")
    assert resp.status_code == 200
    assert resp.json()["results"] == [
        {
            "session_id": "s2",
            "agent": None,
            "cwd": None,
            "title": None,
            "started_at": None,
            "summary": None,
            "entities": [],
            "snippet": "",
            "message_snippets": [],
            "score": 0.2,
        }
    ]
    embedder.embed_query.assert_called_once_with("fix")
    store.search_vector.assert_called_once_with([0.3, 0.4], limit=20, agent=None, entity=None)


@pytest.mark.parametrize("mode", ["text", "vector", "hybrid"])
def test_search_passes_filters_to_candidate_queries(mode: str) -> None:
    """The API scopes candidates in the store without reading transcripts or per-hit entities."""
    client, store, embedder = _client()
    search = getattr(store, f"search_{mode}")
    search.return_value = [{"session_id": "s1", "agent": "codex", "snippet": "a", "rank": -1.0}]
    store.list_entities.return_value = {}
    embedder.embed_query.return_value = [0.1, 0.2]
    response = client.get(f"/api/search?q=bug&mode={mode}&agent=codex&entity=SQLite&limit=1")
    assert response.status_code == 200
    assert [row["session_id"] for row in response.json()["results"]] == ["s1"]
    assert search.call_args.kwargs == {"limit": 1, "agent": "codex", "entity": "SQLite"}
    store.get_session.assert_not_called()
    store.get_entities.assert_not_called()


@pytest.mark.parametrize(
    "mode, field, scores",
    [
        ("text", "rank", [-3.0, -2.0, -1.0]),
        ("vector", "distance", [0.1, 0.2, 0.3]),
        ("hybrid", "score", [0.9, 0.8, 0.7]),
    ],
)
def test_search_grouping_preserves_best_ordered_hit(
    mode: str, field: str, scores: list[float]
) -> None:
    """Grouping retains the first ranked hit and the session ranking."""
    client, store, _ = _client()
    getattr(store, f"search_{mode}").return_value = [
        {"session_id": "s1", "snippet": "best", field: scores[0]},
        {"session_id": "s2", "snippet": "middle", field: scores[1]},
        {"session_id": "s1", "snippet": "worse", field: scores[2]},
    ]
    store.list_entities.return_value = {}
    results = client.get(f"/api/search?q=needle&mode={mode}").json()["results"]
    assert [row["session_id"] for row in results] == ["s1", "s2"]
    assert results[0]["score"] == scores[0]
    assert results[0]["snippet"] == "best"
    assert results[0]["message_snippets"] == ["best", "worse"]


def test_session_feed_filters_and_returns_matching_total() -> None:
    """The feed applies its agent filter before pagination and reports all matches."""
    client, store, _ = _client()
    store.list_sessions.return_value = []
    store.count_sessions.return_value = 17
    response = client.get("/api/sessions?agent=claude&limit=2&offset=20")
    assert response.json() == {"sessions": [], "total": 17}
    store.list_sessions.assert_called_once_with(
        limit=2, offset=20, include_review_agents=False, agent="claude"
    )
    store.count_sessions.assert_called_once_with(include_review_agents=False, agent="claude")
