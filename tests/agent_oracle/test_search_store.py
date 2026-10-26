"""Tests for filtered SQLite text and vector search."""

import pytest
from test_store import _make_embedding, _make_message, _make_session

from agent_oracle.models import AgentType
from agent_oracle.store import Store

# search_text


def test_search_text_returns_matching_sessions(store: Store) -> None:
    """FTS5 search returns session_id, snippet, and rank for matches."""
    store.index_session(
        _make_session(
            session_id="s1",
            messages=[_make_message("hello world from codex")],
        )
    )
    store.index_session(
        _make_session(
            session_id="s2",
            messages=[_make_message("goodbye world")],
        )
    )

    s1 = store.get_session("s1")
    assert s1 is not None

    results = store.search_text("hello", limit=10)
    assert len(results) == 1
    assert results[0]["session_id"] == "s1"
    assert results[0]["message_id"] == s1["messages"][0]["id"]
    assert "hello" in results[0]["snippet"]
    assert isinstance(results[0]["rank"], float)


def test_search_text_no_results(store: Store) -> None:
    """A query matching nothing returns an empty list."""
    store.index_session(_make_session(messages=[_make_message("hello world")]))
    results = store.search_text("nonexistent", limit=10)
    assert results == []


def test_merge_by_score_drops_null_rank_rows() -> None:
    """Rows with a NULL score are dropped instead of crashing the sort."""
    from agent_oracle.search_store import _merge_by_score

    def row(rank, snippet="x"):
        """Build a scored search row."""
        return {"rank": rank, "snippet": snippet}

    message_rows = [row(None, "unscoreable"), row(-0.5, "good")]
    summary_rows = [row(None, "unscoreable-summary")]
    merged = _merge_by_score(message_rows, summary_rows, "rank")
    assert [r["rank"] for r in merged] == [-0.5]


# search_vector


def test_search_vector_returns_nearest(store: Store) -> None:
    """Vector search returns the session with the nearest embedding first."""
    store.index_session(
        _make_session(
            session_id="s1",
            messages=[_make_message("first message")],
        )
    )
    store.index_session(
        _make_session(
            session_id="s2",
            messages=[_make_message("second message")],
        )
    )

    s1 = store.get_session("s1")
    s2 = store.get_session("s2")
    assert s1 is not None and s2 is not None

    store.upsert_embedding(s1["messages"][0]["id"], _make_embedding(1.0, index=0))
    store.upsert_embedding(s2["messages"][0]["id"], _make_embedding(1.0, index=1))

    results = store.search_vector(_make_embedding(1.0, index=0), limit=5)
    assert len(results) == 2
    assert results[0]["session_id"] == "s1"
    assert results[0]["message_id"] == s1["messages"][0]["id"]
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_search_vector_no_results(store: Store) -> None:
    """Vector search on an empty index returns an empty list."""
    results = store.search_vector(_make_embedding(1.0), limit=5)
    assert results == []


@pytest.mark.parametrize("mode", ["text", "vector", "hybrid"])
@pytest.mark.parametrize("source", ["message", "summary"])
@pytest.mark.parametrize(
    "filters", [{"agent": "claude"}, {"entity": "SQLite"}, {"agent": "claude", "entity": "SQLite"}]
)
def test_search_filters_before_candidate_limit(
    store: Store, mode: str, source: str, filters: dict
) -> None:
    """Eligible matches survive when a stronger unrelated match fills top-k."""
    for sid, agent, content, vector in [
        ("excluded", AgentType.CODEX, "needle", _make_embedding(1.0)),
        (
            "included",
            AgentType.CLAUDE,
            "needle with additional context",
            _make_embedding(1.0, index=1),
        ),
    ]:
        store.index_session(
            _make_session(
                sid, agent=agent, messages=[_make_message(content)] if source == "message" else []
            )
        )
        if source == "message":
            session = store.get_session(sid)
            assert session is not None
            message_id = session["messages"][0]["id"]
            store.upsert_embedding(message_id, vector)
        else:
            store.set_summary(sid, content)
            store.upsert_session_embedding(sid, vector)
    store.upsert_entities("included", [{"type": "product", "value": "SQLite"}])
    if mode == "text":
        results = store.search_text("needle", limit=1, **filters)
    elif mode == "vector":
        results = store.search_vector(_make_embedding(1.0), limit=1, **filters)
    else:
        results = store.search_hybrid("needle", _make_embedding(1.0), limit=1, **filters)
    assert [row["session_id"] for row in results] == ["included"]
