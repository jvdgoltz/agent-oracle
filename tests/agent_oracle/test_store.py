"""Tests for the SQLite store.

Covers schema creation, session indexing (upsert), embeddings, entities,
summaries, and text / vector / hybrid search.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_oracle.models import AgentType, Message, MessageRole, Session
from agent_oracle.store import Store

_EMBED_DIM = 384


def _make_embedding(value: float, index: int = 0, dim: int = _EMBED_DIM) -> list[float]:
    """Create a deterministic embedding vector for testing."""
    vec = [0.0] * dim
    vec[index] = value
    return vec


def _make_session(
    session_id: str = "sess-001",
    agent: AgentType = AgentType.CODEX,
    messages: list[Message] | None = None,
    started_at: datetime | None = None,
) -> Session:
    """Build a minimal :class:`Session` for testing."""
    ts = started_at or datetime(2026, 1, 1, tzinfo=UTC)
    return Session(
        id=session_id,
        agent=agent,
        cwd="/tmp/project",
        started_at=ts,
        messages=messages or [],
    )


def _make_message(
    content: str,
    role: MessageRole = MessageRole.USER,
) -> Message:
    """Build a minimal :class:`Message` for testing."""
    return Message(
        role=role,
        content=content,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
    )


# --------------------------------------------------------------------------- #
# Schema / initialization
# --------------------------------------------------------------------------- #


def test_init_creates_tables(store: Store) -> None:
    """All expected tables exist after initialization."""
    names = {
        r[0]
        for r in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        ).fetchall()
    }
    assert "sessions" in names
    assert "messages" in names
    assert "messages_fts" in names
    assert "sessions_fts" in names
    assert "vec_sessions" in names
    assert "vec_messages" in names
    assert "entities" in names


def test_init_creates_indexes(store: Store) -> None:
    """Indexes on session_id exist for messages and entities."""
    names = {
        r[0]
        for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    }
    assert "idx_messages_session_id" in names
    assert "idx_entities_session_id" in names


# --------------------------------------------------------------------------- #
# index_session
# --------------------------------------------------------------------------- #


def test_index_session_inserts_session_and_messages(store: Store) -> None:
    """Indexing a session persists its row and all message rows."""
    session = _make_session(
        messages=[
            _make_message("hello world"),
            _make_message("hi back", role=MessageRole.ASSISTANT),
        ]
    )
    store.index_session(session)

    row = store.conn.execute(
        "SELECT id, agent, cwd FROM sessions WHERE id = ?", ("sess-001",)
    ).fetchone()
    assert row is not None
    assert row[0] == "sess-001"
    assert row[1] == "codex"

    count = store.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("sess-001",)
    ).fetchone()[0]
    assert count == 2


def test_index_session_populates_fts(store: Store) -> None:
    """Indexed message content is searchable via FTS5."""
    store.index_session(_make_session(messages=[_make_message("unique fts token")]))

    count = store.conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'unique'"
    ).fetchone()[0]
    assert count == 1


def test_index_session_upsert_replaces_messages(store: Store) -> None:
    """Re-indexing a session removes old messages and inserts new ones."""
    store.index_session(
        _make_session(
            messages=[
                _make_message("old message one"),
                _make_message("old message two"),
            ]
        )
    )
    store.index_session(_make_session(messages=[_make_message("new message")]))

    count = store.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", ("sess-001",)
    ).fetchone()[0]
    assert count == 1

    content = store.conn.execute(
        "SELECT content FROM messages WHERE session_id = ?", ("sess-001",)
    ).fetchone()[0]
    assert content == "new message"


def test_index_session_upsert_clears_old_fts(store: Store) -> None:
    """Re-indexing removes old FTS entries so stale text is not searchable."""
    store.index_session(_make_session(messages=[_make_message("stale content")]))
    store.index_session(_make_session(messages=[_make_message("fresh content")]))

    stale = store.conn.execute(
        "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'stale'"
    ).fetchone()[0]
    assert stale == 0


# --------------------------------------------------------------------------- #
# upsert_embedding
# --------------------------------------------------------------------------- #


def test_upsert_embedding_inserts_vector(store: Store) -> None:
    """An embedding is stored in vec_messages and retrievable via MATCH."""
    store.index_session(_make_session(messages=[_make_message("hello")]))
    msg = store.get_session("sess-001")
    assert msg is not None
    msg_id = msg["messages"][0]["id"]

    store.upsert_embedding(msg_id, _make_embedding(1.0))

    results = store.search_vector(_make_embedding(1.0), limit=5)
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-001"


def test_upsert_embedding_replaces_existing(store: Store) -> None:
    """Re-inserting an embedding for the same message replaces the old one."""
    store.index_session(_make_session(messages=[_make_message("hello")]))
    msg = store.get_session("sess-001")
    assert msg is not None
    msg_id = msg["messages"][0]["id"]

    store.upsert_embedding(msg_id, _make_embedding(1.0, index=0))
    store.upsert_embedding(msg_id, _make_embedding(1.0, index=1))

    # Searching for the old vector should not find a perfect match.
    results_old = store.search_vector(_make_embedding(1.0, index=0), limit=5)
    # The old vector [1,0,...] vs stored [0,1,0,...] -> distance > 0
    assert all(r["distance"] > 0 for r in results_old)

    results_new = store.search_vector(_make_embedding(1.0, index=1), limit=5)
    assert results_new[0]["distance"] == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# upsert_entities
# --------------------------------------------------------------------------- #


def test_upsert_entities_inserts_rows(store: Store) -> None:
    """Entities are persisted with type and value."""
    store.index_session(_make_session())
    store.upsert_entities(
        "sess-001",
        [
            {"type": "file", "value": "src/main.py"},
            {"type": "function", "value": "parse_config"},
        ],
    )

    entities = store.get_entities("sess-001")
    assert len(entities) == 2
    assert {"entity_type": "file", "entity_value": "src/main.py"} in entities
    assert {"entity_type": "function", "entity_value": "parse_config"} in entities


def test_upsert_entities_replaces_existing(store: Store) -> None:
    """Re-upserting entities for a session replaces old ones."""
    store.index_session(_make_session())
    store.upsert_entities("sess-001", [{"type": "file", "value": "old.py"}])
    store.upsert_entities("sess-001", [{"type": "file", "value": "new.py"}])

    entities = store.get_entities("sess-001")
    assert len(entities) == 1
    assert entities[0]["entity_value"] == "new.py"


# --------------------------------------------------------------------------- #
# search_text
# --------------------------------------------------------------------------- #


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

    results = store.search_text("hello", limit=10)
    assert len(results) == 1
    assert results[0]["session_id"] == "s1"
    assert "hello" in results[0]["snippet"]
    assert isinstance(results[0]["rank"], float)


def test_search_text_no_results(store: Store) -> None:
    """A query matching nothing returns an empty list."""
    store.index_session(_make_session(messages=[_make_message("hello world")]))
    results = store.search_text("nonexistent", limit=10)
    assert results == []


# --------------------------------------------------------------------------- #
# search_vector
# --------------------------------------------------------------------------- #


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
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_search_vector_no_results(store: Store) -> None:
    """Vector search on an empty index returns an empty list."""
    results = store.search_vector(_make_embedding(1.0), limit=5)
    assert results == []


# --------------------------------------------------------------------------- #
# search_hybrid
# --------------------------------------------------------------------------- #


def test_search_hybrid_fuses_results(store: Store) -> None:
    """Hybrid search combines text and vector results via RRF."""
    store.index_session(
        _make_session(
            session_id="s1",
            messages=[_make_message("python testing guide")],
        )
    )
    store.index_session(
        _make_session(
            session_id="s2",
            messages=[_make_message("rust memory safety")],
        )
    )

    s1 = store.get_session("s1")
    s2 = store.get_session("s2")
    assert s1 is not None and s2 is not None

    store.upsert_embedding(s1["messages"][0]["id"], _make_embedding(1.0, index=0))
    store.upsert_embedding(s2["messages"][0]["id"], _make_embedding(1.0, index=1))

    # Query text matches s1; query embedding matches s1.
    results = store.search_hybrid("python", _make_embedding(1.0, index=0), limit=10)
    session_ids = [r["session_id"] for r in results]
    assert "s1" in session_ids
    assert results[0]["session_id"] == "s1"
    assert isinstance(results[0]["score"], float)


def test_search_hybrid_finds_both_sources(store: Store) -> None:
    """Hybrid search includes results found by only one modality."""
    store.index_session(
        _make_session(
            session_id="text-only",
            messages=[_make_message("unique text match")],
        )
    )
    store.index_session(
        _make_session(
            session_id="vec-only",
            messages=[_make_message("unrelated content")],
        )
    )

    vo = store.get_session("vec-only")
    assert vo is not None
    store.upsert_embedding(vo["messages"][0]["id"], _make_embedding(1.0, index=0))

    results = store.search_hybrid("unique", _make_embedding(1.0, index=0), limit=10)
    session_ids = {r["session_id"] for r in results}
    assert "text-only" in session_ids
    assert "vec-only" in session_ids


# --------------------------------------------------------------------------- #
# list_sessions
# --------------------------------------------------------------------------- #


def test_list_sessions_most_recent_first(store: Store) -> None:
    """list_sessions returns sessions ordered by started_at descending."""
    store.index_session(
        _make_session(
            session_id="old",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    store.index_session(
        _make_session(
            session_id="new",
            started_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
    )

    results = store.list_sessions(limit=50)
    assert len(results) == 2
    assert results[0]["id"] == "new"
    assert results[1]["id"] == "old"


def test_list_sessions_respects_limit_and_offset(store: Store) -> None:
    """limit and offset paginate the session list."""
    for i in range(5):
        store.index_session(
            _make_session(
                session_id=f"s{i}",
                started_at=datetime(2026, 1, i + 1, tzinfo=UTC),
            )
        )

    page = store.list_sessions(limit=2, offset=0)
    assert len(page) == 2
    page2 = store.list_sessions(limit=2, offset=2)
    assert len(page2) == 2
    assert page[0]["id"] != page2[0]["id"]


# --------------------------------------------------------------------------- #
# get_session
# --------------------------------------------------------------------------- #


def test_get_session_returns_session_with_messages(store: Store) -> None:
    """get_session returns the session row and its ordered messages."""
    store.index_session(
        _make_session(
            session_id="s1",
            agent=AgentType.FACTORY,
            messages=[
                _make_message("first", role=MessageRole.USER),
                _make_message("second", role=MessageRole.ASSISTANT),
            ],
        )
    )

    result = store.get_session("s1")
    assert result is not None
    assert result["id"] == "s1"
    assert result["agent"] == "factory"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == "first"
    assert result["messages"][0]["seq"] == 0
    assert result["messages"][1]["content"] == "second"
    assert result["messages"][1]["seq"] == 1


def test_get_session_returns_none_for_missing(store: Store) -> None:
    """get_session returns None for a non-existent session."""
    assert store.get_session("does-not-exist") is None


# --------------------------------------------------------------------------- #
# get_entities
# --------------------------------------------------------------------------- #


def test_get_entities_returns_empty_for_missing(store: Store) -> None:
    """get_entities returns an empty list for a session with no entities."""
    assert store.get_entities("no-session") == []
