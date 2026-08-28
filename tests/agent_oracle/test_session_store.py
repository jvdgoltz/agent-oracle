"""Tests for archived session and entity retrieval."""

from datetime import UTC, datetime

import sqlite_vec

from agent_oracle.models import AgentType, Message, MessageRole, Session, TokenUsage
from agent_oracle.store import Store


def _message(content: str, role: MessageRole = MessageRole.USER) -> Message:
    """Build a minimal message."""
    return Message(role=role, content=content, timestamp=datetime(2026, 1, 1, tzinfo=UTC))


def _session(
    session_id: str,
    *,
    started_at: datetime | None = None,
    messages: list[Message] | None = None,
    agent: AgentType = AgentType.CODEX,
    is_review_agent: bool = False,
    parent_thread_id: str | None = None,
) -> Session:
    """Build a minimal session."""
    return Session(
        id=session_id,
        agent=agent,
        cwd="/tmp/project",
        started_at=started_at or datetime(2026, 1, 1, tzinfo=UTC),
        messages=messages or [],
        is_review_agent=is_review_agent,
        parent_thread_id=parent_thread_id,
    )


def _embedding(value: float) -> list[float]:
    """Build a deterministic embedding."""
    return [value, *([0.0] * 383)]


def test_list_sessions_orders_and_paginates(store: Store) -> None:
    """Session listing returns the newest requested page."""
    for day in range(1, 6):
        store.index_session(_session(f"s{day}", started_at=datetime(2026, 1, day, tzinfo=UTC)))

    assert [row["id"] for row in store.list_sessions(limit=2)] == ["s5", "s4"]
    assert [row["id"] for row in store.list_sessions(limit=2, offset=2)] == ["s3", "s2"]


def test_review_session_is_archived_but_hidden_by_default(store: Store) -> None:
    """Review sessions remain retrievable while default listings hide them."""
    store.index_session(
        _session(
            "review",
            messages=[_message("review-only-term")],
            is_review_agent=True,
            parent_thread_id="parent",
        )
    )

    assert store.get_session("review") is not None
    assert store.list_sessions() == []
    assert [row["id"] for row in store.list_sessions(include_review_agents=True)] == ["review"]
    assert store.list_review_sessions(["parent"])["parent"][0]["id"] == "review"


def test_review_session_can_be_reindexed_without_fts_rows(store: Store) -> None:
    """Re-indexing a review session does not delete nonexistent FTS rows."""
    review = _session("review", messages=[_message("review-only-term")], is_review_agent=True)

    store.index_session(review)
    store.index_session(review)

    assert store.get_session("review") is not None


def test_regular_session_can_be_reindexed_without_corrupting_external_content_fts(
    store: Store,
) -> None:
    """Re-indexing replaces external-content FTS rows with their original text."""
    original = _session(
        "regular",
        messages=[
            _message("injected instructions", MessageRole.SYSTEM),
            _message("original searchable text"),
        ],
    )
    replacement = _session("regular", messages=[_message("replacement searchable text")])

    store.index_session(original)
    store.index_session(replacement)

    assert store.search_text("original") == []
    assert [row["session_id"] for row in store.search_text("replacement")] == ["regular"]


def test_search_excludes_stale_review_entries(store: Store) -> None:
    """Search filters review entries left by an earlier database version."""
    store.index_session(
        _session("review", messages=[_message("review-only-term")], is_review_agent=True)
    )
    message_id = store.conn.execute(
        "SELECT id FROM messages WHERE session_id = 'review'"
    ).fetchone()[0]
    session_rowid = store.conn.execute("SELECT rowid FROM sessions WHERE id = 'review'").fetchone()[
        0
    ]
    store.conn.execute(
        "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
        (message_id, "review-only-term"),
    )
    store.conn.execute(
        "INSERT INTO sessions_fts(rowid, summary) VALUES (?, ?)",
        (session_rowid, "review-only-term"),
    )
    store.conn.execute(
        "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
        (message_id, sqlite_vec.serialize_float32(_embedding(1.0))),
    )
    store.conn.execute(
        "INSERT INTO vec_sessions(rowid, embedding) VALUES (?, ?)",
        (session_rowid, sqlite_vec.serialize_float32(_embedding(1.0))),
    )
    store.conn.commit()

    assert store.search_text("review-only-term") == []
    assert store.search_vector(_embedding(1.0)) == []
    assert store.search_hybrid("review-only-term", _embedding(1.0)) == []


def test_get_session_returns_ordered_messages(store: Store) -> None:
    """Session retrieval includes messages in sequence order."""
    store.index_session(
        _session(
            "s1",
            agent=AgentType.FACTORY,
            messages=[_message("first"), _message("second", MessageRole.ASSISTANT)],
        )
    )

    result = store.get_session("s1")
    assert result is not None
    assert result["agent"] == "factory"
    assert [message["content"] for message in result["messages"]] == ["first", "second"]
    assert [message["seq"] for message in result["messages"]] == [0, 1]


def test_missing_session_and_entities_return_empty_values(store: Store) -> None:
    """Missing retrieval targets return their documented empty values."""
    assert store.get_session("does-not-exist") is None
    assert store.get_entities("does-not-exist") == []


def test_token_usage_is_grouped_without_cumulative_double_count(store: Store) -> None:
    """Usage rows aggregate provider-reported per-response values."""
    store.conn.execute("""CREATE TABLE token_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        timestamp TEXT NOT NULL, model TEXT, input_tokens INTEGER, output_tokens INTEGER,
        cached_input_tokens INTEGER, cache_creation_input_tokens INTEGER,
        cache_read_input_tokens INTEGER, reasoning_output_tokens INTEGER, total_tokens INTEGER
    )""")
    session = Session(
        id="tokens",
        agent=AgentType.CODEX,
        cwd="/tmp/project",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        token_usages=[
            TokenUsage(
                datetime(2026, 1, 1, tzinfo=UTC),
                "gpt",
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
            TokenUsage(
                datetime(2026, 1, 1, tzinfo=UTC),
                "gpt",
                input_tokens=20,
                output_tokens=3,
                total_tokens=23,
            ),
        ],
    )
    store.index_session(session)
    rows = store.list_token_usage()
    assert rows[0]["input_tokens"] == 30
    assert rows[0]["total_tokens"] == 35
