"""Tests for SQLite archive-overview queries."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_oracle.models import AgentType, Message, MessageRole, Session
from agent_oracle.store import Store


def _session(session_id: str, agent: AgentType, cwd: str, messages: list[Message]) -> Session:
    """Build an overview-query session fixture."""
    return Session(session_id, agent, cwd, datetime(2026, 8, 1, tzinfo=UTC), messages)


def test_list_overview_rows_excludes_internal_messages_and_counts_entities(store: Store) -> None:
    """Overview rows use visible conversation messages and distinct entity sessions."""
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    store.index_session(
        _session(
            "s1",
            AgentType.CODEX,
            "/work/a",
            [
                Message(MessageRole.USER, "prompt", timestamp),
                Message(MessageRole.ASSISTANT, "answer", timestamp, model="gpt"),
                Message(MessageRole.ASSISTANT, "thought", timestamp, is_thinking=True),
                Message(MessageRole.USER, "injected", timestamp, is_injected=True),
            ],
        )
    )
    store.upsert_entities("s1", [{"type": "file", "value": "a.py"}])

    rows = store.list_overview_rows()

    assert rows["sessions"] == [{"id": "s1", "agent": "codex", "cwd": "/work/a"}]
    assert rows["assistant_messages"] == [{"model": "gpt", "messages": 1}]
    assert len(rows["session_messages"]) == 2
    assert rows["entities"] == [{"session_id": "s1", "entity_type": "file", "entity_value": "a.py"}]
