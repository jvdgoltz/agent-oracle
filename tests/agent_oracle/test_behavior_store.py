"""Tests for SQLite behavior-statistics queries."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_oracle.models import AgentType, Message, MessageRole, Session
from agent_oracle.store import Store


def _session(session_id: str, agent: AgentType, cwd: str, messages: list[Message]) -> Session:
    """Build a behavior-query session fixture."""
    return Session(
        id=session_id,
        agent=agent,
        cwd=cwd,
        started_at=datetime(2026, 8, 1, tzinfo=UTC),
        messages=messages,
    )


def test_list_behavior_messages_excludes_internal_traffic(store: Store) -> None:
    """Behavior rows include only real user messages with session fields."""
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    store.index_session(
        _session(
            "behavior",
            AgentType.CODEX,
            "/work/project",
            [
                Message(role=MessageRole.USER, content="real", timestamp=timestamp),
                Message(role=MessageRole.ASSISTANT, content="assistant", timestamp=timestamp),
                Message(
                    role=MessageRole.USER, content="thinking", timestamp=timestamp, is_thinking=True
                ),
                Message(
                    role=MessageRole.USER,
                    content="system",
                    timestamp=timestamp,
                    is_system_instruction=True,
                ),
                Message(
                    role=MessageRole.USER, content="injected", timestamp=timestamp, is_injected=True
                ),
            ],
        )
    )

    assert store.list_behavior_messages() == [
        {
            "content": "real",
            "timestamp": "2026-08-01T12:00:00+00:00",
            "is_injected": 0,
            "agent": "codex",
            "cwd": "/work/project",
            "model": "unknown",
        }
    ]


def test_list_behavior_messages_filters_agent_and_inclusive_utc_dates(store: Store) -> None:
    """Behavior rows honor agent filters and SQLite's inclusive UTC date bounds."""
    store.index_session(
        _session(
            "codex",
            AgentType.CODEX,
            "/work/codex",
            [
                Message(
                    role=MessageRole.USER,
                    content="included",
                    timestamp=datetime.fromisoformat("2026-08-02T00:30:00+02:00"),
                ),
                Message(
                    role=MessageRole.USER,
                    content="excluded",
                    timestamp=datetime(2026, 8, 2, tzinfo=UTC),
                ),
            ],
        )
    )
    store.index_session(
        _session(
            "claude",
            AgentType.CLAUDE,
            "/work/claude",
            [
                Message(
                    role=MessageRole.USER,
                    content="other",
                    timestamp=datetime(2026, 8, 1, tzinfo=UTC),
                )
            ],
        )
    )

    rows = store.list_behavior_messages(
        agent="codex",
        start=datetime(2026, 8, 1, tzinfo=UTC).date(),
        end=datetime(2026, 8, 1, tzinfo=UTC).date(),
    )

    assert [row["content"] for row in rows] == ["included"]


def test_list_behavior_messages_uses_previous_eligible_assistant_model(store: Store) -> None:
    """Skip assistant thinking and system rows when finding a prior model."""
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    store.index_session(
        _session(
            "models",
            AgentType.CODEX,
            "/work/project",
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content="answer",
                    timestamp=timestamp,
                    model="gpt-5.6",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="thought",
                    timestamp=timestamp,
                    model="thinking-model",
                    is_thinking=True,
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="instruction",
                    timestamp=timestamp,
                    model="system-model",
                    is_system_instruction=True,
                ),
                Message(
                    role=MessageRole.USER,
                    content="prompt",
                    timestamp=timestamp,
                ),
            ],
        )
    )

    assert store.list_behavior_messages()[0]["model"] == "gpt-5.6"


def test_list_behavior_messages_attributes_consecutive_prompts_once_each(store: Store) -> None:
    """Attribute consecutive prompts to their shared prior assistant response."""
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    store.index_session(
        _session(
            "consecutive",
            AgentType.CODEX,
            "/work/project",
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content="answer",
                    timestamp=timestamp,
                    model="gpt-5.6",
                ),
                Message(role=MessageRole.USER, content="first", timestamp=timestamp),
                Message(role=MessageRole.USER, content="second", timestamp=timestamp),
            ],
        )
    )

    rows = store.list_behavior_messages()

    assert [(row["content"], row["model"]) for row in rows] == [
        ("first", "gpt-5.6"),
        ("second", "gpt-5.6"),
    ]


def test_list_behavior_messages_uses_unknown_without_a_model_or_response(store: Store) -> None:
    """Use unknown for a first prompt or a prior assistant without a model."""
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    store.index_session(
        _session(
            "unknown",
            AgentType.CODEX,
            "/work/project",
            [
                Message(role=MessageRole.USER, content="null model", timestamp=timestamp),
                Message(role=MessageRole.ASSISTANT, content="answer", timestamp=timestamp),
                Message(role=MessageRole.USER, content="no response", timestamp=timestamp),
            ],
        )
    )

    assert [row["model"] for row in store.list_behavior_messages()] == ["unknown", "unknown"]


def test_list_behavior_messages_does_not_cross_session_model_boundaries(store: Store) -> None:
    """Do not attribute an assistant model from another session."""
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    store.index_session(
        _session(
            "prompt-session",
            AgentType.CODEX,
            "/work/project",
            [Message(role=MessageRole.USER, content="prompt", timestamp=timestamp)],
        )
    )
    store.index_session(
        _session(
            "answer-session",
            AgentType.CODEX,
            "/work/project",
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content="answer",
                    timestamp=timestamp,
                    model="gpt-5.6",
                )
            ],
        )
    )

    rows = store.list_behavior_messages()
    prompt = next(row for row in rows if row["content"] == "prompt")
    assert prompt["model"] == "unknown"
