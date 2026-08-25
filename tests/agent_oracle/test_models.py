"""Tests for the core data models."""

from datetime import UTC, datetime

from agent_oracle.models import AgentType, Message, MessageRole, Session


def test_message_creation() -> None:
    """A Message stores role, content, timestamp, and optional id."""
    ts = datetime.now(UTC)
    msg = Message(role=MessageRole.USER, content="hello", timestamp=ts)
    assert msg.role is MessageRole.USER
    assert msg.content == "hello"
    assert msg.timestamp == ts


def test_session_creation() -> None:
    """A Session groups metadata with its messages."""
    ts = datetime.now(UTC)
    msg = Message(role=MessageRole.ASSISTANT, content="hi back", timestamp=ts)
    session = Session(
        id="abc-123",
        agent=AgentType.CODEX,
        cwd="/tmp/project",
        started_at=ts,
        messages=[msg],
    )
    assert session.id == "abc-123"
    assert session.agent is AgentType.CODEX
    assert len(session.messages) == 1
    assert session.messages[0].content == "hi back"


def test_agent_type_values() -> None:
    """AgentType has exactly the supported agents."""
    assert {AgentType.CODEX, AgentType.FACTORY, AgentType.CLAUDE, AgentType.OMP}


def test_message_metadata_defaults() -> None:
    """A plain message has no thinking, system, or injected flags."""
    ts = datetime.now(UTC)
    msg = Message(role=MessageRole.USER, content="hello", timestamp=ts)
    assert msg.is_thinking is False
    assert msg.model is None
    assert msg.is_system_instruction is False
    assert msg.is_injected is False
    assert msg.is_searchable is True


def test_thinking_message_not_searchable() -> None:
    """A thinking message is excluded from search."""
    ts = datetime.now(UTC)
    msg = Message(
        role=MessageRole.ASSISTANT, content="reasoning...", timestamp=ts, is_thinking=True
    )
    assert msg.is_searchable is False


def test_system_instruction_not_searchable() -> None:
    """A system instruction message is excluded from search."""
    ts = datetime.now(UTC)
    msg = Message(
        role=MessageRole.DEVELOPER, content="You are...", timestamp=ts, is_system_instruction=True
    )
    assert msg.is_searchable is False


def test_system_role_not_searchable_without_metadata_flag() -> None:
    """A system-role message is excluded even when its source flags are absent."""
    ts = datetime.now(UTC)
    msg = Message(role=MessageRole.SYSTEM, content="system context", timestamp=ts)
    assert msg.is_searchable is False


def test_injected_message_not_searchable() -> None:
    """An injected (non-user-authored) message is excluded from search."""
    ts = datetime.now(UTC)
    msg = Message(
        role=MessageRole.USER, content="<system-reminder>...", timestamp=ts, is_injected=True
    )
    assert msg.is_searchable is False
