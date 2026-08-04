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
    """AgentType has exactly the three supported agents."""
    assert {AgentType.CODEX, AgentType.FACTORY, AgentType.CLAUDE}
