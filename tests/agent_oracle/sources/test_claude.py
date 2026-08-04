"""Tests for the Claude Code session normalizer."""

import json
from pathlib import Path

from agent_oracle.models import AgentType, MessageRole
from agent_oracle.sources.claude import parse_claude_session


def _write_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    """Write a JSONL file from a list of dicts and return its path."""
    p = tmp_path / "claude-test.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return p


def test_parse_basic_session(tmp_path: Path) -> None:
    """Claude session with user and assistant messages."""
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "Hello claude"},
            "uuid": "u1",
            "timestamp": "2026-07-23T15:01:20.027Z",
            "sessionId": "claude-001",
            "cwd": "/tmp/project",
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi from claude"}],
            },
            "uuid": "a1",
            "timestamp": "2026-07-23T15:01:25.000Z",
            "sessionId": "claude-001",
            "cwd": "/tmp/project",
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_claude_session(path)

    assert session.id == "claude-001"
    assert session.agent is AgentType.CLAUDE
    assert session.cwd == "/tmp/project"
    assert len(session.messages) == 2
    assert session.messages[0].role is MessageRole.USER
    assert session.messages[0].content == "Hello claude"
    assert session.messages[1].role is MessageRole.ASSISTANT
    assert session.messages[1].content == "Hi from claude"


def test_parse_skips_non_message_records(tmp_path: Path) -> None:
    """Queue operations and attachments are skipped."""
    lines = [
        {
            "type": "queue-operation",
            "operation": "enqueue",
            "timestamp": "2026-07-23T15:01:19.983Z",
            "sessionId": "claude-002",
            "content": "test",
        },
        {
            "type": "queue-operation",
            "operation": "dequeue",
            "timestamp": "2026-07-23T15:01:19.984Z",
            "sessionId": "claude-002",
        },
        {
            "type": "attachment",
            "attachment": {"type": "deferred_tools_delta", "addedNames": []},
            "uuid": "att1",
            "timestamp": "2026-07-23T15:01:20.026Z",
            "sessionId": "claude-002",
            "cwd": "/x",
        },
        {
            "type": "user",
            "message": {"role": "user", "content": "actual message"},
            "uuid": "u1",
            "timestamp": "2026-07-23T15:01:20.027Z",
            "sessionId": "claude-002",
            "cwd": "/x",
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_claude_session(path)

    assert session.id == "claude-002"
    assert len(session.messages) == 1
    assert session.messages[0].content == "actual message"


def test_parse_content_as_string(tmp_path: Path) -> None:
    """Claude content can be a plain string instead of an array."""
    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "plain text content"},
            "uuid": "u1",
            "timestamp": "2026-07-23T15:01:20.027Z",
            "sessionId": "claude-003",
            "cwd": "/x",
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_claude_session(path)

    assert session.messages[0].content == "plain text content"
