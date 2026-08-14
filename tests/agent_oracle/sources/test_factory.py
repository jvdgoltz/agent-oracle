"""Tests for the Factory Droid session normalizer."""

import json
from pathlib import Path

from agent_oracle.models import AgentType, MessageRole
from agent_oracle.sources.factory import parse_factory_session


def _write_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    """Write a JSONL file from a list of dicts and return its path."""
    p = tmp_path / "test-session.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return p


def test_parse_basic_session(tmp_path: Path) -> None:
    """A session_start followed by user and assistant messages."""
    lines = [
        {
            "type": "session_start",
            "id": "fac-001",
            "title": "Test Session",
            "owner": "user",
            "version": 2,
            "cwd": "/tmp/project",
        },
        {
            "type": "message",
            "id": "msg-001",
            "timestamp": "2026-07-03T14:16:22.322Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Hello factory"}],
            },
        },
        {
            "type": "message",
            "id": "msg-002",
            "timestamp": "2026-07-03T14:16:26.724Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi from factory"}],
            },
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_factory_session(path)

    assert session.id == "fac-001"
    assert session.agent is AgentType.FACTORY
    assert session.cwd == "/tmp/project"
    assert len(session.messages) == 2
    assert session.messages[0].role is MessageRole.USER
    assert session.messages[0].content == "Hello factory"
    assert session.messages[1].role is MessageRole.ASSISTANT
    assert session.messages[1].content == "Hi from factory"


def test_parse_skips_non_message_records(tmp_path: Path) -> None:
    """Records that are not messages are skipped."""
    lines = [
        {"type": "session_start", "id": "fac-002", "cwd": "/x"},
        {"type": "settings_update", "id": "s1", "cwd": "/x"},
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_factory_session(path)

    assert session.id == "fac-002"
    assert len(session.messages) == 0


def test_parse_concatenates_content_parts(tmp_path: Path) -> None:
    """Multiple text content parts are joined."""
    lines = [
        {"type": "session_start", "id": "fac-003", "cwd": "/x"},
        {
            "type": "message",
            "id": "msg-1",
            "timestamp": "2026-07-03T14:16:22.322Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first "},
                    {"type": "text", "text": "second"},
                ],
            },
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_factory_session(path)

    assert session.messages[0].content == "first second"


def test_parse_thinking_part_uses_thinking_field(tmp_path: Path) -> None:
    """Thinking parts store their text in the 'thinking' field, not 'text'."""
    lines = [
        {"type": "session_start", "id": "fac-004", "cwd": "/x"},
        {
            "type": "message",
            "id": "msg-1",
            "timestamp": "2026-07-03T14:16:22.322Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "pondering deeply"},
                    {"type": "text", "text": "The answer."},
                ],
            },
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_factory_session(path)

    assert len(session.messages) == 2
    assert session.messages[0].content == "pondering deeply"
    assert session.messages[0].is_thinking is True
    assert session.messages[1].content == "The answer."
    assert session.messages[1].is_thinking is False


def test_parse_skips_tool_parts(tmp_path: Path) -> None:
    """tool_use and tool_result parts are not added to the index."""
    lines = [
        {"type": "session_start", "id": "fac-005", "cwd": "/x"},
        {
            "type": "message",
            "id": "msg-1",
            "timestamp": "2026-07-03T14:16:22.322Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "Skill", "input": {"skill": "x"}}],
            },
        },
        {
            "type": "message",
            "id": "msg-2",
            "timestamp": "2026-07-03T14:16:30.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "tool output text"}],
            },
        },
        {
            "type": "message",
            "id": "msg-3",
            "timestamp": "2026-07-03T14:16:40.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "text", "text": "Visible reply."},
                ],
            },
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_factory_session(path)

    assert len(session.messages) == 1
    assert session.messages[0].content == "Visible reply."


def test_parse_records_cancelled_turn(tmp_path: Path) -> None:
    """Factory outcomes use a turn ID distinct from the preceding user message ID."""
    session = parse_factory_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "message",
                    "id": "user-message-1",
                    "timestamp": "2026-08-01T12:00:00Z",
                    "message": {"role": "user", "content": [{"type": "text", "text": "stop"}]},
                },
                {"type": "agent_turn_outcome", "turnId": "turn-1", "reason": "cancelled"},
            ],
        )
    )

    assert [(item.source_id, item.user_message_seq) for item in session.interruptions] == [
        ("turn-1", 0)
    ]


def test_parse_deduplicates_cancelled_turn_and_ignores_injected_messages(tmp_path: Path) -> None:
    """Factory links a cancelled turn to its real user message exactly once."""
    session = parse_factory_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "message",
                    "id": "user-message-1",
                    "timestamp": "2026-08-01T12:00:00Z",
                    "message": {"role": "user", "content": [{"type": "text", "text": "stop"}]},
                },
                {
                    "type": "message",
                    "id": "tool-result",
                    "timestamp": "2026-08-01T12:00:01Z",
                    "message": {
                        "role": "user",
                        "visibility": "llm_only",
                        "content": [{"type": "tool_result", "content": "hidden"}],
                    },
                },
                {"type": "agent_turn_outcome", "turnId": "turn-1", "reason": "cancelled"},
                {"type": "agent_turn_outcome", "turnId": "turn-1", "reason": "cancelled"},
            ],
        )
    )

    assert [(item.source_id, item.user_message_seq) for item in session.interruptions] == [
        ("turn-1", 0)
    ]
