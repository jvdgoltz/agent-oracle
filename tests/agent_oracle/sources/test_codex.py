"""Tests for the Codex session normalizer."""

import json
from pathlib import Path

from agent_oracle.models import AgentType, MessageRole
from agent_oracle.sources.codex import parse_codex_session


def _write_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    """Write a JSONL file from a list of dicts and return its path."""
    p = tmp_path / "rollout-test.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return p


def test_parse_basic_session(tmp_path: Path) -> None:
    """A session with metadata and two messages parses correctly."""
    lines = [
        {
            "timestamp": "2026-05-28T07:20:57.986Z",
            "type": "session_meta",
            "payload": {
                "id": "sess-001",
                "timestamp": "2026-05-28T07:20:57.878Z",
                "cwd": "/tmp/project",
                "originator": "codex-tui",
                "cli_version": "0.134.0",
            },
        },
        {
            "timestamp": "2026-05-28T07:21:03.695Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello world"}],
            },
        },
        {
            "timestamp": "2026-05-28T07:21:10.000Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there"}],
            },
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_codex_session(path)

    assert session.id == "sess-001"
    assert session.agent is AgentType.CODEX
    assert session.cwd == "/tmp/project"
    assert len(session.messages) == 2
    assert session.messages[0].role is MessageRole.USER
    assert session.messages[0].content == "Hello world"
    assert session.messages[1].role is MessageRole.ASSISTANT
    assert session.messages[1].content == "Hi there"


def test_parse_skips_non_message_records(tmp_path: Path) -> None:
    """Event messages and turn contexts are skipped."""
    lines = [
        {
            "timestamp": "2026-05-28T07:20:57.986Z",
            "type": "session_meta",
            "payload": {"id": "sess-002", "timestamp": "2026-05-28T07:20:57.878Z", "cwd": "/x"},
        },
        {
            "timestamp": "2026-05-28T07:20:57.988Z",
            "type": "event_msg",
            "payload": {"type": "task_started"},
        },
        {
            "timestamp": "2026-05-28T07:21:03.697Z",
            "type": "turn_context",
            "payload": {"turn_id": "t1"},
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_codex_session(path)

    assert session.id == "sess-002"
    assert len(session.messages) == 0


def test_parse_concatenates_content_parts(tmp_path: Path) -> None:
    """Multiple content parts in a single message are joined."""
    lines = [
        {
            "timestamp": "2026-05-28T07:20:57.986Z",
            "type": "session_meta",
            "payload": {"id": "sess-003", "timestamp": "2026-05-28T07:20:57.878Z", "cwd": "/x"},
        },
        {
            "timestamp": "2026-05-28T07:21:03.695Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "part1 "},
                    {"type": "input_text", "text": "part2"},
                ],
            },
        },
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_codex_session(path)

    assert session.messages[0].content == "part1 part2"
