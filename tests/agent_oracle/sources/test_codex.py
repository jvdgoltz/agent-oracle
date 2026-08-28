"""Tests for the Codex session normalizer."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from agent_oracle.models import AgentType, MessageRole
from agent_oracle.sources.codex import (
    _read_codex_title,
    is_codex_session_archived,
    load_codex_session,
    parse_codex_session,
)


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
    assert not session.is_review_agent
    assert session.parent_thread_id is None
    assert len(session.messages) == 2
    assert session.messages[0].role is MessageRole.USER
    assert session.messages[0].content == "Hello world"
    assert session.messages[1].role is MessageRole.ASSISTANT
    assert session.messages[1].content == "Hi there"


def test_parse_uses_latest_codex_thread_name(tmp_path: Path) -> None:
    """The latest append-only Codex thread name becomes the session title."""
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "08" / "21"
    session_dir.mkdir(parents=True)
    path = session_dir / "rollout-sess-title.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "sess-title", "cwd": "/tmp", "timestamp": "2026-01-01Z"},
            }
        )
        + "\n"
    )
    (codex_home / "session_index.jsonl").write_text(
        '{"id":"sess-title","thread_name":"Old title","updated_at":"one"}\n'
        '{"id":"sess-title","thread_name":"Current title","updated_at":"two"}\n'
    )

    assert parse_codex_session(path).title == "Current title"


def test_state_title_lookup_closes_sqlite_connection(tmp_path: Path, monkeypatch) -> None:
    """State database title lookup releases its read-only connection."""
    session_dir = tmp_path / ".codex" / "sessions" / "2026" / "08" / "25"
    session_dir.mkdir(parents=True)
    path = session_dir / "rollout-sess-title.jsonl"
    state_path = tmp_path / ".codex" / "state_5.sqlite"
    state_path.touch()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = ("State title", None)
    monkeypatch.setattr(
        "agent_oracle.sources.codex.sqlite3.connect", lambda *_args, **_kwargs: connection
    )

    assert _read_codex_title(path, "sess-title") == "State title"
    connection.close.assert_called_once_with()


def test_parse_preserves_review_parent_metadata(tmp_path: Path) -> None:
    """Codex guardian sessions retain their review marker and parent thread ID."""
    session = parse_codex_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "review-1",
                        "cwd": "/work",
                        "timestamp": "2026-08-01T00:00:00Z",
                        "parent_thread_id": "parent-1",
                        "source": {"subagent": {"other": "guardian"}},
                    },
                }
            ],
        )
    )

    assert session.is_review_agent
    assert session.parent_thread_id == "parent-1"


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


def test_parse_skips_token_count_without_usage_info(tmp_path: Path) -> None:
    """A token-count event with null info does not abort session parsing."""
    session = parse_codex_session(
        _write_jsonl(
            tmp_path,
            [
                {"type": "session_meta", "payload": {"id": "null-usage", "cwd": "/x"}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
            ],
        )
    )

    assert session.id == "null-usage"
    assert session.token_usages == []


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


def test_parse_skips_truncated_line(tmp_path: Path) -> None:
    """A malformed final line (partial write) is skipped, not fatal."""
    truncated = (
        '{"type":"response_item","payload":{"type":"message","role":"assistant",'
        '"content":[{"type":"text","text":"par'
    )
    path = tmp_path / "rollout-trunc.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "trunc-001",
                    "cwd": "/tmp",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
                "timestamp": "2024-01-01T00:00:01Z",
            }
        )
        + "\n"
        + truncated
    )
    session = parse_codex_session(path)

    assert session.id == "trunc-001"
    assert len(session.messages) == 1
    assert session.messages[0].content == "hi"


def test_parse_records_interrupted_turn(tmp_path: Path) -> None:
    """Codex turn_aborted events mark the preceding real user message."""
    session = parse_codex_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"text": "stop"}]},
                },
                {"type": "turn_context", "payload": {"model": "gpt-5.6"}},
                {
                    "type": "event_msg",
                    "timestamp": "2026-08-01T12:00:03Z",
                    "payload": {
                        "type": "turn_aborted",
                        "reason": "interrupted",
                        "turn_id": "turn-1",
                        "completed_at": 1786701652,
                    },
                },
            ],
        )
    )

    assert [
        (item.source_id, item.model, item.user_message_seq) for item in session.interruptions
    ] == [("turn-1", "gpt-5.6", 0)]
    timestamp = session.interruptions[0].timestamp
    assert timestamp is not None
    assert timestamp.isoformat() == "2026-08-01T12:00:03+00:00"


def test_parse_excludes_synthetic_user_abort_marker(tmp_path: Path) -> None:
    """Codex interruption events skip their preceding synthetic user notice."""
    marker = "<turn_aborted>\nThe user interrupted the previous turn on purpose.\n</turn_aborted>"
    session = parse_codex_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"text": "real"}]},
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"text": "answer"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": [{"text": marker}]},
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "turn_aborted",
                        "reason": "interrupted",
                        "turn_id": "turn-1",
                    },
                },
            ],
        )
    )

    assert session.messages[2].is_injected is True
    assert session.interruptions[0].user_message_seq == 0


def test_parse_marks_recommended_plugins_user_message_as_injected(tmp_path: Path) -> None:
    """Codex plugin recommendation metadata is not a real user prompt."""
    session = parse_codex_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": "<recommended_plugins>\n- GitHub"}],
                    },
                }
            ],
        )
    )

    assert session.messages[0].is_injected is True


def test_parse_marks_standalone_codex_context_markers_as_injected(tmp_path: Path) -> None:
    """Codex-injected AGENTS and environment wrappers are not user prompts."""
    markers = [
        "# AGENTS.md instructions for /tmp/project\n",
        "<environment_context>\n",
        '<codex_internal_context source="goal">\n',
        "<codex_delegation>\n",
        "<realtime_delegation>\n",
        "<subagent_notification>\n",
        "# Task Tool Invocation\n",
        "### Session update [in progress — more steps follow]\n",
        "<system-notification>\n",
    ]
    for marker in markers:
        session = parse_codex_session(
            _write_jsonl(
                tmp_path,
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "text", "text": marker + "injected context"}],
                        },
                    }
                ],
            )
        )
        assert session.messages[0].is_injected is True


def test_parse_keeps_ordinary_user_message_non_injected(tmp_path: Path) -> None:
    """A normal user message remains searchable conversation content."""
    session = parse_codex_session(
        _write_jsonl(
            tmp_path,
            [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "# AGENTS.md instructions are unclear"}
                        ],
                    },
                }
            ],
        )
    )

    assert session.messages[0].is_injected is False


def test_load_codex_session_reads_the_matching_local_thread(tmp_path: Path) -> None:
    """A direct thread lookup loads only the requested local Codex JSONL."""
    path = tmp_path / "2026" / "08" / "14" / "rollout-2026-08-14-thread-1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "thread-1", "cwd": "/tmp/project"},
            }
        )
        + "\n"
    )

    session = load_codex_session("thread-1", tmp_path)

    assert session is not None
    assert session.id == "thread-1"
    assert load_codex_session("missing", tmp_path) is None


def test_load_codex_session_finds_an_archived_thread(tmp_path: Path, monkeypatch) -> None:
    """Default lookup reads Codex sessions after Codex archives their JSONL."""
    codex_home = tmp_path / ".codex"
    archived = codex_home / "archived_sessions" / "rollout-2026-thread-1.jsonl"
    archived.parent.mkdir(parents=True)
    archived.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "thread-1", "cwd": "/tmp/project"},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    session = load_codex_session("thread-1")

    assert session is not None
    assert session.id == "thread-1"
    assert is_codex_session_archived("thread-1") is True
