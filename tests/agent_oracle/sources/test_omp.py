"""Tests for the OMP session normalizer."""

import json
from pathlib import Path

from agent_oracle.models import AgentType, MessageRole
from agent_oracle.sources.omp import parse_omp_session


def _write_jsonl(tmp_path: Path, lines: list[dict]) -> Path:
    """Write a JSONL file from a list of dicts and return its path."""
    p = tmp_path / "2026-08-05T10-00-00-000Z_omp-fallback.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return p


def _session_record() -> dict:
    """Return the session metadata record opening every OMP file."""
    return {
        "type": "session",
        "version": 3,
        "id": "omp-001",
        "timestamp": "2026-08-05T10:00:00.000Z",
        "cwd": "/tmp/project",
        "title": "Demo",
    }


def _message_record(
    role: str, content: list[dict], message_id: str, model: str | None = None
) -> dict:
    """Wrap message content in an OMP message record."""
    message: dict = {"role": role, "content": content}
    if model is not None:
        message["model"] = model
    return {
        "type": "message",
        "id": message_id,
        "timestamp": "2026-08-05T10:00:05.000Z",
        "message": message,
    }


def test_parse_basic_session(tmp_path: Path) -> None:
    """A session record followed by user and assistant messages."""
    lines = [
        _session_record(),
        _message_record("user", [{"type": "text", "text": "Hello omp"}], "u1"),
        _message_record(
            "assistant", [{"type": "text", "text": "Hi from omp"}], "a1", model="test-model"
        ),
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert session.id == "omp-001"
    assert session.agent is AgentType.OMP
    assert session.cwd == "/tmp/project"
    assert session.started_at.isoformat() == "2026-08-05T10:00:00+00:00"
    assert len(session.messages) == 2
    assert session.messages[0].role is MessageRole.USER
    assert session.messages[0].content == "Hello omp"
    assert session.messages[0].message_id == "u1"
    assert session.messages[1].role is MessageRole.ASSISTANT
    assert session.messages[1].content == "Hi from omp"
    assert session.messages[1].model == "test-model"


def test_parse_skips_non_message_records(tmp_path: Path) -> None:
    """Records that are not messages are skipped."""
    lines = [
        _session_record(),
        {"type": "thinking_level_change", "id": "t1", "timestamp": "2026-08-05T10:00:01.000Z"},
        {"type": "model_change", "id": "m1", "timestamp": "2026-08-05T10:00:02.000Z"},
        {
            "type": "custom",
            "customType": "tool_execution_start",
            "data": {},
            "timestamp": "2026-08-05T10:00:03.000Z",
        },
        {"type": "title", "title": "Demo", "timestamp": "2026-08-05T10:00:04.000Z"},
        _message_record("user", [{"type": "text", "text": "actual message"}], "u1"),
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert len(session.messages) == 1
    assert session.messages[0].content == "actual message"


def test_parse_skips_tool_traffic_roles(tmp_path: Path) -> None:
    """toolResult and bashExecution messages are agent traffic, not conversation."""
    lines = [
        _session_record(),
        _message_record(
            "toolResult", [{"type": "toolResult", "toolName": "read", "result": "bytes"}], "t1"
        ),
        _message_record("bashExecution", [{"type": "text", "text": "$ ls output"}], "b1"),
        _message_record("user", [{"type": "text", "text": "next question"}], "u1"),
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert len(session.messages) == 1
    assert session.messages[0].content == "next question"


def test_parse_concatenates_text_parts(tmp_path: Path) -> None:
    """Multiple text content parts are joined."""
    lines = [
        _message_record(
            "assistant",
            [{"type": "text", "text": "first "}, {"type": "text", "text": "second"}],
            "a1",
        ),
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert session.messages[0].content == "first second"


def test_parse_thinking_part_uses_thinking_field(tmp_path: Path) -> None:
    """Thinking parts store their text in the 'thinking' field, not 'text'."""
    lines = [
        _message_record(
            "assistant",
            [
                {"type": "thinking", "thinking": "reasoning here", "thinkingSignature": "sig"},
                {"type": "text", "text": "visible answer"},
            ],
            "a1",
        ),
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert len(session.messages) == 2
    assert session.messages[0].is_thinking is True
    assert session.messages[0].content == "reasoning here"
    assert session.messages[1].is_thinking is False
    assert session.messages[1].content == "visible answer"


def test_parse_skips_tool_call_and_image_parts(tmp_path: Path) -> None:
    """toolCall and image parts are not added to the index."""
    lines = [
        _message_record(
            "assistant",
            [
                {"type": "toolCall", "id": "call_1", "name": "read", "arguments": {}},
                {"type": "image", "data": "blob:sha256:abc", "mimeType": "image/webp"},
                {"type": "text", "text": "Visible reply."},
            ],
            "a1",
        ),
    ]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert len(session.messages) == 1
    assert session.messages[0].content == "Visible reply."


def test_parse_falls_back_to_path_stem_without_session_record(tmp_path: Path) -> None:
    """Files without a session record use the file name and first message time."""
    lines = [_message_record("user", [{"type": "text", "text": "orphan"}], "u1")]
    path = _write_jsonl(tmp_path, lines)
    session = parse_omp_session(path)

    assert session.id == "2026-08-05T10-00-00-000Z_omp-fallback"
    assert session.cwd == ""
    assert session.started_at.isoformat() == "2026-08-05T10:00:05+00:00"


def test_parse_skips_truncated_line(tmp_path: Path) -> None:
    """A malformed final line (partial write) is skipped, not fatal."""
    path = tmp_path / "2026-08-05T10-00-00-000Z_omp-fallback.jsonl"
    path.write_text(
        json.dumps(_session_record())
        + "\n"
        + json.dumps(_message_record("user", [{"type": "text", "text": "hello"}], "u1"))
        + "\n"
        + '{"type":"message","id":"trunc","timestamp":"2026-08-05T10:00:06.000Z","message":{"ro'
    )
    session = parse_omp_session(path)

    assert len(session.messages) == 1
    assert session.messages[0].content == "hello"


def test_parse_records_user_interruption(tmp_path: Path) -> None:
    """OMP aborted assistant messages mark the preceding real user message."""
    record = _message_record("assistant", [], "a1", model="omp-model")
    record["message"].update({"stopReason": "aborted", "errorMessage": "Interrupted by user"})
    session = parse_omp_session(
        _write_jsonl(
            tmp_path, [_message_record("user", [{"type": "text", "text": "stop"}], "u1"), record]
        )
    )

    assert [
        (item.source_id, item.model, item.user_message_seq) for item in session.interruptions
    ] == [("a1", "omp-model", 0)]
