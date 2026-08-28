"""Tests for the Pi session normalizer."""

import json
from pathlib import Path

from agent_oracle.models import AgentType, MessageRole
from agent_oracle.sources.pi import parse_pi_session


def test_parse_pi_session_supports_string_content_and_metadata(tmp_path: Path) -> None:
    """Pi's string user content and array assistant content are normalized."""
    path = tmp_path / "pi-session.jsonl"
    records = [
        {
            "type": "session",
            "id": "pi-1",
            "timestamp": "2026-08-28T10:00:00Z",
            "cwd": "/tmp/pi",
        },
        {
            "type": "session_info",
            "id": "i1",
            "timestamp": "2026-08-28T10:00:01Z",
            "name": "Pi work",
        },
        {
            "type": "message",
            "id": "u1",
            "timestamp": "2026-08-28T10:00:02Z",
            "message": {"role": "user", "content": "Hello Pi"},
        },
        {
            "type": "message",
            "id": "a1",
            "timestamp": "2026-08-28T10:00:03Z",
            "message": {
                "role": "assistant",
                "model": "pi-model",
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": "Hi"},
                ],
            },
        },
        {
            "type": "message",
            "id": "t1",
            "timestamp": "2026-08-28T10:00:04Z",
            "message": {
                "role": "toolResult",
                "content": [{"type": "text", "text": "ignored"}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    session = parse_pi_session(path)

    assert session.id == "pi-1"
    assert session.agent is AgentType.PI
    assert session.cwd == "/tmp/pi"
    assert session.title == "Pi work"
    assert [(message.role, message.content) for message in session.messages] == [
        (MessageRole.USER, "Hello Pi"),
        (MessageRole.ASSISTANT, "internal"),
        (MessageRole.ASSISTANT, "Hi"),
    ]
    assert session.messages[1].is_thinking is True
    assert session.messages[1].model == "pi-model"
