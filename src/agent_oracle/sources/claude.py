"""Normalizer for Claude Code session JSONL files.

Claude stores sessions at ``~/.claude/projects/<project>/<uuid>.jsonl``.
Each line is a JSON object with ``type`` field:

- ``user`` / ``assistant``: messages with ``message.role`` and ``message.content``
  (content can be a string or an array of ``{"type": "text", "text": "..."}``)
- ``queue-operation``, ``attachment``, etc.: skipped

Session metadata (sessionId, cwd) is extracted from the first message record.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Message, MessageRole, Session

_MESSAGE_TYPES = {"user", "assistant"}
_MESSAGE_ROLES = {
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
}


def parse_claude_session(path: Path) -> Session:
    """Parse a Claude Code JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_type = record.get("type", "")

        if record_type not in _MESSAGE_TYPES:
            continue

        session_id = record.get("sessionId", session_id)
        cwd = record.get("cwd", cwd) or cwd
        timestamp = _parse_timestamp(record.get("timestamp", ""))
        if started_at == datetime.fromtimestamp(0):
            started_at = timestamp

        msg = _extract_message(record, timestamp)
        if msg is not None:
            messages.append(msg)

    return Session(
        id=session_id,
        agent=AgentType.CLAUDE,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_message(record: dict, timestamp: datetime) -> Message | None:
    """Build a :class:`Message` from a Claude message record."""
    msg_data = record.get("message", {})
    role_str = msg_data.get("role", "")
    role = _MESSAGE_ROLES.get(role_str)
    if role is None:
        return None
    text = _extract_content_text(msg_data.get("content"))
    return Message(
        role=role,
        content=text,
        timestamp=timestamp,
        message_id=record.get("uuid"),
    )


def _extract_content_text(content: object) -> str:
    """Extract text from Claude content which can be a string or array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        return "".join(parts)
    return ""


def _parse_timestamp(raw: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a :class:`datetime`."""
    if not raw:
        return datetime.fromtimestamp(0)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
