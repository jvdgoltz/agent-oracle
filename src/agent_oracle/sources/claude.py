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
from agent_oracle.sources.common import MESSAGE_ROLES, parse_timestamp

_MESSAGE_TYPES = {"user", "assistant"}


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
        timestamp = parse_timestamp(record.get("timestamp", ""))
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
    role = MESSAGE_ROLES.get(role_str)
    if role is None:
        return None
    content = msg_data.get("content")
    text = _extract_content_text(content)
    is_thinking = _has_thinking_content(content)
    model = msg_data.get("model")
    is_system = role is MessageRole.SYSTEM
    is_injected = bool(record.get("isMeta"))

    return Message(
        role=role,
        content=text,
        timestamp=timestamp,
        message_id=record.get("uuid"),
        is_thinking=is_thinking,
        model=model,
        is_system_instruction=is_system,
        is_injected=is_injected,
    )


def _has_thinking_content(content: object) -> bool:
    """Return True when the content array contains a thinking part."""
    if not isinstance(content, list):
        return False
    return any(isinstance(p, dict) and p.get("type") == "thinking" for p in content)


def _extract_content_text(content: object) -> str:
    """Extract text from Claude content which can be a string or array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        return "".join(parts)
    return ""
