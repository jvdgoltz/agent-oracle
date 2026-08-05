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

        msg = _extract_messages(record, timestamp)
        messages.extend(msg)

    return Session(
        id=session_id,
        agent=AgentType.CLAUDE,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_messages(record: dict, timestamp: datetime) -> list[Message]:
    """Build :class:`Message` objects from a Claude message record.

    Claude content can be a plain string (user turns) or an array of parts:
    ``text`` parts are concatenated into one message; ``thinking`` parts carry
    their text in the ``thinking`` field (often empty for redacted thinking,
    which is skipped); ``tool_use`` and ``tool_result`` parts are tool traffic
    and never enter the index. Records with no conversation content yield no
    messages.
    """
    msg_data = record.get("message", {})
    role = MESSAGE_ROLES.get(msg_data.get("role", ""))
    if role is None:
        return []
    content = msg_data.get("content")
    model = msg_data.get("model")
    is_injected = bool(record.get("isMeta"))

    def build(content_text: str, *, is_thinking: bool = False) -> Message:
        return Message(
            role=role,
            content=content_text,
            timestamp=timestamp,
            message_id=record.get("uuid"),
            is_thinking=is_thinking,
            model=model,
            is_system_instruction=role is MessageRole.SYSTEM,
            is_injected=is_injected,
        )

    if isinstance(content, str):
        return [build(content)] if content else []
    if not isinstance(content, list):
        return []

    messages: list[Message] = []
    text = "".join(
        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
    )
    for part in content:
        if isinstance(part, dict) and part.get("type") == "thinking":
            thinking = part.get("thinking", "")
            if thinking:
                messages.append(build(thinking, is_thinking=True))
    if text:
        messages.append(build(text))
    return messages
