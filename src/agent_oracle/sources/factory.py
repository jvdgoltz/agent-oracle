"""Normalizer for Factory Droid session JSONL files.

Factory stores sessions at ``~/.factory/sessions/<project>/<uuid>.jsonl``.
Each line is a JSON object:

- ``session_start``: session id, cwd, title, owner
- ``message``: user/assistant messages with ``message.role`` and ``message.content[]``
- Other record types (``settings_update``, etc.) are skipped.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Message, MessageRole, Session
from agent_oracle.sources.common import MESSAGE_ROLES, parse_timestamp


def parse_factory_session(path: Path) -> Session:
    """Parse a Factory Droid JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_type = record.get("type")

        if record_type == "session_start":
            session_id = record.get("id", session_id)
            cwd = record.get("cwd", "")
        elif record_type == "message":
            timestamp = parse_timestamp(record.get("timestamp", ""))
            extracted = _extract_messages(record, timestamp)
            if extracted:
                if not started_at or started_at == datetime.fromtimestamp(0):
                    started_at = timestamp
                messages.extend(extracted)

    if started_at == datetime.fromtimestamp(0) and messages:
        started_at = messages[0].timestamp

    return Session(
        id=session_id,
        agent=AgentType.FACTORY,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_messages(record: dict, timestamp: datetime) -> list[Message]:
    """Build one :class:`Message` per logical part of a Factory message record.

    Factory content arrays mix part types: ``text`` (concatenated into one
    message) and ``thinking`` (text lives in the ``thinking`` field, emitted as
    a thinking message). ``tool_use`` and ``tool_result`` parts are skipped:
    tool traffic is not conversation and must not enter the index.
    Records producing no content yield no messages.
    """
    msg_data = record.get("message", {})
    role = MESSAGE_ROLES.get(msg_data.get("role", ""))
    if role is None:
        return []
    content_parts = msg_data.get("content", [])
    message_id = record.get("id")
    is_injected = msg_data.get("visibility") == "llm_only"
    is_system = role is MessageRole.SYSTEM

    def build(content: str, *, is_thinking: bool = False) -> Message:
        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
            is_thinking=is_thinking,
            is_system_instruction=is_system,
            is_injected=is_injected,
        )

    messages: list[Message] = []

    for part in content_parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "thinking":
            thinking = part.get("thinking", "")
            if thinking:
                messages.append(build(thinking, is_thinking=True))

    text = "".join(
        p.get("text", "") for p in content_parts if isinstance(p, dict) and p.get("type") == "text"
    )
    if text:
        messages.append(build(text))
    return messages
