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

from agent_oracle.models import AgentType, Message, Session
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
            msg = _extract_message(record, timestamp)
            if msg is not None:
                if not started_at or started_at == datetime.fromtimestamp(0):
                    started_at = timestamp
                messages.append(msg)

    if started_at == datetime.fromtimestamp(0) and messages:
        started_at = messages[0].timestamp

    return Session(
        id=session_id,
        agent=AgentType.FACTORY,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_message(record: dict, timestamp: datetime) -> Message | None:
    """Build a :class:`Message` from a Factory message record."""
    msg_data = record.get("message", {})
    role_str = msg_data.get("role", "")
    role = MESSAGE_ROLES.get(role_str)
    if role is None:
        return None
    content_parts = msg_data.get("content", [])
    text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
    return Message(
        role=role,
        content=text,
        timestamp=timestamp,
        message_id=record.get("id"),
    )
