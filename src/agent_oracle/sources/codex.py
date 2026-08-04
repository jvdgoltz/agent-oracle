"""Normalizer for Codex (OpenAI Codex CLI) session JSONL files.

Codex stores sessions at ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.
Each line is a JSON object with ``type`` and ``payload`` fields:

- ``session_meta``: session id, cwd, timestamp, model, git info
- ``response_item`` with ``payload.type == "message"``: user/assistant/developer messages
- ``event_msg``, ``turn_context``: non-message records that are skipped
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Message, MessageRole, Session

_MESSAGE_ROLES = {
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "developer": MessageRole.DEVELOPER,
    "system": MessageRole.SYSTEM,
}


def parse_codex_session(path: Path) -> Session:
    """Parse a Codex JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_type = record.get("type")
        payload = record.get("payload", {})
        timestamp = _parse_timestamp(record.get("timestamp", ""))

        if record_type == "session_meta":
            session_id = payload.get("id", session_id)
            cwd = payload.get("cwd", "")
            started_at = _parse_timestamp(payload.get("timestamp", "")) or started_at
        elif record_type == "response_item" and payload.get("type") == "message":
            msg = _extract_message(payload, timestamp)
            if msg is not None:
                messages.append(msg)

    return Session(
        id=session_id,
        agent=AgentType.CODEX,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_message(payload: dict, timestamp: datetime) -> Message | None:
    """Build a :class:`Message` from a Codex response_item payload."""
    role_str = payload.get("role", "")
    role = _MESSAGE_ROLES.get(role_str)
    if role is None:
        return None
    content_parts = payload.get("content", [])
    text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
    return Message(role=role, content=text, timestamp=timestamp)


def _parse_timestamp(raw: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a :class:`datetime`."""
    if not raw:
        return datetime.fromtimestamp(0)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
