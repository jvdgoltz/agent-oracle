"""Normalizer for Codex (OpenAI Codex CLI) session JSONL files.

Codex stores sessions at ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``.
Each line is a JSON object with ``type`` and ``payload`` fields:

- ``session_meta``: session id, cwd, timestamp, model, git info
- ``response_item`` with ``payload.type == "message"``: user/assistant/developer messages
- ``response_item`` with ``payload.type == "reasoning"``: thinking messages
- ``turn_context``: carries the current model name per turn
- ``event_msg``: non-message events that are skipped
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Message, MessageRole, Session
from agent_oracle.sources.common import MESSAGE_ROLES, parse_jsonl_line, parse_timestamp

#: Content tags that mark a user message as injected context rather than real input.
_INJECTED_TAGS = ("<system-reminder>", "<permissions instructions>", "<collaboration_mode>")


def parse_codex_session(path: Path) -> Session:
    """Parse a Codex JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []
    current_model: str | None = None

    for line in path.read_text().splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            continue
        record_type = record.get("type")
        payload = record.get("payload", {})
        timestamp = parse_timestamp(record.get("timestamp", ""))

        if record_type == "session_meta":
            session_id = payload.get("id", session_id)
            cwd = payload.get("cwd", "")
            started_at = parse_timestamp(payload.get("timestamp", "")) or started_at
        elif record_type == "turn_context":
            current_model = payload.get("model", current_model)
        elif record_type == "response_item":
            msg = _extract_message(payload, timestamp, current_model)
            if msg is not None:
                messages.append(msg)

    return Session(
        id=session_id,
        agent=AgentType.CODEX,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_message(payload: dict, timestamp: datetime, model: str | None) -> Message | None:
    """Build a :class:`Message` from a Codex response_item payload."""
    payload_type = payload.get("type")

    if payload_type == "reasoning":
        return _extract_reasoning(payload, timestamp, model)

    if payload_type != "message":
        return None

    role_str = payload.get("role", "")
    role = MESSAGE_ROLES.get(role_str)
    if role is None:
        return None
    content_parts = payload.get("content", [])
    text = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))

    is_system = role is MessageRole.DEVELOPER
    is_injected = role is MessageRole.USER and text.startswith(_INJECTED_TAGS)

    return Message(
        role=role,
        content=text,
        timestamp=timestamp,
        model=model,
        is_system_instruction=is_system,
        is_injected=is_injected,
    )


def _extract_reasoning(payload: dict, timestamp: datetime, model: str | None) -> Message:
    """Build a thinking :class:`Message` from a Codex reasoning payload."""
    summary_parts = payload.get("summary", [])
    text = "".join(part.get("text", "") for part in summary_parts if isinstance(part, dict))
    return Message(
        role=MessageRole.ASSISTANT,
        content=text,
        timestamp=timestamp,
        model=model,
        is_thinking=True,
    )
