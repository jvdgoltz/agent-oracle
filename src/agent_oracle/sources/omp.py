"""Normalizer for Oh My Pi (OMP) session JSONL files.

OMP stores sessions at ``~/.omp/agent/sessions/<project>/<timestamp>_<uuid>.jsonl``.
Each line is a JSON object with a ``type`` field:

- ``session``: session id, cwd, timestamp, title
- ``message``: user/assistant messages with ``message.role`` and
  ``message.content[]``; the ``toolResult`` and ``bashExecution`` roles are
  agent traffic and are skipped
- Other record types (``thinking_level_change``, ``model_change``, ``custom``,
  ``title``, ``title_change``) are skipped
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Message, Session
from agent_oracle.sources.common import MESSAGE_ROLES, parse_jsonl_line, parse_timestamp

#: Roles that represent tool execution rather than conversation.
_SKIP_ROLES = {"toolResult", "bashExecution"}


def parse_omp_session(path: Path) -> Session:
    """Parse an OMP JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []

    for line in path.read_text().splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            continue
        record_type = record.get("type")

        if record_type == "session":
            session_id = record.get("id", session_id)
            cwd = record.get("cwd", "")
            ts = parse_timestamp(record.get("timestamp", ""))
            if ts != datetime.fromtimestamp(0):
                started_at = ts
        elif record_type == "message":
            timestamp = parse_timestamp(record.get("timestamp", ""))
            extracted = _extract_messages(record, timestamp)
            if extracted:
                if started_at == datetime.fromtimestamp(0):
                    started_at = timestamp
                messages.extend(extracted)

    if started_at == datetime.fromtimestamp(0) and messages:
        started_at = messages[0].timestamp

    return Session(
        id=session_id,
        agent=AgentType.OMP,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
    )


def _extract_messages(record: dict, timestamp: datetime) -> list[Message]:
    """Build one :class:`Message` per logical part of an OMP message record.

    OMP content arrays mix part types: ``thinking`` (text lives in the
    ``thinking`` field, emitted as a separate thinking message) and ``text``
    (concatenated into one message).  ``toolCall`` and ``image`` parts are
    agent traffic and are skipped, matching the convention that tool calls and
    tool results are never added to the index.
    """
    msg_data = record.get("message", {})
    role = msg_data.get("role")
    if role in _SKIP_ROLES or MESSAGE_ROLES.get(role) is None:
        return []

    message_id = record.get("id")
    model = msg_data.get("model")
    content_parts = msg_data.get("content", [])
    role_enum = MESSAGE_ROLES[role]

    def build(content: str, *, is_thinking: bool = False) -> Message:
        return Message(
            role=role_enum,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
            is_thinking=is_thinking,
            model=model,
        )

    messages: list[Message] = []
    for part in content_parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "thinking":
            thinking = part.get("thinking", "")
            if thinking:
                messages.append(build(thinking, is_thinking=True))

    text = "".join(
        p.get("text", "") for p in content_parts if isinstance(p, dict) and p.get("type") == "text"
    )
    if text:
        messages.append(build(text))
    return messages
