"""Normalizer for Factory Droid session JSONL files.

Factory stores sessions at ``~/.factory/sessions/<project>/<uuid>.jsonl``.
Each line is a JSON object:

- ``session_start``: session id, cwd, title, owner
- ``message``: user/assistant messages with ``message.role`` and ``message.content[]``
- Other record types (``settings_update``, etc.) are skipped.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Interruption, Message, MessageRole, Session
from agent_oracle.sources.common import MESSAGE_ROLES, parse_jsonl_line, parse_timestamp


def parse_factory_session(path: Path) -> Session:
    """Parse a Factory Droid JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []
    interruptions: list[Interruption] = []
    model = _read_factory_session_model(path)

    for line in path.read_text().splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            continue
        record_type = record.get("type")

        if record_type == "session_start":
            session_id = record.get("id", session_id)
            cwd = record.get("cwd", "")
            ts = parse_timestamp(record.get("timestamp", ""))
            if ts != datetime.fromtimestamp(0):
                started_at = ts
        elif record_type == "message":
            timestamp = parse_timestamp(record.get("timestamp", ""))
            messages.extend(_extract_messages(record, timestamp, model))
            if started_at == datetime.fromtimestamp(0) and messages:
                started_at = timestamp
        elif record_type == "agent_turn_outcome" and record.get("reason") == "cancelled":
            interruption = _interruption_from_outcome(record, messages, model)
            if interruption is not None:
                interruptions.append(interruption)

    if started_at == datetime.fromtimestamp(0) and messages:
        started_at = messages[0].timestamp

    return Session(
        id=session_id,
        agent=AgentType.FACTORY,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
        interruptions=_deduplicate_interruptions(interruptions),
    )


def _interruption_from_outcome(
    record: dict, messages: list[Message], model: str | None
) -> Interruption | None:
    """Link a cancelled Factory outcome to its preceding real user prompt."""
    turn_id = record.get("turnId")
    if not turn_id:
        return None
    source_id = str(turn_id)
    return Interruption(
        source_id=source_id,
        timestamp=None,
        model=model,
        user_message_seq=_last_user_message_seq(messages),
    )


def _last_user_message_seq(messages: list[Message]) -> int | None:
    """Return the latest visible user prompt before an outcome record."""
    for sequence in range(len(messages) - 1, -1, -1):
        message = messages[sequence]
        if message.role is MessageRole.USER and not message.is_injected:
            return sequence
    return None


def _deduplicate_interruptions(interruptions: list[Interruption]) -> list[Interruption]:
    """Keep one Factory interruption per cancelled user turn."""
    return list({interruption.source_id: interruption for interruption in interruptions}.values())


def _extract_messages(record: dict, timestamp: datetime, model: str | None) -> list[Message]:
    """Build one :class:`Message` per logical part of a Factory message record.

    Factory content arrays mix reasoning (thinking) and text parts. Thinking
    parts store their text in the 'thinking' field and are emitted as separate
    thinking messages. The session model (from .settings.json) is attached to
    ALL assistant messages, including thinking messages.
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
        # Model is for assistant messages (including thinking)
        msg_model = model if role is MessageRole.ASSISTANT else None
        return Message(
            role=role,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
            is_thinking=is_thinking,
            is_injected=is_injected,
            is_system_instruction=is_system,
            model=msg_model,
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


def _read_factory_session_model(path: Path) -> str | None:
    """Read the model identifier from a Factory .settings.json companion file.

    Factory Droid stores session metadata (including the model) in a sibling
    ``<uuid>.settings.json`` file. The model field is at the root:
    ``{\"model\": \"custom:provider:model-name\", ...}``.

    Returns None if the settings file is missing or unparseable.
    """
    settings_path = path.with_suffix(".settings.json")
    if not settings_path.exists():
        return None
    try:
        import json

        settings = json.loads(settings_path.read_text())
        return settings.get("model")
    except (json.JSONDecodeError, OSError, KeyError):
        return None
