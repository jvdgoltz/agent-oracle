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

from agent_oracle.models import AgentType, Interruption, Message, MessageRole, Session, TokenUsage
from agent_oracle.sources.common import (
    MESSAGE_ROLES,
    is_injected_message,
    normalize_title,
    parse_jsonl_line,
    parse_timestamp,
)


def parse_omp_session(path: Path) -> Session:  # noqa: C901
    """Parse an OMP JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []
    interruptions: list[Interruption] = []
    title_slot: str | None = None
    header_title: str | None = None
    changed_title: str | None = None
    token_usages: list[TokenUsage] = []

    for line in path.read_text().splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            continue
        record_type = record.get("type")
        title_slot, header_title, changed_title = _update_titles(
            record, title_slot, header_title, changed_title
        )

        if record_type == "session":
            session_id = record.get("id", session_id)
            cwd = record.get("cwd", "")
            ts = parse_timestamp(record.get("timestamp", ""))
            if ts != datetime.fromtimestamp(0):
                started_at = ts
        elif record_type == "message":
            timestamp = parse_timestamp(record.get("timestamp", ""))
            message_data = record.get("message", {})
            usage = message_data.get("usage") or record.get("usage") or {}
            if usage:
                token_usages.append(_token_usage(timestamp, message_data.get("model"), usage))
            if (
                message_data.get("role") == "assistant"
                and message_data.get("stopReason") == "aborted"
                and message_data.get("errorMessage") == "Interrupted by user"
                and record.get("id")
            ):
                interruptions.append(
                    Interruption(
                        source_id=str(record["id"]),
                        timestamp=timestamp,
                        model=message_data.get("model"),
                        user_message_seq=_last_user_message_seq(messages),
                    )
                )
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
        title=title_slot or changed_title or header_title,
        messages=messages,
        interruptions=interruptions,
        token_usages=token_usages,
    )


def _update_titles(
    record: dict,
    title_slot: str | None,
    header_title: str | None,
    changed_title: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Apply one OMP title-bearing record to accumulated title state."""
    record_type = record.get("type")
    if record_type == "title":
        title_slot = normalize_title(record.get("title")) or title_slot
    elif record_type == "session":
        header_title = normalize_title(record.get("title")) or header_title
    elif record_type == "title_change":
        changed_title = normalize_title(record.get("title")) or changed_title
    return title_slot, header_title, changed_title


def _last_user_message_seq(messages: list[Message]) -> int | None:
    """Return the latest real user message sequence for an interruption."""
    for seq in range(len(messages) - 1, -1, -1):
        if messages[seq].role is MessageRole.USER:
            return seq
    return None


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
    if MESSAGE_ROLES.get(role) is None:
        return []

    message_id = record.get("id")
    model = msg_data.get("model")
    content_parts = msg_data.get("content", [])
    role_enum = MESSAGE_ROLES[role]
    text = "".join(
        p.get("text", "") for p in content_parts if isinstance(p, dict) and p.get("type") == "text"
    )

    def build(content: str, *, is_thinking: bool = False) -> Message:
        return Message(
            role=role_enum,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
            is_thinking=is_thinking,
            model=model,
            is_injected=role_enum is MessageRole.USER and is_injected_message(text),
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

    if text:
        messages.append(build(text))
    return messages


def _token_usage(timestamp: datetime, model: str | None, usage: dict) -> TokenUsage:
    """Normalize provider usage fields found in an OMP response."""
    return TokenUsage(
        timestamp=timestamp,
        model=model,
        input_tokens=_value(usage, "input_tokens", "input"),
        output_tokens=_value(usage, "output_tokens", "output"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        cache_creation_input_tokens=_value(usage, "cache_creation_input_tokens", "cacheWrite"),
        cache_read_input_tokens=_value(usage, "cache_read_input_tokens", "cacheRead"),
        reasoning_output_tokens=_value(usage, "reasoning_output_tokens", "reasoning"),
        total_tokens=_value(usage, "total_tokens", "totalTokens"),
    )


def _value(values: dict, snake: str, camel: str) -> int | None:
    """Read a usage field while preserving a reported zero."""
    return values[snake] if snake in values else values.get(camel)
