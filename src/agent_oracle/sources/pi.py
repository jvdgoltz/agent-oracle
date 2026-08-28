"""Normalizer for Pi coding-agent session JSONL files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Interruption, Message, MessageRole, Session, TokenUsage
from agent_oracle.sources.common import (
    MESSAGE_ROLES,
    is_injected_message,
    parse_jsonl_line,
    parse_timestamp,
)


def parse_pi_session(path: Path) -> Session:
    """Parse a Pi session file into the shared :class:`Session` model."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    title: str | None = None
    messages: list[Message] = []
    interruptions: list[Interruption] = []
    token_usages: list[TokenUsage] = []

    for line in path.read_text().splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            continue
        timestamp = parse_timestamp(record.get("timestamp", ""))
        if record.get("type") == "session":
            session_id = record.get("id", session_id)
            cwd = record.get("cwd", "")
            started_at = timestamp
        elif record.get("type") == "session_info":
            title = record.get("name") or title
        elif record.get("type") == "message":
            usage = record.get("message", {}).get("usage") or record.get("usage") or {}
            if usage:
                token_usages.append(
                    TokenUsage(
                        timestamp=timestamp,
                        model=record.get("message", {}).get("model"),
                        input_tokens=_value(usage, "input_tokens", "input"),
                        output_tokens=_value(usage, "output_tokens", "output"),
                        cached_input_tokens=usage.get("cached_input_tokens"),
                        cache_creation_input_tokens=_value(
                            usage, "cache_creation_input_tokens", "cacheWrite"
                        ),
                        cache_read_input_tokens=_value(
                            usage, "cache_read_input_tokens", "cacheRead"
                        ),
                        reasoning_output_tokens=_value(
                            usage, "reasoning_output_tokens", "reasoning"
                        ),
                        total_tokens=_value(usage, "total_tokens", "totalTokens"),
                    )
                )
            message, interruption = _parse_message(record, timestamp, messages)
            messages.extend(message)
            if interruption is not None:
                interruptions.append(interruption)

    if started_at == datetime.fromtimestamp(0) and messages:
        started_at = messages[0].timestamp
    return Session(
        id=session_id,
        agent=AgentType.PI,
        cwd=cwd,
        started_at=started_at,
        title=title,
        messages=messages,
        interruptions=interruptions,
        token_usages=token_usages,
    )


def _last_user_message_seq(messages: list[Message]) -> int | None:
    """Return the latest real user-message sequence."""
    for seq in range(len(messages) - 1, -1, -1):
        if messages[seq].role is MessageRole.USER:
            return seq
    return None


def _parse_message(
    record: dict, timestamp: datetime, messages: list[Message]
) -> tuple[list[Message], Interruption | None]:
    """Parse one Pi message entry, excluding tool traffic."""
    message = record.get("message", {})
    role = message.get("role")
    role_enum = MESSAGE_ROLES.get(role)
    if role_enum is None:
        return [], None
    interruption = None
    if (
        role == "assistant"
        and message.get("stopReason") == "aborted"
        and message.get("errorMessage") == "Interrupted by user"
        and record.get("id")
    ):
        interruption = Interruption(
            source_id=str(record["id"]),
            timestamp=timestamp,
            model=message.get("model"),
            user_message_seq=_last_user_message_seq(messages),
        )
    content = message.get("content", "")
    parts = [(content, False)] if isinstance(content, str) else _content_parts(content)
    return [
        Message(
            role=role_enum,
            content=text,
            timestamp=timestamp,
            message_id=record.get("id"),
            model=message.get("model"),
            is_thinking=is_thinking,
            is_injected=role_enum is MessageRole.USER and is_injected_message(text),
        )
        for text, is_thinking in parts
        if text
    ], interruption


def _content_parts(content: list) -> list[tuple[str, bool]]:
    """Return Pi thinking and visible text parts, excluding tool blocks."""
    parts = [
        (part.get("thinking", ""), True)
        for part in content
        if isinstance(part, dict) and part.get("type") == "thinking"
    ]
    parts.append(
        (
            "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ),
            False,
        )
    )
    return parts


def _value(values: dict, snake: str, camel: str) -> int | None:
    """Read a usage field while preserving a reported zero."""
    return values[snake] if snake in values else values.get(camel)


def _value(values: dict, snake: str, camel: str) -> int | None:
    """Read a usage field while preserving a reported zero."""
    return values[snake] if snake in values else values.get(camel)
