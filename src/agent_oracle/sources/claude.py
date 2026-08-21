"""Normalizer for Claude Code session JSONL files.

Claude stores sessions at ``~/.claude/projects/<project>/<uuid>.jsonl``.
Each line is a JSON object with ``type`` field:

- ``user`` / ``assistant``: messages with ``message.role`` and ``message.content``
  (content can be a string or an array of ``{"type": "text", "text": "..."}``)
- ``queue-operation``, ``attachment``, etc.: skipped

Session metadata (sessionId, cwd) is extracted from the first message record.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_oracle.models import AgentType, Interruption, Message, MessageRole, Session
from agent_oracle.sources.common import (
    MESSAGE_ROLES,
    normalize_title,
    parse_jsonl_line,
    parse_timestamp,
)

_MESSAGE_TYPES = {"user", "assistant"}


def parse_claude_session(path: Path) -> Session:
    """Parse a Claude Code JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []
    interruptions: list[Interruption] = []
    assistant_models: dict[str, str | None] = {}
    title: str | None = None

    for line in path.read_text().splitlines():
        record = parse_jsonl_line(line)
        if record is None:
            continue
        record_type = record.get("type", "")

        if record_type == "custom-title":
            title = normalize_title(record.get("customTitle")) or title

        if record_type not in _MESSAGE_TYPES:
            continue

        session_id = record.get("sessionId", session_id)
        cwd = record.get("cwd", cwd) or cwd
        timestamp = parse_timestamp(record.get("timestamp", ""))
        if started_at == datetime.fromtimestamp(0):
            started_at = timestamp

        interrupted_id = record.get("interruptedMessageId")
        if interrupted_id and _is_interruption_sentinel(record):
            interruptions.append(
                Interruption(
                    source_id=str(interrupted_id),
                    timestamp=timestamp,
                    model=assistant_models.get(str(interrupted_id)),
                    user_message_seq=_last_user_message_seq(messages),
                )
            )
            continue
        if record_type == "assistant" and record.get("message", {}).get("id"):
            assistant_models[str(record["message"]["id"])] = record["message"].get("model")
        msg = _extract_messages(record, timestamp)
        messages.extend(msg)

    return Session(
        id=session_id,
        agent=AgentType.CLAUDE,
        cwd=cwd,
        started_at=started_at,
        title=title,
        messages=messages,
        interruptions=_deduplicate_interruptions(interruptions),
    )


def _is_interruption_sentinel(record: dict) -> bool:
    """Return True for Claude's explicit manual-interruption user record."""
    content = record.get("message", {}).get("content")
    marker = "[Request interrupted by user]"
    return content == marker or content == [{"type": "text", "text": marker}]


def _last_user_message_seq(messages: list[Message]) -> int | None:
    """Return the latest real user message sequence for an interruption."""
    for seq in range(len(messages) - 1, -1, -1):
        if messages[seq].role is MessageRole.USER and not messages[seq].is_injected:
            return seq
    return None


def _deduplicate_interruptions(interruptions: list[Interruption]) -> list[Interruption]:
    """Keep one Claude interruption per interrupted assistant API message."""
    return list({interruption.source_id: interruption for interruption in interruptions}.values())


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
