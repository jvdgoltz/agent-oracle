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

import sqlite3
from contextlib import closing
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


def find_codex_session_path(session_id: str, sessions_root: Path | None = None) -> Path | None:
    """Find an active or archived Codex JSONL by its stable thread ID."""
    roots = (
        (sessions_root,)
        if sessions_root is not None
        else (
            Path.home() / ".codex" / "sessions",
            Path.home() / ".codex" / "archived_sessions",
        )
    )
    for root in roots:
        for path in root.rglob(f"*-{session_id}.jsonl"):
            return path
    return None


def is_codex_session_archived(session_id: str) -> bool:
    """Return whether Codex moved a thread into its archived session directory."""
    path = find_codex_session_path(session_id)
    archive_root = Path.home() / ".codex" / "archived_sessions"
    return path is not None and path.is_relative_to(archive_root)


def archived_codex_session_ids(session_ids: set[str]) -> set[str]:
    """Return candidate thread IDs whose JSONL is in the Codex archive."""
    archive_root = Path.home() / ".codex" / "archived_sessions"
    filenames = [path.name for path in archive_root.rglob("*.jsonl")]
    return {
        session_id
        for session_id in session_ids
        if any(name.endswith(f"-{session_id}.jsonl") for name in filenames)
    }


def load_codex_session(session_id: str, sessions_root: Path | None = None) -> Session | None:
    """Load one active or archived Codex session by its stable thread ID."""
    path = find_codex_session_path(session_id, sessions_root)
    if path is not None:
        session = parse_codex_session(path)
        if session.id == session_id:
            return session
    return None


def parse_codex_session(path: Path) -> Session:  # noqa: C901
    """Parse a Codex JSONL session file into a :class:`Session`."""
    session_id = path.stem
    cwd = ""
    started_at = datetime.fromtimestamp(0)
    messages: list[Message] = []
    interruptions: list[Interruption] = []
    current_model: str | None = None
    parent_thread_id: str | None = None
    is_review_agent = False
    token_usages: list[TokenUsage] = []

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
            parent_thread_id = payload.get("parent_thread_id")
            source = payload.get("source")
            is_review_agent = (
                isinstance(source, dict)
                and isinstance(source.get("subagent"), dict)
                and source["subagent"].get("other") == "guardian"
            )
        elif record_type == "turn_context":
            current_model = payload.get("model", current_model)
        elif (
            record_type == "event_msg"
            and payload.get("type") == "turn_aborted"
            and payload.get("reason") == "interrupted"
            and payload.get("turn_id")
        ):
            interruptions.append(
                Interruption(
                    source_id=str(payload["turn_id"]),
                    timestamp=timestamp,
                    model=current_model,
                    user_message_seq=_last_user_message_seq(messages),
                )
            )
        elif record_type == "event_msg" and payload.get("type") == "token_count":
            info = (payload.get("info") or {}).get("last_token_usage", {})
            if info:
                token_usages.append(
                    TokenUsage(
                        timestamp=timestamp,
                        model=current_model,
                        input_tokens=info.get("input_tokens"),
                        output_tokens=info.get("output_tokens"),
                        cached_input_tokens=info.get("cached_input_tokens"),
                        reasoning_output_tokens=info.get("reasoning_output_tokens"),
                        total_tokens=info.get("total_tokens"),
                    )
                )
        elif record_type == "response_item":
            msg = _extract_message(payload, timestamp, current_model)
            if msg is not None:
                messages.append(msg)

    if started_at == datetime.fromtimestamp(0) and messages:
        started_at = messages[0].timestamp

    return Session(
        id=session_id,
        title=_read_codex_title(path, session_id),
        agent=AgentType.CODEX,
        cwd=cwd,
        started_at=started_at,
        messages=messages,
        interruptions=interruptions,
        token_usages=token_usages,
        parent_thread_id=parent_thread_id,
        is_review_agent=is_review_agent,
    )


def _read_codex_title(path: Path, session_id: str) -> str | None:
    """Read the latest Codex thread name, falling back to state metadata."""
    sessions_root = next(
        (parent for parent in path.parents if parent.name in {"sessions", "archived_sessions"}),
        None,
    )
    if sessions_root is None:
        return None
    codex_home = sessions_root.parent
    index_path = codex_home / "session_index.jsonl"
    title = None
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            record = parse_jsonl_line(line)
            if record is not None and record.get("id") == session_id:
                title = normalize_title(record.get("thread_name")) or title
    if title is not None:
        return title
    state_path = codex_home / "state_5.sqlite"
    if not state_path.exists():
        return None
    try:
        with closing(sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)) as connection:
            row = connection.execute(
                "SELECT name, title FROM threads WHERE id = ?", (session_id,)
            ).fetchone()
    except sqlite3.Error:
        return None
    return normalize_title(row[0]) or normalize_title(row[1]) if row is not None else None


def _last_user_message_seq(messages: list[Message]) -> int | None:
    """Return the latest real user message sequence for an interruption."""
    for seq in range(len(messages) - 1, -1, -1):
        message = messages[seq]
        if message.role is MessageRole.USER and not message.is_injected:
            return seq
    return None


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

    # Model is only for assistant messages (including reasoning/thinking)
    msg_model = model if role is MessageRole.ASSISTANT else None
    is_system = role is MessageRole.DEVELOPER
    is_injected = role is MessageRole.USER and is_injected_message(text)

    return Message(
        role=role,
        content=text,
        timestamp=timestamp,
        model=msg_model,
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
