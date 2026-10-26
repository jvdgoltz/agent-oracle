"""Shared helpers for source normalizers.

Constants and utilities that are identical across Codex, Factory Droid, and
Claude Code session parsers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from agent_oracle.models import MessageRole

logger = logging.getLogger(__name__)

#: Maps raw role strings from any agent to the unified :class:`MessageRole`.
MESSAGE_ROLES: dict[str, MessageRole] = {
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "developer": MessageRole.DEVELOPER,
    "system": MessageRole.SYSTEM,
}


def parse_timestamp(raw: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a :class:`datetime`."""
    if not raw:
        return datetime.fromtimestamp(0)
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def normalize_title(value: object) -> str | None:
    """Return a stripped non-empty title string, or None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_jsonl_line(line: str) -> dict | None:
    """Parse one JSONL line, returning None for blank or truncated lines.

    Session files are appended to live, so the last line can be a partial
    write.  Rather than crashing the entire file parse, skip the bad line
    and log it at debug level.
    """
    if not line.strip():
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        logger.debug("Skipping malformed JSONL line: %s", line[:120])
        return None


def iter_lines(path: Path) -> Iterator[str]:
    """Read source text one line at a time without duplicating the whole file."""
    with path.open(encoding="utf-8") as source:
        yield from source
