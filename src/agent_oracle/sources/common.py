"""Shared helpers for source normalizers.

Constants and utilities that are identical across Codex, Factory Droid, and
Claude Code session parsers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

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
