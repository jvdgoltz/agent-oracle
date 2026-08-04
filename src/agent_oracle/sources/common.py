"""Shared helpers for source normalizers.

Constants and utilities that are identical across Codex, Factory Droid, and
Claude Code session parsers.
"""

from __future__ import annotations

from datetime import datetime

from agent_oracle.models import MessageRole

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
