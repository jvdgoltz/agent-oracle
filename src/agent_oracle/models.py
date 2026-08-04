"""Core data models for Agent Oracle.

Every source normalizer (Codex, Factory Droid, Claude Code) produces instances
of :class:`Session` containing a list of :class:`Message` records.  This shared
shape is what the store, embeddings, and API consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class AgentType(StrEnum):
    """The coding agents whose sessions Agent Oracle archives."""

    CODEX = "codex"
    FACTORY = "factory"
    CLAUDE = "claude"


class MessageRole(StrEnum):
    """Roles a message can have across all three agents."""

    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Message:
    """A single message within a session."""

    role: MessageRole
    content: str
    timestamp: datetime
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """A coding agent session with its messages."""

    id: str
    agent: AgentType
    cwd: str
    started_at: datetime
    messages: list[Message] = field(default_factory=list)
