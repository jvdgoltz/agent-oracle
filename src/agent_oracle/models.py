"""Core data models for Agent Oracle.

Every source normalizer produces instances
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
    OMP = "omp"
    PI = "pi"


class MessageRole(StrEnum):
    """Roles a message can have across all supported agents."""

    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Message:
    """A single message within a session.

    The ``is_thinking``, ``is_system_instruction``, and ``is_injected`` flags
    distinguish real conversation turns from internal reasoning and injected
    context.  Messages with any of these flags set are visible in the session
    detail view but excluded from the search index.
    """

    role: MessageRole
    content: str
    timestamp: datetime
    message_id: str | None = None
    is_thinking: bool = False
    model: str | None = None
    is_system_instruction: bool = False
    is_injected: bool = False

    @property
    def is_searchable(self) -> bool:
        """Return True when this message should appear in the search index."""
        return self.role in (MessageRole.USER, MessageRole.ASSISTANT) and not (
            self.is_thinking or self.is_system_instruction or self.is_injected
        )


@dataclass(frozen=True, slots=True)
class Interruption:
    """An explicit source interruption linked to a session user-message sequence."""

    source_id: str
    timestamp: datetime | None
    model: str | None
    user_message_seq: int | None


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported token counts for one model response."""

    timestamp: datetime
    model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Session:
    """A coding agent session with its messages."""

    id: str
    agent: AgentType
    cwd: str
    started_at: datetime
    messages: list[Message] = field(default_factory=list)
    interruptions: list[Interruption] = field(default_factory=list)
    token_usages: list[TokenUsage] = field(default_factory=list)
    title: str | None = None
    parent_thread_id: str | None = None
    is_review_agent: bool = False

    @property
    def interruption_models(self) -> dict[int, str | None]:
        """Return interrupted user-message sequences keyed to their assistant model."""
        return {
            item.user_message_seq: item.model
            for item in self.interruptions
            if item.user_message_seq is not None
        }
