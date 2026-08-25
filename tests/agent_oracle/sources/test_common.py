"""Tests for shared source-normalizer classification helpers."""

import pytest

from agent_oracle.sources.common import is_injected_message


@pytest.mark.parametrize(
    "content",
    [
        "# AGENTS.md instructions for /tmp/project",
        "<environment_context>",
        "<system-reminder>",
        "<permissions instructions>",
        "<collaboration_mode>",
        "<recommended_plugins>",
        "<turn_aborted>",
        '<codex_internal_context source="goal">',
        "<codex_delegation>",
        "<realtime_delegation>",
        "<subagent_notification>",
        "# Task Tool Invocation",
        "### Session update [in progress — more steps follow]",
        "<system-notification>",
    ],
)
def test_generated_prefixes_are_injected(content: str) -> None:
    """Every known generated envelope marker is classified at column zero."""
    assert is_injected_message(content + "\nbody")


@pytest.mark.parametrize(
    "content",
    [
        "I saw <system-reminder> in a transcript",
        " # AGENTS.md instructions for /tmp/project",
        "<context>\nreal user request",
        "<attachment>\nuser-provided evidence",
    ],
)
def test_user_content_markers_are_not_overmatched(content: str) -> None:
    """Prose, indented markers, context, and attachments remain ordinary input."""
    assert not is_injected_message(content)
