"""Tests for embedded Codex payload translation."""

from types import SimpleNamespace

import pytest

from agent_oracle.agent_payload import source_messages, validated_image_data_url


def test_validated_image_data_url_accepts_png() -> None:
    """A matching PNG data URL passes through unchanged."""
    value = "data:image/png;base64,iVBORw0KGgo="

    assert validated_image_data_url({"image_data_url": value}) == value


def test_validated_image_data_url_rejects_mismatched_content() -> None:
    """Declared image types must match their decoded signature."""
    with pytest.raises(ValueError, match="does not match"):
        validated_image_data_url({"image_data_url": "data:image/png;base64,R0lGODlh"})


def test_source_messages_translates_archive_fields() -> None:
    """Source messages become session-detail dictionaries."""
    message = SimpleNamespace(
        role="user",
        content="Inspect",
        timestamp=SimpleNamespace(isoformat=lambda: "2026-08-25T12:00:00+00:00"),
        is_thinking=False,
        model="gpt-5",
        is_system_instruction=False,
        is_injected=False,
    )

    assert source_messages([message], "thread")[0] == {
        "id": 0,
        "session_id": "thread",
        "role": "user",
        "content": "Inspect",
        "timestamp": "2026-08-25T12:00:00+00:00",
        "seq": 0,
        "is_thinking": 0,
        "model": "gpt-5",
        "is_system_instruction": 0,
        "is_injected": 0,
    }
