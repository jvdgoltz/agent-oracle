"""Tests for query-time archive overview aggregations."""

from __future__ import annotations

from agent_oracle.overview_summary import summarize_overview


def test_summarize_overview_counts_and_ranks_archive_dimensions() -> None:
    """Count sessions, entities, models, and eligible conversation lengths."""
    report = summarize_overview(
        sessions=[
            {"id": "s1", "agent": "codex", "cwd": "/work/a"},
            {"id": "s2", "agent": "claude", "cwd": "/work/a"},
        ],
        entities=[
            {"entity_type": "file", "entity_value": "a.py", "session_id": "s1"},
            {"entity_type": "file", "entity_value": "a.py", "session_id": "s2"},
            {"entity_type": "package", "entity_value": "fastapi", "session_id": "s1"},
        ],
        assistant_messages=[
            {"model": "gpt", "messages": 3},
            {"model": None, "messages": 1},
        ],
        session_messages=[
            {"session_id": "s1", "timestamp": "2026-08-01T10:00:00+00:00"},
            {"session_id": "s1", "timestamp": "2026-08-01T10:02:00+00:00"},
            {"session_id": "s2", "timestamp": "2026-08-01T10:00:00+00:00"},
        ],
    )

    assert report["totals"] == {
        "sessions": 2,
        "conversation_messages": 3,
        "assistant_messages": 4,
        "average_session_messages": 1.5,
        "median_session_messages": 1.5,
        "average_session_duration_seconds": 60.0,
        "median_session_duration_seconds": 60.0,
    }
    assert report["agents"] == [
        {"agent": "claude", "sessions": 1},
        {"agent": "codex", "sessions": 1},
    ]
    assert report["projects"] == [{"cwd": "/work/a", "sessions": 2}]
    assert report["entities"] == [
        {"entity_type": "file", "entity_value": "a.py", "sessions": 2},
        {"entity_type": "package", "entity_value": "fastapi", "sessions": 1},
    ]
    assert report["models"] == [
        {"model": "gpt", "messages": 3},
        {"model": "unknown", "messages": 1},
    ]
    assert report["session_lengths"] == [
        {
            "session_id": "s1",
            "agent": "codex",
            "cwd": "/work/a",
            "messages": 2,
            "duration_seconds": 120.0,
        },
        {
            "session_id": "s2",
            "agent": "claude",
            "cwd": "/work/a",
            "messages": 1,
            "duration_seconds": 0.0,
        },
    ]


def test_summarize_overview_includes_empty_sessions() -> None:
    """Retain sessions without eligible conversation messages in the totals."""
    report = summarize_overview(
        sessions=[{"id": "empty", "agent": "codex", "cwd": "/work/a"}],
        entities=[],
        assistant_messages=[],
        session_messages=[],
    )

    assert report["totals"]["sessions"] == 1
    assert report["totals"]["conversation_messages"] == 0
    assert report["session_lengths"][0]["messages"] == 0
    assert report["session_lengths"][0]["duration_seconds"] == 0.0
