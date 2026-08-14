"""Query-time aggregations for archive overview statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Any


def _duration_seconds(timestamps: list[str]) -> float:
    """Return elapsed seconds between the first and last ISO timestamps."""
    if len(timestamps) < 2:
        return 0.0
    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in timestamps]
    return (max(parsed) - min(parsed)).total_seconds()


def _count_rows(counts: Counter[str], key: str) -> list[dict[str, str | int]]:
    """Return count rows ranked by count then key."""
    return [
        {key: value, "sessions" if key != "model" else "messages": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def summarize_overview(
    *,
    sessions: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    assistant_messages: list[dict[str, Any]],
    session_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize archive sessions, entities, assistant models, and lengths."""
    by_session: dict[str, list[str]] = defaultdict(list)
    for message in session_messages:
        by_session[str(message["session_id"])].append(str(message["timestamp"]))

    lengths = []
    for session in sessions:
        timestamps = by_session[str(session["id"])]
        lengths.append(
            {
                "session_id": session["id"],
                "agent": session["agent"],
                "cwd": session["cwd"],
                "messages": len(timestamps),
                "duration_seconds": _duration_seconds(timestamps),
            }
        )
    lengths.sort(
        key=lambda row: (
            -int(row["messages"]),
            -float(row["duration_seconds"]),
            str(row["session_id"]),
        )
    )

    entity_sessions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entity in entities:
        entity_sessions[(str(entity["entity_type"]), str(entity["entity_value"]))].add(
            str(entity["session_id"])
        )
    entity_rows = [
        {"entity_type": entity_type, "entity_value": value, "sessions": len(session_ids)}
        for (entity_type, value), session_ids in entity_sessions.items()
    ]
    entity_rows.sort(
        key=lambda row: (
            -int(row["sessions"]),
            str(row["entity_type"]),
            str(row["entity_value"]),
        )
    )

    agents = Counter(str(session["agent"]) for session in sessions)
    projects = Counter(str(session["cwd"]) for session in sessions)
    models = Counter(
        {
            str(message["model"] or "unknown"): int(message["messages"])
            for message in assistant_messages
        }
    )
    message_counts = [int(row["messages"]) for row in lengths]
    durations = [float(row["duration_seconds"]) for row in lengths]
    assistant_count = sum(models.values())
    return {
        "totals": {
            "sessions": len(sessions),
            "conversation_messages": sum(message_counts),
            "assistant_messages": assistant_count,
            "average_session_messages": sum(message_counts) / len(sessions) if sessions else 0.0,
            "median_session_messages": float(median(message_counts)) if message_counts else 0.0,
            "average_session_duration_seconds": sum(durations) / len(sessions) if sessions else 0.0,
            "median_session_duration_seconds": float(median(durations)) if durations else 0.0,
        },
        "entities": entity_rows,
        "models": _count_rows(models, "model"),
        "agents": _count_rows(agents, "agent"),
        "projects": _count_rows(projects, "cwd"),
        "session_lengths": lengths,
    }
