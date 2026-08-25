"""Read archived sessions and entities from SQLite."""

from __future__ import annotations

import sqlite3


def list_sessions(
    connection: sqlite3.Connection,
    limit: int,
    offset: int,
    *,
    include_review_agents: bool,
) -> list[dict]:
    """Return sessions ordered by start time with pagination."""
    rows = connection.execute(
        """
        SELECT id, agent, cwd, title, started_at, summary, enriched,
               parent_thread_id, is_review_agent
        FROM sessions
        WHERE ? OR NOT is_review_agent
        ORDER BY started_at DESC
        LIMIT ? OFFSET ?
        """,
        (include_review_agents, limit, offset),
    ).fetchall()
    return [dict(row) for row in rows]


def list_review_sessions(
    connection: sqlite3.Connection, parent_thread_ids: list[str]
) -> dict[str, list[dict]]:
    """Return review summaries grouped by parent Codex thread ID."""
    if not parent_thread_ids:
        return {}
    placeholders = ",".join("?" for _ in parent_thread_ids)
    rows = connection.execute(
        "SELECT id, agent, cwd, title, started_at, summary, parent_thread_id "
        "FROM sessions WHERE is_review_agent "
        f"AND parent_thread_id IN ({placeholders}) ORDER BY started_at",
        parent_thread_ids,
    ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        review = dict(row)
        grouped.setdefault(review["parent_thread_id"], []).append(review)
    return grouped


def get_session(connection: sqlite3.Connection, session_id: str) -> dict | None:
    """Return one session and its ordered messages."""
    row = connection.execute(
        "SELECT id, agent, cwd, title, started_at, summary, enriched, parent_thread_id, "
        "is_review_agent FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    messages = connection.execute(
        "SELECT id, session_id, role, content, timestamp, seq, "
        "is_thinking, model, is_system_instruction, is_injected "
        "FROM messages WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    result["messages"] = [dict(message) for message in messages]
    return result


def get_entities(connection: sqlite3.Connection, session_id: str) -> list[dict]:
    """Return all entities for one session."""
    rows = connection.execute(
        "SELECT entity_type, entity_value FROM entities WHERE session_id = ?", (session_id,)
    ).fetchall()
    return [
        {"entity_type": row["entity_type"], "entity_value": row["entity_value"]} for row in rows
    ]


def list_entities(connection: sqlite3.Connection, session_ids: list[str]) -> dict[str, list[dict]]:
    """Return entities keyed by session ID."""
    if not session_ids:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    rows = connection.execute(
        "SELECT session_id, entity_type, entity_value FROM entities "
        f"WHERE session_id IN ({placeholders})",
        session_ids,
    ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["session_id"], []).append(
            {"entity_type": row["entity_type"], "entity_value": row["entity_value"]}
        )
    return grouped
