"""SQLite queries that provide rows for archive overview statistics."""

from __future__ import annotations

import sqlite3
from datetime import date


def list_overview_rows(
    connection: sqlite3.Connection,
    *,
    agent: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, list[dict]]:
    """Return scoped sessions and visible-message rows for overview statistics."""
    clauses = ["1 = 1"]
    params: list[str] = []
    if agent is not None:
        clauses.append("s.agent = ?")
        params.append(agent)
    if start is not None:
        clauses.append("date(s.started_at) >= date(?)")
        params.append(start.isoformat())
    if end is not None:
        clauses.append("date(s.started_at) <= date(?)")
        params.append(end.isoformat())
    where = " AND ".join(clauses)
    visible = (
        "m.is_thinking = 0 AND m.is_system_instruction = 0 AND m.is_injected = 0 "
        "AND m.role IN ('user', 'assistant')"
    )
    session_rows = connection.execute(
        f"SELECT s.id, s.agent, s.cwd FROM sessions s WHERE {where}", params
    ).fetchall()
    entity_rows = connection.execute(
        "SELECT e.session_id, e.entity_type, e.entity_value FROM entities e "
        f"JOIN sessions s ON s.id = e.session_id WHERE {where}",
        params,
    ).fetchall()
    message_rows = connection.execute(
        "SELECT m.session_id, m.timestamp FROM messages m "
        f"JOIN sessions s ON s.id = m.session_id WHERE {where} AND {visible}",
        params,
    ).fetchall()
    assistant_rows = connection.execute(
        "SELECT m.model, COUNT(*) AS messages FROM messages m "
        f"JOIN sessions s ON s.id = m.session_id WHERE {where} AND {visible} "
        "AND m.role = 'assistant' GROUP BY m.model",
        params,
    ).fetchall()
    return {
        "sessions": [dict(row) for row in session_rows],
        "entities": [dict(row) for row in entity_rows],
        "session_messages": [dict(row) for row in message_rows],
        "assistant_messages": [dict(row) for row in assistant_rows],
    }
