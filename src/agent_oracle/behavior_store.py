"""SQLite queries that provide messages for behavior statistics."""

from __future__ import annotations

import sqlite3
from datetime import date


def list_behavior_messages(
    connection: sqlite3.Connection,
    *,
    agent: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> list[dict]:
    """Return real user messages with session fields for behavior statistics."""
    clauses = [
        "m.role = 'user'",
        "m.is_thinking = 0",
        "m.is_system_instruction = 0",
        "m.is_injected = 0",
    ]
    params: list[str] = []
    if agent is not None:
        clauses.append("s.agent = ?")
        params.append(agent)
    if start is not None:
        clauses.append("date(m.timestamp) >= date(?)")
        params.append(start.isoformat())
    if end is not None:
        clauses.append("date(m.timestamp) <= date(?)")
        params.append(end.isoformat())
    rows = connection.execute(
        "SELECT m.content, m.timestamp, m.is_injected, m.is_interrupted, s.agent, s.cwd, "
        "COALESCE(m.interruption_model, (SELECT a.model FROM messages a "
        "WHERE a.session_id = m.session_id AND a.role = 'assistant' AND a.seq < m.seq "
        "AND a.is_thinking = 0 AND a.is_system_instruction = 0 AND a.is_injected = 0 "
        "ORDER BY a.seq DESC, a.id DESC LIMIT 1), 'unknown') AS model "
        "FROM messages m JOIN sessions s ON s.id = m.session_id WHERE "
        + " AND ".join(clauses)
        + " ORDER BY m.timestamp",
        params,
    ).fetchall()
    return [dict(row) for row in rows]
