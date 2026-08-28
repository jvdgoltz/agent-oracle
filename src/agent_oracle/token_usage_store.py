"""Query provider-reported token usage from SQLite."""

from __future__ import annotations

import sqlite3
from datetime import date
from typing import Any

from agent_oracle.models import Session


def list_token_usage(
    connection: sqlite3.Connection,
    *,
    agent: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> list[dict]:
    """Return token usage grouped by agent and model."""
    if not _has_token_usage_table(connection):
        return []
    clauses = ["NOT s.is_review_agent"]
    params: list[Any] = []
    if agent:
        clauses.append("s.agent = ?")
        params.append(agent)
    if start:
        clauses.append("date(u.timestamp) >= ?")
        params.append(start.isoformat())
    if end:
        clauses.append("date(u.timestamp) <= ?")
        params.append(end.isoformat())
    rows = connection.execute(
        f"""
        SELECT s.agent, COALESCE(u.model, 'unknown') AS model,
          COUNT(*) AS responses,
          SUM(u.input_tokens) AS input_tokens, SUM(u.output_tokens) AS output_tokens,
          SUM(u.cached_input_tokens) AS cached_input_tokens,
          SUM(u.cache_creation_input_tokens) AS cache_creation_input_tokens,
          SUM(u.cache_read_input_tokens) AS cache_read_input_tokens,
          SUM(u.reasoning_output_tokens) AS reasoning_output_tokens,
          SUM(u.total_tokens) AS total_tokens
        FROM token_usage u JOIN sessions s ON s.id = u.session_id
        WHERE {" AND ".join(clauses)}
        GROUP BY s.agent, u.model ORDER BY total_tokens DESC NULLS LAST
        """,
        params,
    ).fetchall()
    return [dict(row) for row in rows]


def replace_token_usage(connection: sqlite3.Connection, session: Session) -> None:
    """Replace one indexed session's usage rows after the explicit migration."""
    if not _has_token_usage_table(connection):
        return
    connection.execute("DELETE FROM token_usage WHERE session_id = ?", (session.id,))
    connection.executemany(
        "INSERT INTO token_usage (session_id, timestamp, model, input_tokens, output_tokens, "
        "cached_input_tokens, cache_creation_input_tokens, cache_read_input_tokens, "
        "reasoning_output_tokens, total_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                session.id,
                usage.timestamp.isoformat(),
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
                usage.reasoning_output_tokens,
                usage.total_tokens,
            )
            for usage in session.token_usages
        ],
    )


def _has_token_usage_table(connection: sqlite3.Connection) -> bool:
    """Return whether the token-usage table exists."""
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'token_usage'"
        ).fetchone()
        is not None
    )
