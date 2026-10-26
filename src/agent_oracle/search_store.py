"""SQLite text and vector queries with session-scoped candidate selection."""

import sqlite3
from typing import Any

import sqlite_vec


def sanitize_fts_query(query: str) -> str:
    """Escape a user query into a safe FTS5 MATCH expression."""
    tokens = query.strip().split()
    if not tokens:
        return ""
    return " ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def session_filter(agent: str | None, entity: str | None) -> tuple[str, list[str]]:
    """Build a session predicate applied before search candidates are limited."""
    clauses = ["NOT s.is_review_agent"]
    parameters = []
    if agent is not None:
        clauses.append("s.agent = ?")
        parameters.append(agent)
    if entity is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM entities e WHERE e.session_id = s.id AND e.entity_value = ?)"
        )
        parameters.append(entity)
    return " AND ".join(clauses), parameters


def _merge_by_score(
    message_rows: list[sqlite3.Row], summary_rows: list[sqlite3.Row], score_key: str
) -> list[dict[str, Any]]:
    """Merge message and summary rows, ordered by *score_key*, dropping NULLs.

    NULL scores (e.g. ``bm25()`` on rows whose FTS5 docsize entry is missing)
    cannot be ordered and indicate an unscoreable match, so those rows are
    dropped rather than crashing the sort.
    """
    merged = [dict(r) for r in (*message_rows, *summary_rows) if r[score_key] is not None]
    merged.sort(key=lambda r: r[score_key])
    return merged


def search_text(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    *,
    agent: str | None = None,
    entity: str | None = None,
) -> list[dict]:
    """FTS5 BM25 search over messages and session summaries."""
    fts_query = sanitize_fts_query(query)
    if not fts_query:
        return []
    predicate, parameters = session_filter(agent, entity)
    message_rows = conn.execute(
        f"""
        SELECT m.session_id        AS session_id,
               m.id                AS message_id,
               s.agent             AS agent,
               s.cwd               AS cwd,
               s.title      AS title,
               s.started_at        AS started_at,
               s.summary           AS summary,
               snippet(messages_fts, 0, '[', ']', '...', 8) AS snippet,
               bm25(messages_fts)   AS rank
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        JOIN sessions s ON s.id = m.session_id
        WHERE messages_fts MATCH ? AND {predicate}
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, *parameters, limit),
    ).fetchall()
    summary_rows = conn.execute(
        f"""
        SELECT s.id                              AS session_id,
               NULL                              AS message_id,
               s.agent                           AS agent,
               s.cwd                             AS cwd,
               s.title      AS title,
               s.started_at                      AS started_at,
               s.summary                         AS summary,
               snippet(sessions_fts, 0, '[', ']', '...', 8) AS snippet,
               bm25(sessions_fts)                AS rank
        FROM sessions_fts
        JOIN sessions s ON s.rowid = sessions_fts.rowid
        WHERE sessions_fts MATCH ? AND {predicate}
        ORDER BY rank
        LIMIT ?
        """,
        (fts_query, *parameters, limit),
    ).fetchall()
    return _merge_by_score(message_rows, summary_rows, "rank")[:limit]


def search_vector(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 20,
    *,
    agent: str | None = None,
    entity: str | None = None,
) -> list[dict]:
    """sqlite-vec cosine search over message and summary embeddings."""
    blob = sqlite_vec.serialize_float32(query_embedding)
    predicate, parameters = session_filter(agent, entity)
    message_rows = conn.execute(
        f"""
        SELECT m.session_id AS session_id,
               m.id         AS message_id,
               s.agent      AS agent,
               s.cwd        AS cwd,
               s.title      AS title,
               s.started_at AS started_at,
               s.summary    AS summary,
               v.distance  AS distance
        FROM vec_messages v
        JOIN messages m ON m.id = v.rowid
        JOIN sessions s ON s.id = m.session_id
        WHERE v.embedding MATCH ? AND k = ?
          AND v.rowid IN (
              SELECT m.id FROM messages m JOIN sessions s ON s.id = m.session_id
              WHERE {predicate}
          )
        ORDER BY v.distance
        """,
        (blob, limit, *parameters),
    ).fetchall()
    summary_rows = conn.execute(
        f"""
        SELECT s.id        AS session_id,
               NULL        AS message_id,
               s.agent     AS agent,
               s.cwd       AS cwd,
               s.title      AS title,
               s.started_at AS started_at,
               s.summary   AS summary,
               v.distance  AS distance
        FROM vec_sessions v
        JOIN sessions s ON s.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
          AND v.rowid IN (SELECT s.rowid FROM sessions s WHERE {predicate})
        ORDER BY v.distance
        """,
        (blob, limit, *parameters),
    ).fetchall()
    return _merge_by_score(message_rows, summary_rows, "distance")[:limit]
