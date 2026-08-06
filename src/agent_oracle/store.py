"""SQLite store for Agent Oracle.

Manages a single database file holding sessions, messages (with FTS5 full-text
search and sqlite-vec vector search), entities, and summaries.  Provides text,
vector, and hybrid (reciprocal rank fusion) search over archived sessions.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sqlite_vec

from agent_oracle.models import Session

logger = logging.getLogger(__name__)


_EMBED_DIM = 384
_RRF_K = 60


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


def _sanitize_fts_query(query: str) -> str:
    """Escape a user query into a safe FTS5 MATCH expression.

    FTS5 treats characters like ``? * " ( )`` as query syntax.  To avoid
    syntax errors, each whitespace-separated token is wrapped in double
    quotes (phrase query), and any embedded double quotes are doubled
    (FTS5 escape rule).  Returns an empty string if nothing remains.
    """
    tokens = query.strip().split()
    if not tokens:
        return ""
    escaped = []
    for token in tokens:
        safe = token.replace('"', '""')
        escaped.append(f'"{safe}"')
    return " ".join(escaped)


class Store:
    """Persistent SQLite store for sessions, messages, embeddings, and entities."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the database at *db_path* and initialize its schema."""
        self.db_path = db_path
        self.conn = self._connect()
        self._init_schema()
        logger.info("Store initialized at %s", db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open the SQLite connection and load the sqlite-vec extension."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn

    def _init_schema(self) -> None:
        """Create all tables, indexes, and virtual tables if they do not exist."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id        TEXT PRIMARY KEY,
                agent     TEXT,
                cwd       TEXT,
                started_at TEXT,
                summary   TEXT,
                enriched  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id           TEXT,
                role                 TEXT,
                content              TEXT,
                timestamp            TEXT,
                seq                  INTEGER,
                is_thinking          INTEGER DEFAULT 0,
                model                TEXT,
                is_system_instruction INTEGER DEFAULT 0,
                is_injected          INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id);

            CREATE TABLE IF NOT EXISTS entities (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                entity_type  TEXT,
                entity_value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_entities_session_id
                ON entities(session_id);
            """
        )
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts "
            "USING fts5(content, content='messages', content_rowid='id')"
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages "
            f"USING vec0(embedding float[{_EMBED_DIM}])"
        )
        # Summary indexes keyed by the sessions rowid so session-level
        # enrichment becomes searchable alongside messages.
        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(summary)")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sessions "
            f"USING vec0(embedding float[{_EMBED_DIM}])"
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def index_session(self, session: Session) -> None:
        """Insert or replace *session* and its messages (regular table + FTS5)."""
        self._delete_session_messages(session.id)
        self.conn.execute(
            "INSERT OR REPLACE INTO sessions (id, agent, cwd, started_at, summary, enriched) "
            "VALUES (?, ?, ?, ?, "
            "(SELECT summary FROM sessions WHERE id = ?), "
            "COALESCE((SELECT enriched FROM sessions WHERE id = ?), 0))",
            (
                session.id,
                session.agent.value,
                session.cwd,
                session.started_at.isoformat(),
                session.id,
                session.id,
            ),
        )
        for seq, msg in enumerate(session.messages):
            cursor = self.conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, timestamp, seq, "
                "is_thinking, model, is_system_instruction, is_injected) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    msg.role.value,
                    msg.content,
                    msg.timestamp.isoformat(),
                    seq,
                    int(msg.is_thinking),
                    msg.model,
                    int(msg.is_system_instruction),
                    int(msg.is_injected),
                ),
            )
            if msg.is_searchable:
                self.conn.execute(
                    "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                    (cursor.lastrowid, msg.content),
                )
        self.conn.commit()
        logger.debug("Indexed session %s with %d messages", session.id, len(session.messages))

    def _delete_session_messages(self, session_id: str) -> None:
        """Remove all messages and their FTS and vector index entries for *session_id*."""
        self.conn.execute(
            "DELETE FROM messages_fts WHERE rowid IN "
            "(SELECT id FROM messages WHERE session_id = ?)",
            (session_id,),
        )
        self.conn.execute(
            "DELETE FROM vec_messages WHERE rowid IN "
            "(SELECT id FROM messages WHERE session_id = ?)",
            (session_id,),
        )
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    def upsert_embedding(self, message_id: int, embedding: list[float]) -> None:
        """Insert (or replace) the *embedding* for message *message_id* into vec_messages."""
        self.conn.execute("DELETE FROM vec_messages WHERE rowid = ?", (message_id,))
        self.conn.execute(
            "INSERT INTO vec_messages (rowid, embedding) VALUES (?, ?)",
            (message_id, sqlite_vec.serialize_float32(embedding)),
        )
        self.conn.commit()

    def upsert_entities(self, session_id: str, entities: list[dict]) -> None:
        """Replace all entities for *session_id* with *entities* and mark it enriched."""
        self.conn.execute("DELETE FROM entities WHERE session_id = ?", (session_id,))
        self.conn.executemany(
            "INSERT INTO entities (session_id, entity_type, entity_value) VALUES (?, ?, ?)",
            [(session_id, e["type"], e["value"]) for e in entities],
        )
        self.conn.execute("UPDATE sessions SET enriched = 1 WHERE id = ?", (session_id,))
        self.conn.commit()

    def set_summary(self, session_id: str, summary: str) -> None:
        """Set the *summary* text on the session row and index it for FTS."""
        self.conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary, session_id),
        )
        rowid = self._session_rowid(session_id)
        if rowid is not None:
            self.conn.execute("DELETE FROM sessions_fts WHERE rowid = ?", (rowid,))
            if summary:
                self.conn.execute(
                    "INSERT INTO sessions_fts (rowid, summary) VALUES (?, ?)",
                    (rowid, summary),
                )
        self.conn.commit()

    def upsert_session_embedding(self, session_id: str, embedding: list[float]) -> None:
        """Insert (or replace) the *embedding* of the session summary into vec_sessions."""
        rowid = self._session_rowid(session_id)
        if rowid is None:
            logger.warning("No session %s found; skipping summary embedding", session_id)
            return
        self.conn.execute("DELETE FROM vec_sessions WHERE rowid = ?", (rowid,))
        self.conn.execute(
            "INSERT INTO vec_sessions (rowid, embedding) VALUES (?, ?)",
            (rowid, sqlite_vec.serialize_float32(embedding)),
        )
        self.conn.commit()

    def _session_rowid(self, session_id: str) -> int | None:
        """Return the SQLite rowid of *session_id*, or None if it does not exist."""
        row = self.conn.execute("SELECT rowid FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return int(row[0]) if row else None

    def backfill_summary_indexes(
        self, embed_batch: Callable[[list[str]], list[list[float]]]
    ) -> int:
        """Index summaries of sessions missing from sessions_fts or vec_sessions.

        *embed_batch* vectorizes the summaries that lack a vec_sessions entry.
        Also marks sessions with a summary as enriched (the flag predates the
        backfill). Returns the number of sessions that were backfilled.
        """
        rows = self.conn.execute(
            """
            SELECT s.rowid AS rowid, s.id AS session_id, s.summary AS summary,
                   EXISTS(SELECT 1 FROM sessions_fts f WHERE f.rowid = s.rowid) AS has_fts,
                   EXISTS(SELECT 1 FROM vec_sessions v WHERE v.rowid = s.rowid) AS has_vec
            FROM sessions s
            WHERE s.summary IS NOT NULL AND s.summary != ''
            """
        ).fetchall()
        need_fts = [r for r in rows if not r["has_fts"]]
        need_vec = [r for r in rows if not r["has_vec"]]
        if not need_fts and not need_vec:
            return 0
        for r in need_fts:
            self.conn.execute(
                "INSERT INTO sessions_fts (rowid, summary) VALUES (?, ?)",
                (r["rowid"], r["summary"]),
            )
        if need_vec:
            embeddings = embed_batch([r["summary"] for r in need_vec])
            for r, embedding in zip(need_vec, embeddings, strict=True):
                self.conn.execute(
                    "INSERT INTO vec_sessions (rowid, embedding) VALUES (?, ?)",
                    (r["rowid"], sqlite_vec.serialize_float32(embedding)),
                )
        self.conn.execute(
            "UPDATE sessions SET enriched = 1 WHERE summary IS NOT NULL AND summary != ''"
        )
        self.conn.commit()
        logger.info("Backfilled summary indexes: %d FTS, %d vectors", len(need_fts), len(need_vec))
        return len(need_fts) + len(need_vec)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search_text(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 BM25 search over messages and session summaries."""
        fts_query = _sanitize_fts_query(query)
        if not fts_query:
            return []
        message_rows = self.conn.execute(
            """
            SELECT m.session_id        AS session_id,
                   m.id                AS message_id,
                   s.agent             AS agent,
                   s.cwd               AS cwd,
                   s.started_at        AS started_at,
                   s.summary           AS summary,
                   snippet(messages_fts, 0, '[', ']', '...', 8) AS snippet,
                   bm25(messages_fts)   AS rank
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        summary_rows = self.conn.execute(
            """
            SELECT s.id                              AS session_id,
                   NULL                              AS message_id,
                   s.agent                           AS agent,
                   s.cwd                             AS cwd,
                   s.started_at                      AS started_at,
                   s.summary                         AS summary,
                   snippet(sessions_fts, 0, '[', ']', '...', 8) AS snippet,
                   bm25(sessions_fts)                AS rank
            FROM sessions_fts
            JOIN sessions s ON s.rowid = sessions_fts.rowid
            WHERE sessions_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return _merge_by_score(message_rows, summary_rows, "rank")[:limit]

    def search_vector(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        """sqlite-vec cosine search over message and summary embeddings."""
        blob = sqlite_vec.serialize_float32(query_embedding)
        message_rows = self.conn.execute(
            """
            SELECT m.session_id AS session_id,
                   m.id         AS message_id,
                   s.agent      AS agent,
                   s.cwd        AS cwd,
                   s.started_at AS started_at,
                   s.summary    AS summary,
                   v.distance  AS distance
            FROM vec_messages v
            JOIN messages m ON m.id = v.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (blob, limit),
        ).fetchall()
        summary_rows = self.conn.execute(
            """
            SELECT s.id        AS session_id,
                   NULL        AS message_id,
                   s.agent     AS agent,
                   s.cwd       AS cwd,
                   s.started_at AS started_at,
                   s.summary   AS summary,
                   v.distance  AS distance
            FROM vec_sessions v
            JOIN sessions s ON s.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (blob, limit),
        ).fetchall()
        return _merge_by_score(message_rows, summary_rows, "distance")[:limit]

    def search_hybrid(
        self, query: str, query_embedding: list[float], limit: int = 20
    ) -> list[dict]:
        """Reciprocal rank fusion of text and vector results at message level.

        Each candidate is keyed by its ``message_id`` (or the session for
        summary rows), so different messages of the same session are ranked
        independently and both surface in the results.  Text results carry a
        matched snippet; vector-only hits contribute score but no snippet.
        """
        text_results = self.search_text(query, limit=limit)
        vec_results = self.search_vector(query_embedding, limit=limit)
        # Accumulate RRF scores per candidate and keep the row with a snippet
        # when the same message appears in both lists.
        scores: dict[tuple[str, object], float] = {}
        meta: dict[tuple[str, object], dict] = {}
        for results in (text_results, vec_results):
            for rank, result in enumerate(results):
                message_id = result.get("message_id")
                key = (
                    ("msg", message_id) if message_id is not None else ("sum", result["session_id"])
                )
                scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
                if key not in meta or (result.get("snippet") and not meta[key].get("snippet")):
                    meta[key] = result
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {
                "session_id": meta[key]["session_id"],
                "message_id": meta[key].get("message_id"),
                "agent": meta[key].get("agent"),
                "cwd": meta[key].get("cwd"),
                "started_at": meta[key].get("started_at"),
                "summary": meta[key].get("summary"),
                "snippet": meta[key].get("snippet", ""),
                "score": score,
            }
            for key, score in ranked
        ]

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Return sessions ordered by started_at descending with pagination."""
        rows = self.conn.execute(
            """
            SELECT id, agent, cwd, started_at, summary, enriched
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def is_session_indexed(self, session_id: str) -> bool:
        """Return True if *session_id* exists and has been enriched."""
        row = self.conn.execute(
            "SELECT enriched FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None and bool(row["enriched"])

    def get_session(self, session_id: str) -> dict | None:
        """Return the session and all its messages, or None if not found."""
        row = self.conn.execute(
            "SELECT id, agent, cwd, started_at, summary, enriched FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        msg_rows = self.conn.execute(
            "SELECT id, session_id, role, content, timestamp, seq, "
            "is_thinking, model, is_system_instruction, is_injected "
            "FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        result["messages"] = [dict(r) for r in msg_rows]
        return result

    def get_entities(self, session_id: str) -> list[dict]:
        """Return all entities for *session_id*."""
        rows = self.conn.execute(
            "SELECT entity_type, entity_value FROM entities WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return [{"entity_type": r["entity_type"], "entity_value": r["entity_value"]} for r in rows]

    def list_entities(self, session_ids: list[str]) -> dict[str, list[dict]]:
        """Return entities keyed by session_id for all given *session_ids* in one query."""
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        rows = self.conn.execute(
            "SELECT session_id, entity_type, entity_value FROM entities "
            f"WHERE session_id IN ({placeholders})",
            session_ids,
        ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for r in rows:
            grouped.setdefault(r["session_id"], []).append(
                {"entity_type": r["entity_type"], "entity_value": r["entity_value"]}
            )
        return grouped
