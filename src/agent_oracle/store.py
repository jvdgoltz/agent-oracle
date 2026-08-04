"""SQLite store for Agent Oracle.

Manages a single database file holding sessions, messages (with FTS5 full-text
search and sqlite-vec vector search), entities, and summaries.  Provides text,
vector, and hybrid (reciprocal rank fusion) search over archived sessions.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from agent_oracle.models import Session

logger = logging.getLogger(__name__)

_EMBED_DIM = 384
_RRF_K = 60


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
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
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
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role       TEXT,
                content    TEXT,
                timestamp  TEXT,
                seq        INTEGER
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
            "(SELECT enriched FROM sessions WHERE id = ?))",
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
                "INSERT INTO messages (session_id, role, content, timestamp, seq) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session.id,
                    msg.role.value,
                    msg.content,
                    msg.timestamp.isoformat(),
                    seq,
                ),
            )
            self.conn.execute(
                "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
                (cursor.lastrowid, msg.content),
            )
        self.conn.commit()
        logger.debug("Indexed session %s with %d messages", session.id, len(session.messages))

    def _delete_session_messages(self, session_id: str) -> None:
        """Remove all messages and FTS entries for *session_id*."""
        self.conn.execute(
            "DELETE FROM messages_fts WHERE rowid IN "
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
        """Replace all entities for *session_id* with *entities*."""
        self.conn.execute("DELETE FROM entities WHERE session_id = ?", (session_id,))
        self.conn.executemany(
            "INSERT INTO entities (session_id, entity_type, entity_value) VALUES (?, ?, ?)",
            [(session_id, e["type"], e["value"]) for e in entities],
        )
        self.conn.commit()

    def set_summary(self, session_id: str, summary: str) -> None:
        """Set the *summary* text on the session row."""
        self.conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary, session_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #

    def search_text(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 BM25 search returning session_id, snippet, and rank per match."""
        rows = self.conn.execute(
            """
            SELECT m.session_id        AS session_id,
                   snippet(messages_fts, 0, '[', ']', '...', 8) AS snippet,
                   bm25(messages_fts)   AS rank
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_vector(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        """sqlite-vec cosine search returning session_id and distance per match."""
        rows = self.conn.execute(
            """
            SELECT m.session_id AS session_id,
                   v.distance   AS distance
            FROM vec_messages v
            JOIN messages m ON m.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(query_embedding), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_hybrid(
        self, query: str, query_embedding: list[float], limit: int = 20
    ) -> list[dict]:
        """Reciprocal rank fusion of text and vector search results."""
        text_results = self._dedupe_by_session(self.search_text(query, limit=limit))
        vec_results = self._dedupe_by_session(self.search_vector(query_embedding, limit=limit))
        scores: dict[str, float] = {}
        for rank, result in enumerate(text_results):
            scores[result["session_id"]] = scores.get(result["session_id"], 0.0) + 1.0 / (
                _RRF_K + rank + 1
            )
        for rank, result in enumerate(vec_results):
            scores[result["session_id"]] = scores.get(result["session_id"], 0.0) + 1.0 / (
                _RRF_K + rank + 1
            )
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"session_id": sid, "score": score} for sid, score in ranked]

    @staticmethod
    def _dedupe_by_session(results: list[dict]) -> list[dict]:
        """Keep only the first (best) result per session_id."""
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            sid = r["session_id"]
            if sid not in seen:
                seen.add(sid)
                deduped.append(r)
        return deduped

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
            "SELECT id, session_id, role, content, timestamp, seq "
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
