"""Serialized SQLite access and schema for the local archive."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar

import sqlite_vec

logger = logging.getLogger(__name__)
_EMBED_DIM = 384
_P = ParamSpec("_P")
_T = TypeVar("_T")
_S = TypeVar("_S", bound="StoreDatabase")


def serialized(
    method: Callable[Concatenate[_S, _P], _T],
) -> Callable[Concatenate[_S, _P], _T]:
    """Isolate an entire store operation and roll back failed transactions."""

    @wraps(method)
    def run(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        """Hold the connection lock until the operation has committed or rolled back."""
        with self._lock, self.conn:
            return method(self, *args, **kwargs)

    return run


class StoreDatabase:
    """Own the connection, schema, and reentrant transaction lock."""

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the database at *db_path* and initialize its schema."""
        self._lock = threading.RLock()
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
                title     TEXT,
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
                is_injected          INTEGER DEFAULT 0,
                is_interrupted       INTEGER DEFAULT 0,
                interruption_model   TEXT
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
        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(summary)")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sessions "
            f"USING vec0(embedding float[{_EMBED_DIM}])"
        )
        self.conn.commit()
