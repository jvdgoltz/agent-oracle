"""Tests for the targeted excluded-content repair script."""

import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import sqlite_vec

from agent_oracle.models import AgentType, Message, MessageRole, Session
from agent_oracle.store import Store


def _load_module() -> ModuleType:
    """Load the standalone repair script."""
    script = Path(__file__).parents[2] / "scripts" / "repair_excluded_content.py"
    spec = importlib.util.spec_from_file_location("repair_excluded_content", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _database(path: Path) -> None:
    """Create a disposable archive containing ordinary and excluded rows."""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, summary TEXT, enriched INTEGER);
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
                is_thinking INTEGER DEFAULT 0, is_system_instruction INTEGER DEFAULT 0,
                is_injected INTEGER DEFAULT 0
            );
            CREATE TABLE entities (session_id TEXT, entity_type TEXT, entity_value TEXT);
            CREATE VIRTUAL TABLE messages_fts USING fts5(content);
            CREATE VIRTUAL TABLE sessions_fts USING fts5(summary);
            CREATE TABLE vec_messages (rowid INTEGER PRIMARY KEY, embedding BLOB);
            CREATE TABLE vec_sessions (rowid INTEGER PRIMARY KEY, embedding BLOB);
            INSERT INTO sessions VALUES ('s1', 'old summary', 1);
            INSERT INTO messages VALUES
                (1, 's1', 'user', 'ordinary', 0, 0, 0),
                (2, 's1', 'user', '<codex_internal_context>secret', 0, 0, 0),
                (3, 's1', 'developer', 'developer', 0, 0, 0);
            INSERT INTO messages_fts(rowid, content)
                VALUES (1, 'ordinary'), (2, '<codex_internal_context>secret'), (3, 'developer');
            INSERT INTO sessions_fts(rowid, summary) VALUES (1, 'old summary');
            INSERT INTO vec_messages VALUES (1, X'01'), (2, X'02'), (3, X'03');
            INSERT INTO vec_sessions VALUES (1, X'04');
            INSERT INTO entities VALUES ('s1', 'product', 'sqlite');
            """
        )


def test_preview_does_not_write(tmp_path: Path) -> None:
    """Preview reports matching rows and leaves data untouched."""
    database = tmp_path / "index.db"
    _database(database)
    result = _load_module().preview(database)

    assert result.matched_messages == 2
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 3
        assert connection.execute("SELECT summary FROM sessions").fetchone()[0] == "old summary"


def test_repair_backs_up_and_removes_only_excluded_indexes(tmp_path: Path) -> None:
    """Write mode backs up, flags rows, removes vectors, and invalidates summaries."""
    database = tmp_path / "index.db"
    _database(database)
    result = _load_module().repair(database)

    assert result.backup_path is not None and result.backup_path.exists()
    assert result.deleted_fts == 2
    assert result.deleted_vectors == 2
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0] == 1
        assert connection.execute("SELECT summary, enriched FROM sessions").fetchone() == (None, 0)
        assert connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
        assert connection.execute(
            "SELECT is_system_instruction, is_injected FROM messages WHERE id = 2"
        ).fetchone() == (0, 1)
        assert connection.execute(
            "SELECT is_system_instruction, is_injected FROM messages WHERE id = 3"
        ).fetchone() == (1, 0)


def test_repair_uses_production_store_schema_and_preserves_flags(tmp_path: Path) -> None:
    """Repair removes stale entries from real external-content FTS and sqlite-vec tables."""
    database = tmp_path / "index.db"
    store = Store(database)
    ts = datetime.now(UTC)
    store.index_session(
        Session(
            id="s1",
            agent=AgentType.CODEX,
            cwd="/tmp",
            started_at=ts,
            messages=[
                Message(MessageRole.USER, "ordinary", ts),
                Message(MessageRole.USER, "generated", ts),
                Message(MessageRole.USER, "native injected", ts),
                Message(MessageRole.USER, "developer", ts),
            ],
        )
    )
    rows = store.conn.execute("SELECT id FROM messages ORDER BY id").fetchall()
    store.conn.execute(
        "UPDATE messages SET role = CASE WHEN id = 4 THEN 'developer' ELSE role END, "
        "is_injected = CASE WHEN id = 3 THEN 1 ELSE is_injected END, "
        "is_thinking = CASE WHEN id = 2 THEN 1 ELSE is_thinking END, "
        "is_system_instruction = CASE WHEN id = 4 THEN 1 ELSE is_system_instruction END "
        "WHERE id > 1"
    )
    for row in rows[1:]:
        store.conn.execute(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            (row[0], sqlite_vec.serialize_float32([0.0] * 384)),
        )
    store.conn.commit()
    store.conn.close()

    result = _load_module().repair(database)

    assert result.deleted_fts == 3
    assert result.deleted_vectors == 3
    preview = _load_module().preview(database)
    assert preview.deleted_fts == 0
    assert preview.deleted_vectors == 0
    assert preview.invalidated_sessions == 0
    repaired = Store(database)
    try:
        assert repaired.search_text("ordinary")
        assert repaired.search_text("generated") == []
        assert repaired.search_text("injected") == []
        assert repaired.search_text("developer") == []
        assert repaired.conn.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0] == 0
        assert (
            repaired.conn.execute(
                "SELECT is_injected FROM messages WHERE content = 'native injected'"
            ).fetchone()[0]
            == 1
        )
    finally:
        repaired.conn.close()
