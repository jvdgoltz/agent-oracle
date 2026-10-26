"""Tests for the targeted Codex review-session metadata migration."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    """Load the standalone review-session migration script for direct testing."""
    script = Path(__file__).parents[2] / "scripts" / "backfill_codex_review_sessions.py"
    spec = importlib.util.spec_from_file_location("backfill_codex_review_sessions", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _database(path: Path) -> None:
    """Create a pre-review-metadata archive with stale index rows."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, agent TEXT, cwd TEXT, started_at TEXT, "
            "summary TEXT, enriched INTEGER)"
        )
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE messages_fts "
            "USING fts5(content, content='messages', content_rowid='id')"
        )
        connection.execute("CREATE VIRTUAL TABLE sessions_fts USING fts5(summary)")
        connection.execute("CREATE TABLE vec_messages (embedding BLOB)")
        connection.execute("CREATE TABLE vec_sessions (embedding BLOB)")
        connection.executemany(
            "INSERT INTO sessions VALUES (?, 'codex', '', '', '', 0)",
            [("parent",), ("review",)],
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?)",
            [
                (1, "parent", "parent"),
                (2, "review", "review"),
                (3, "review", "never indexed"),
            ],
        )
        connection.executemany(
            "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
            [(1, "parent"), (2, "review")],
        )
        rows = connection.execute("SELECT rowid, id FROM sessions").fetchall()
        connection.executemany(
            "INSERT INTO sessions_fts(rowid, summary) VALUES (?, ?)",
            [(rowid, session_id) for rowid, session_id in rows],
        )
        connection.executemany(
            "INSERT INTO vec_messages(rowid, embedding) VALUES (?, ?)",
            [(1, b"p"), (2, b"r"), (3, b"u")],
        )
        connection.executemany(
            "INSERT INTO vec_sessions(rowid, embedding) VALUES (?, ?)",
            [(rowid, session_id.encode()) for rowid, session_id in rows],
        )


def _review_source(home: Path) -> None:
    """Write one source review session that links to the parent session."""
    path = home / ".codex" / "sessions" / "review.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "review",
                    "cwd": "/work",
                    "timestamp": "2026-08-17T00:00:00Z",
                    "parent_thread_id": "parent",
                    "source": {"subagent": {"other": "guardian"}},
                },
            }
        )
        + "\n"
    )


def test_preview_is_read_only_without_review_metadata_columns(tmp_path: Path) -> None:
    """Preview identifies changes without adding columns or purging any indexes."""
    database = tmp_path / "index.db"
    home = tmp_path / "home"
    _database(database)
    _review_source(home)

    result = _load_module().preview(database, home)

    assert (result.matched_sessions, result.changed_sessions, result.review_sessions) == (1, 1, 1)
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        assert "is_review_agent" not in columns
        assert connection.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 2


def test_migrate_creates_backup_and_links_reviews_without_reindexing(
    tmp_path: Path,
) -> None:
    """Write migration is backed up and changes only review metadata."""
    database = tmp_path / "index.db"
    home = tmp_path / "home"
    _database(database)
    _review_source(home)

    result = _load_module().migrate(database, home)

    assert result.backup_path is not None and result.backup_path.exists()
    assert result.changed_sessions == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT is_review_agent, parent_thread_id FROM sessions WHERE id = 'review'"
        ).fetchone() == (1, "parent")
        assert connection.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM sessions_fts").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM vec_messages").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM vec_sessions").fetchone()[0] == 2
