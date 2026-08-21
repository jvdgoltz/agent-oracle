"""Tests for the session-title database migration."""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    """Load the standalone title migration for direct testing."""
    script = Path(__file__).parents[2] / "scripts" / "backfill_session_titles.py"
    spec = importlib.util.spec_from_file_location("backfill_session_titles", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_titles_backs_up_schema_and_populates_existing_rows(tmp_path: Path) -> None:
    """Migration backs up first, adds title, and updates only matching sessions."""
    database = tmp_path / "index.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, agent TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('fac-001', 'factory')")

    session_dir = tmp_path / ".factory" / "sessions" / "project"
    session_dir.mkdir(parents=True)
    (session_dir / "fac-001.jsonl").write_text(
        json.dumps({"type": "session_start", "id": "fac-001", "title": "Factory title"}) + "\n"
    )

    backup, updated = _load_module().migrate_titles(database, tmp_path)

    with sqlite3.connect(backup) as connection:
        assert "title" not in {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute("SELECT title FROM sessions WHERE id = 'fac-001'").fetchone()[0]
            == "Factory title"
        )
    assert updated == 1
