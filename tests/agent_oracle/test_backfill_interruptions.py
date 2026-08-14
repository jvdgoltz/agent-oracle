"""Tests for the interruption-only archive migration."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from agent_oracle.models import AgentType, Interruption, Session


def _load_module() -> ModuleType:
    """Load the standalone migration script for direct testing."""
    script = Path(__file__).parents[2] / "scripts" / "backfill_interruptions.py"
    spec = importlib.util.spec_from_file_location("backfill_interruptions", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _database(path: Path) -> None:
    """Create the legacy message schema used before interruption metadata."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, seq INTEGER, "
            "is_thinking INTEGER, is_system_instruction INTEGER, is_injected INTEGER)"
        )
        connection.execute("INSERT INTO messages VALUES ('session-1', 'user', 'stop', 0, 0, 0, 0)")


def _session() -> Session:
    """Build a parsed session containing one confirmed interruption."""
    return Session(
        id="session-1",
        agent=AgentType.CLAUDE,
        cwd="/work",
        started_at=datetime(2026, 8, 1, tzinfo=UTC),
        interruptions=[Interruption("assistant-1", None, "claude-fable-5", 0)],
    )


def test_preview_is_read_only_for_legacy_schema(tmp_path: Path) -> None:
    """Preview calculates changes before the additive columns exist."""
    database = tmp_path / "index.db"
    _database(database)

    result = _load_module().preview(database, [_session()])

    assert (result.source_interruptions, result.matched_messages, result.changed_messages) == (
        1,
        1,
        1,
    )
    with sqlite3.connect(database) as connection:
        assert "is_interrupted" not in {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }


def test_migrate_backs_up_and_is_idempotent(tmp_path: Path) -> None:
    """Migration creates a verified backup and only changes the target message once."""
    database = tmp_path / "index.db"
    _database(database)
    migration = _load_module()

    first = migration.migrate(database, [_session()])
    second = migration.migrate(database, [_session()])

    assert first.backup_path is not None and first.backup_path.exists()
    assert first.changed_messages == 1
    assert second.changed_messages == 0
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT is_interrupted, interruption_model FROM messages"
        ).fetchone()
    assert row == (1, "claude-fable-5")


def test_migrate_corrects_only_legacy_codex_abort_marker(tmp_path: Path) -> None:
    """Move a legacy Codex interruption mark from its synthetic notice to the real user turn."""
    database = tmp_path / "index.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT, agent TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('session-1', 'codex')")
        connection.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, seq INTEGER, "
            "is_thinking INTEGER, is_system_instruction INTEGER, is_injected INTEGER, "
            "is_interrupted INTEGER, interruption_model TEXT)"
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, 'user', ?, ?, 0, 0, ?, ?, ?)",
            [
                ("session-1", "real user", 0, 0, 0, None),
                (
                    "session-1",
                    "<turn_aborted>\nThe user interrupted the previous turn on purpose.\n"
                    "</turn_aborted>",
                    1,
                    0,
                    1,
                    "gpt-5.6",
                ),
            ],
        )
    session = Session(
        id="session-1",
        agent=AgentType.CODEX,
        cwd="/work",
        started_at=datetime(2026, 8, 1, tzinfo=UTC),
        interruptions=[Interruption("turn-1", None, "gpt-5.6", 0)],
    )

    result = _load_module().migrate(database, [session])

    assert result.corrected_markers == 1
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT seq, is_injected, is_interrupted, interruption_model FROM messages ORDER BY seq"
        ).fetchall()
    assert rows == [(0, 0, 1, "gpt-5.6"), (1, 1, 0, None)]


def test_preview_counts_codex_marker_correction_without_writing(tmp_path: Path) -> None:
    """Preview reports legacy Codex marker cleanup without changing its stored fields."""
    database = tmp_path / "index.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT, agent TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('session-1', 'codex')")
        connection.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, seq INTEGER, "
            "is_thinking INTEGER, is_system_instruction INTEGER, is_injected INTEGER, "
            "is_interrupted INTEGER, interruption_model TEXT)"
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, 'user', ?, 0, 0, 0, 0, 1, 'gpt-5.6')",
            (
                "session-1",
                "<turn_aborted>\nThe user interrupted the previous turn on purpose.\n"
                "</turn_aborted>",
            ),
        )

    result = _load_module().preview(database, [])

    assert result.corrected_markers == 1
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT is_injected, is_interrupted, interruption_model FROM messages"
        ).fetchone()
    assert row == (0, 1, "gpt-5.6")


def test_preview_does_not_treat_similar_codex_text_as_an_abort_marker(tmp_path: Path) -> None:
    """Marker cleanup uses literal equality rather than SQL LIKE wildcards."""
    database = tmp_path / "index.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT, agent TEXT)")
        connection.execute("INSERT INTO sessions VALUES ('session-1', 'codex')")
        connection.execute(
            "CREATE TABLE messages (session_id TEXT, role TEXT, content TEXT, seq INTEGER, "
            "is_thinking INTEGER, is_system_instruction INTEGER, is_injected INTEGER, "
            "is_interrupted INTEGER, interruption_model TEXT)"
        )
        connection.execute(
            "INSERT INTO messages VALUES (?, 'user', ?, 0, 0, 0, 0, 1, 'gpt-5.6')",
            (
                "session-1",
                "<turn_aborted>\nThe user interrupted the previous turn on purpose.X\n"
                "</turn_aborted>",
            ),
        )

    result = _load_module().preview(database, [])

    assert result.corrected_markers == 0
