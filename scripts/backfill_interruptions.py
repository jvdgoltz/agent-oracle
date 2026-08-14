#!/usr/bin/env python3
"""Backfill verified user interruptions without re-indexing sessions."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from agent_oracle.models import Session
from agent_oracle.sources.claude import parse_claude_session
from agent_oracle.sources.codex import parse_codex_session
from agent_oracle.sources.factory import parse_factory_session
from agent_oracle.sources.omp import parse_omp_session

DEFAULT_DATABASE = Path.home() / ".agent-oracle" / "index.db"
Parser = Callable[[Path], Session]
_CODEX_ABORTED_MARKER = (
    "<turn_aborted>\nThe user interrupted the previous turn on purpose.\n</turn_aborted>"
)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Counts and backup path from an interruption backfill."""

    source_interruptions: int
    matched_messages: int
    changed_messages: int
    corrected_markers: int
    backup_path: Path | None


def _source_paths(home: Path) -> Iterable[tuple[Path, Parser]]:
    """Yield archived source files and their matching parser."""
    sources = (
        (home / ".codex" / "sessions", parse_codex_session),
        (home / ".factory" / "sessions", parse_factory_session),
        (home / ".claude" / "projects", parse_claude_session),
        (home / ".omp" / "agent" / "sessions", parse_omp_session),
    )
    for directory, parser in sources:
        if directory.exists():
            yield from ((path, parser) for path in sorted(directory.rglob("*.jsonl")))


def _create_verified_backup(database: Path) -> Path:
    """Create and integrity-check a consistent backup before writing."""
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    backup_path = database.with_name(f"{database.name}.interruptions.bak.{time.time_ns()}")
    with (
        sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source,
        sqlite3.connect(backup_path) as backup,
    ):
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Source database integrity check failed.")
        source.backup(backup)
        if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup database integrity check failed.")
    return backup_path


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Add the two additive interruption columns when they are absent."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    if "is_interrupted" not in columns:
        connection.execute("ALTER TABLE messages ADD COLUMN is_interrupted INTEGER DEFAULT 0")
    if "interruption_model" not in columns:
        connection.execute("ALTER TABLE messages ADD COLUMN interruption_model TEXT")


def _apply_sessions(connection: sqlite3.Connection, sessions: Iterable[Session]) -> BackfillResult:
    """Persist source-confirmed interruption marks for already indexed sessions."""
    source_interruptions = 0
    matched_messages = 0
    changed_messages = 0
    seen: set[tuple[str, str]] = set()
    for session in sessions:
        for interruption in session.interruptions:
            if interruption.user_message_seq is None:
                continue
            key = (session.id, interruption.source_id)
            if key in seen:
                continue
            seen.add(key)
            source_interruptions += 1
            row = _find_user_message(connection, session.id, interruption.user_message_seq)
            if row is None:
                continue
            matched_messages += 1
            model = interruption.model
            if row[0] and row[1] == model:
                continue
            connection.execute(
                "UPDATE messages SET is_interrupted = 1, interruption_model = ? "
                "WHERE session_id = ? AND seq = ?",
                (model, session.id, interruption.user_message_seq),
            )
            changed_messages += 1
    return BackfillResult(source_interruptions, matched_messages, changed_messages, 0, None)


def preview(database: Path, sessions: Iterable[Session]) -> BackfillResult:
    """Count source interruptions that match indexed real user messages."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        result = _apply_sessions_read_only(connection, sessions)
        corrected = _correct_codex_markers(connection, write=False)
    return BackfillResult(
        result.source_interruptions,
        result.matched_messages,
        result.changed_messages,
        corrected,
        None,
    )


def _apply_sessions_read_only(
    connection: sqlite3.Connection, sessions: Iterable[Session]
) -> BackfillResult:
    """Count would-be marks using the same matching rules as the write path."""
    source_interruptions = 0
    matched_messages = 0
    changed_messages = 0
    seen: set[tuple[str, str]] = set()
    for session in sessions:
        for interruption in session.interruptions:
            if interruption.user_message_seq is None:
                continue
            key = (session.id, interruption.source_id)
            if key in seen:
                continue
            seen.add(key)
            source_interruptions += 1
            row = _find_user_message(connection, session.id, interruption.user_message_seq)
            if row is None:
                continue
            matched_messages += 1
            if not row[0] or row[1] != interruption.model:
                changed_messages += 1
    return BackfillResult(source_interruptions, matched_messages, changed_messages, 0, None)


def _correct_codex_markers(connection: sqlite3.Connection, *, write: bool) -> int:
    """Mark only legacy Codex user abort notices as injected and uninterruptible."""
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    if "sessions" not in tables or "is_interrupted" not in columns:
        return 0
    where = (
        "session_id IN (SELECT id FROM sessions WHERE agent = 'codex') "
        "AND role = 'user' AND content = ?"
    )
    changed = connection.execute(
        "SELECT COUNT(*) FROM messages WHERE "
        + where
        + " AND (is_injected = 0 OR is_interrupted != 0 OR interruption_model IS NOT NULL)",
        (_CODEX_ABORTED_MARKER,),
    ).fetchone()[0]
    if write and changed:
        connection.execute(
            "UPDATE messages SET is_injected = 1, is_interrupted = 0, "
            "interruption_model = NULL WHERE " + where,
            (_CODEX_ABORTED_MARKER,),
        )
    return int(changed)


def _find_user_message(
    connection: sqlite3.Connection, session_id: str, sequence: int
) -> tuple[int, str | None] | None:
    """Return an interruption state for one real indexed user message."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
    state = "is_interrupted, interruption_model" if "is_interrupted" in columns else "0, NULL"
    return connection.execute(
        f"SELECT {state} FROM messages WHERE session_id = ? AND seq = ? AND role = 'user' "
        "AND is_thinking = 0 AND is_system_instruction = 0 AND is_injected = 0",
        (session_id, sequence),
    ).fetchone()


def migrate(database: Path, sessions: Iterable[Session]) -> BackfillResult:
    """Back up, add columns, and backfill only interruption metadata."""
    backup_path = _create_verified_backup(database)
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection)
        corrected = _correct_codex_markers(connection, write=True)
        result = _apply_sessions(connection, sessions)
        connection.commit()
    return BackfillResult(
        result.source_interruptions,
        result.matched_messages,
        result.changed_messages,
        corrected,
        backup_path,
    )


def _load_sessions(home: Path) -> list[Session]:
    """Parse local archives only to extract interruption metadata."""
    return [parser(path) for path, parser in _source_paths(home)]


def main() -> int:
    """Preview by default, or back up and write after ``--write``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument(
        "--write", action="store_true", help="Create a backup and apply the backfill."
    )
    args = parser.parse_args()
    sessions = _load_sessions(args.home)
    result = migrate(args.database, sessions) if args.write else preview(args.database, sessions)
    print(f"Source interruptions: {result.source_interruptions}")
    print(f"Indexed user messages matched: {result.matched_messages}")
    print(f"Messages changed: {result.changed_messages}")
    print(f"Synthetic Codex markers corrected: {result.corrected_markers}")
    if result.backup_path is not None:
        print(f"Verified backup: {result.backup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
