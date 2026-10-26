#!/usr/bin/env python3
"""Backfill Codex review-session metadata without re-indexing archives."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent_oracle.sources.codex import parse_codex_session

DEFAULT_DATABASE = Path.home() / ".agent-oracle" / "index.db"


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Count updated sessions and provide the optional backup path."""

    matched_sessions: int
    changed_sessions: int
    review_sessions: int
    backup_path: Path | None


def _create_verified_backup(database: Path) -> Path:
    """Create and integrity-check a consistent database backup."""
    backup_path = database.with_name(f"{database.name}.review-sessions.bak.{time.time_ns()}")
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


def _source_sessions(home: Path):
    """Yield parsed Codex sessions from local archive files."""
    root = home / ".codex" / "sessions"
    if root.exists():
        yield from (parse_codex_session(path) for path in sorted(root.rglob("*.jsonl")))


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Add nullable review metadata columns to the sessions table."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
    if "parent_thread_id" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN parent_thread_id TEXT")
    if "is_review_agent" not in columns:
        connection.execute(
            "ALTER TABLE sessions ADD COLUMN is_review_agent INTEGER NOT NULL DEFAULT 0"
        )


def _apply(connection: sqlite3.Connection, home: Path, *, write: bool) -> BackfillResult:
    """Compare source metadata to indexed sessions and optionally persist it."""
    matched = changed = reviews = 0
    columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
    parent_thread_id = "parent_thread_id" if "parent_thread_id" in columns else "NULL"
    is_review_agent = "is_review_agent" if "is_review_agent" in columns else "0"
    for session in _source_sessions(home):
        row = connection.execute(
            f"SELECT {parent_thread_id}, {is_review_agent} FROM sessions "
            "WHERE id = ? AND agent = 'codex'",
            (session.id,),
        ).fetchone()
        if row is None:
            continue
        matched += 1
        if session.is_review_agent:
            reviews += 1
        metadata = (session.parent_thread_id, session.is_review_agent)
        if tuple(row) != metadata:
            changed += 1
            if write:
                connection.execute(
                    "UPDATE sessions SET parent_thread_id = ?, is_review_agent = ? WHERE id = ?",
                    (*metadata, session.id),
                )
    return BackfillResult(matched, changed, reviews, None)


def preview(database: Path, home: Path) -> BackfillResult:
    """Report the metadata update without modifying the database."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        return _apply(connection, home, write=False)


def migrate(database: Path, home: Path) -> BackfillResult:
    """Back up, add columns, and backfill Codex review-session metadata."""
    backup_path = _create_verified_backup(database)
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection)
        result = _apply(connection, home, write=True)
        connection.commit()
    return BackfillResult(
        result.matched_sessions, result.changed_sessions, result.review_sessions, backup_path
    )


def main() -> int:
    """Preview by default, or apply the backfill after ``--write``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--write", action="store_true", help="Create a backup and apply changes.")
    args = parser.parse_args()
    result = migrate(args.database, args.home) if args.write else preview(args.database, args.home)
    print(f"Indexed Codex sessions matched: {result.matched_sessions}")
    print(f"Sessions changed: {result.changed_sessions}")
    print(f"Review sessions: {result.review_sessions}")
    if result.backup_path is not None:
        print(f"Verified backup: {result.backup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
