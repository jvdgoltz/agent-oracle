#!/usr/bin/env python3
"""Remove excluded messages from search indexes without re-indexing archives."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from agent_oracle.sources.common import is_injected_message

DEFAULT_DATABASE = Path.home() / ".agent-oracle" / "index.db"


@dataclass(frozen=True, slots=True)
class RepairResult:
    """Report rows classified, index entries removed, and affected sessions."""

    matched_messages: int
    newly_flagged_messages: int
    deleted_fts: int
    deleted_vectors: int
    invalidated_sessions: int
    backup_path: Path | None = None


def _create_verified_backup(database: Path) -> Path:
    """Create and integrity-check a backup while the write lock is held."""
    backup = database.with_name(f"{database.name}.excluded-content.bak.{time.time_ns()}")
    with sqlite3.connect(backup) as target:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
            if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Source database integrity check failed.")
            source.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup database integrity check failed.")
    return backup


def _excluded_ids(connection: sqlite3.Connection) -> list[int]:
    """Return message IDs that must not be indexed or sent to enrichment."""
    rows = connection.execute(
        "SELECT id, role, content, is_thinking, is_system_instruction, is_injected FROM messages"
    ).fetchall()
    return [
        row[0]
        for row in rows
        if row[1] in ("system", "developer")
        or row[3]
        or row[4]
        or row[5]
        or (row[1] == "user" and is_injected_message(row[2]))
    ]


def _stale_fts_count(connection: sqlite3.Connection, message_ids: list[int]) -> int:
    """Count excluded rows still returned by FTS after external-content tombstones."""
    count = 0
    for message_id in message_ids:
        row = connection.execute(
            "SELECT content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        token = re.search(r"[\w-]+", row[0]) if row else None
        if (
            token
            and connection.execute(
                "SELECT 1 FROM messages_fts WHERE rowid = ? AND messages_fts MATCH ? LIMIT 1",
                (message_id, f'"{token.group(0)}"'),
            ).fetchone()
        ):
            count += 1
    return count


def _apply(connection: sqlite3.Connection, *, write: bool) -> RepairResult:
    """Preview or apply targeted flags, index deletions, and invalidation."""
    excluded = _excluded_ids(connection)
    if not excluded:
        return RepairResult(0, 0, 0, 0, 0)
    placeholders = ",".join("?" for _ in excluded)
    params = [*excluded]
    sessions = [
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT session_id FROM messages WHERE id IN ({placeholders})", params
        )
    ]
    newly_flagged = 0
    for row in connection.execute(
        f"SELECT role, content, is_system_instruction, is_injected "
        f"FROM messages WHERE id IN ({placeholders})",
        params,
    ):
        expected_system = int(row[0] in ("system", "developer"))
        expected_injected = int(row[0] == "user" and is_injected_message(row[1]))
        newly_flagged += int(expected_system and not row[2])
        newly_flagged += int(expected_injected and not row[3])
    fts_count = _stale_fts_count(connection, excluded)
    vec_count = connection.execute(
        f"SELECT COUNT(*) FROM vec_messages WHERE rowid IN ({placeholders})", params
    ).fetchone()[0]
    stale_sessions = len(sessions) if fts_count or vec_count else 0
    if not write:
        return RepairResult(len(excluded), newly_flagged, fts_count, vec_count, stale_sessions)

    # Classify each selected row using the same prefix-safe Python classifier
    # used by source parsers, while retaining the update scope to selected IDs.
    for message_id in excluded:
        row = connection.execute(
            "SELECT role, content FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        connection.execute(
            "UPDATE messages SET is_system_instruction = is_system_instruction OR ?, "
            "is_injected = is_injected OR ? WHERE id = ?",
            (
                int(row[0] in ("system", "developer")),
                int(row[0] == "user" and is_injected_message(row[1])),
                message_id,
            ),
        )
    fts_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
    ).fetchone()[0]
    if "content='messages'" in fts_sql.replace(" ", ""):
        for message_id in excluded:
            content = connection.execute(
                "SELECT content FROM messages WHERE id = ?", (message_id,)
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', ?, ?)",
                (message_id, content),
            )
    else:
        connection.execute(f"DELETE FROM messages_fts WHERE rowid IN ({placeholders})", params)
    connection.execute(f"DELETE FROM vec_messages WHERE rowid IN ({placeholders})", params)
    session_placeholders = ",".join("?" for _ in sessions)
    session_rowids = [
        row[0]
        for row in connection.execute(
            f"SELECT rowid FROM sessions WHERE id IN ({session_placeholders})", sessions
        )
    ]
    for rowid in session_rowids:
        connection.execute("DELETE FROM sessions_fts WHERE rowid = ?", (rowid,))
    connection.execute(
        f"DELETE FROM vec_sessions WHERE rowid IN "
        f"(SELECT rowid FROM sessions WHERE id IN ({session_placeholders}))",
        sessions,
    )
    connection.execute(
        f"DELETE FROM entities WHERE session_id IN ({session_placeholders})", sessions
    )
    connection.execute(
        f"UPDATE sessions SET summary = NULL, enriched = 0 WHERE id IN ({session_placeholders})",
        sessions,
    )
    return RepairResult(len(excluded), newly_flagged, fts_count, vec_count, len(sessions))


def _connect(database: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a database connection with the production sqlite-vec extension."""
    target = f"file:{database}?mode=ro" if read_only else str(database)
    connection = sqlite3.connect(target, uri=read_only)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    return connection


def preview(database: Path) -> RepairResult:
    """Preview affected rows without modifying *database*."""
    with _connect(database, read_only=True) as connection:
        return _apply(connection, write=False)


def repair(database: Path) -> RepairResult:
    """Back up and apply the targeted repair in one transaction."""
    with _connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            backup = _create_verified_backup(database)
            result = _apply(connection, write=True)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return RepairResult(
        result.matched_messages,
        result.newly_flagged_messages,
        result.deleted_fts,
        result.deleted_vectors,
        result.invalidated_sessions,
        backup,
    )


def main() -> int:
    """Preview by default; apply only when ``--write`` is provided."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = repair(args.database) if args.write else preview(args.database)
    fields = (
        "matched_messages",
        "newly_flagged_messages",
        "deleted_fts",
        "deleted_vectors",
        "invalidated_sessions",
    )
    for field in fields:
        print(f"{field}: {getattr(result, field)}")
    if result.backup_path:
        print(f"verified_backup: {result.backup_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
