#!/usr/bin/env python3
"""Normalize existing entity values without re-indexing archived sessions."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent_oracle.enrich import normalize_entity_value

DEFAULT_DATABASE = Path.home() / ".agent-oracle" / "index.db"


@dataclass(frozen=True, slots=True)
class Preview:
    """Counts produced by inspecting entity normalization effects."""

    changed_rows: int
    collision_groups: int
    collision_rows: int


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Counts and backup location produced by a successful migration."""

    changed_rows: int
    collision_groups: int
    collision_rows: int
    backup_path: Path


def preview(database: Path) -> Preview:
    """Inspect normalization effects without modifying *database*."""
    with _connect_read_only(database) as connection:
        rows = connection.execute(
            "SELECT id, session_id, entity_type, entity_value FROM entities"
        ).fetchall()
    changed_rows = sum(normalize_entity_value(row[3]) != row[3] for row in rows)
    groups: dict[tuple[str, str, str], int] = {}
    for _, session_id, entity_type, entity_value in rows:
        key = (session_id, entity_type, normalize_entity_value(entity_value))
        groups[key] = groups.get(key, 0) + 1
    duplicate_counts = [count for count in groups.values() if count > 1]
    return Preview(
        changed_rows=changed_rows,
        collision_groups=len(duplicate_counts),
        collision_rows=sum(duplicate_counts),
    )


def migrate(database: Path) -> MigrationResult:
    """Back up *database* and normalize entity values in one transaction."""
    report = preview(database)
    backup_path = _create_verified_backup(database)
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT id, entity_value FROM entities").fetchall()
        changes = [
            (normalize_entity_value(entity_value), entity_id)
            for entity_id, entity_value in rows
            if normalize_entity_value(entity_value) != entity_value
        ]
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany("UPDATE entities SET entity_value = ? WHERE id = ?", changes)
        connection.commit()
    return MigrationResult(
        changed_rows=len(changes),
        collision_groups=report.collision_groups,
        collision_rows=report.collision_rows,
        backup_path=backup_path,
    )


def _connect_read_only(database: Path) -> sqlite3.Connection:
    """Open *database* read-only, refusing to create a missing database."""
    if not database.is_file():
        raise FileNotFoundError(f"Database not found: {database}")
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def _create_verified_backup(database: Path) -> Path:
    """Create and integrity-check a consistent SQLite backup before a write."""
    backup_path = database.with_name(f"{database.name}.entity-values.bak.{time.time_ns()}")
    with _connect_read_only(database) as source, sqlite3.connect(backup_path) as backup:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Source database integrity check failed.")
        source.backup(backup)
        if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup database integrity check failed.")
    return backup_path


def _format_report(report: Preview | MigrationResult) -> str:
    """Format aggregate counts without exposing entity values."""
    lines = [
        f"Rows needing normalization: {report.changed_rows}",
        f"Normalization collision groups: {report.collision_groups}",
        f"Rows in collision groups: {report.collision_rows}",
    ]
    if isinstance(report, MigrationResult):
        lines.append(f"Verified backup: {report.backup_path}")
    return "\n".join(lines)


def main() -> int:
    """Run a dry preview by default, or write after an explicit --write flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--write", action="store_true", help="Back up and apply the migration.")
    args = parser.parse_args()
    report = migrate(args.database) if args.write else preview(args.database)
    print(_format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
