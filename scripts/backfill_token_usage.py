#!/usr/bin/env python3
"""Backfill provider-reported token usage without re-indexing sessions."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from agent_oracle.models import Session
from agent_oracle.sources.claude import parse_claude_session
from agent_oracle.sources.codex import parse_codex_session
from agent_oracle.sources.factory import parse_factory_session
from agent_oracle.sources.omp import parse_omp_session
from agent_oracle.sources.pi import parse_pi_session

DEFAULT_DATABASE = Path.home() / ".agent-oracle" / "index.db"
Parser = Callable[[Path], Session]


def _source_paths(home: Path) -> Iterable[tuple[Path, Parser]]:
    """Yield local archive files and their parser."""
    sources = (
        (home / ".codex" / "sessions", parse_codex_session),
        (home / ".factory" / "sessions", parse_factory_session),
        (home / ".claude" / "projects", parse_claude_session),
        (home / ".omp" / "agent" / "sessions", parse_omp_session),
        (home / ".pi" / "agent" / "sessions", parse_pi_session),
    )
    for directory, parser in sources:
        if directory.exists():
            yield from ((path, parser) for path in sorted(directory.rglob("*.jsonl")))


def _backup(database: Path) -> Path:
    """Create and integrity-check a backup before migration writes."""
    backup_path = database.with_name(f"{database.name}.token-usage.bak.{time.time_ns()}")
    with sqlite3.connect(database) as source, sqlite3.connect(backup_path) as backup:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Source database integrity check failed")
        source.backup(backup)
        if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup database integrity check failed")
    return backup_path


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create only the additive token usage table and index."""
    connection.execute("""CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL, model TEXT, input_tokens INTEGER,
            output_tokens INTEGER, cached_input_tokens INTEGER,
            cache_creation_input_tokens INTEGER, cache_read_input_tokens INTEGER,
            reasoning_output_tokens INTEGER, total_tokens INTEGER
        )""")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_token_usage_session ON token_usage(session_id)"
    )


def _count(connection: sqlite3.Connection, sessions: Iterable[Session]) -> tuple[int, int]:
    """Return source usage count and rows that match indexed sessions."""
    source = matched = 0
    for session in sessions:
        source += len(session.token_usages)
        if connection.execute("SELECT 1 FROM sessions WHERE id = ?", (session.id,)).fetchone():
            matched += len(session.token_usages)
    return source, matched


def preview(database: Path, sessions: Iterable[Session]) -> tuple[int, int]:
    """Report usage rows that would be inserted."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        return _count(connection, sessions)


def migrate(database: Path, sessions: Iterable[Session]) -> tuple[int, int, Path]:
    """Back up and replace usage rows for indexed sessions only."""
    sessions = list(sessions)
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        backup_path = _backup(database)
        _ensure_schema(connection)
        source, matched = _count(connection, sessions)
        for session in sessions:
            if not connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session.id,)
            ).fetchone():
                continue
            connection.execute("DELETE FROM token_usage WHERE session_id = ?", (session.id,))
            connection.executemany(
                "INSERT INTO token_usage (session_id,timestamp,model,input_tokens,output_tokens,"
                "cached_input_tokens,cache_creation_input_tokens,cache_read_input_tokens,"
                "reasoning_output_tokens,total_tokens) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        session.id,
                        u.timestamp.isoformat(),
                        u.model,
                        u.input_tokens,
                        u.output_tokens,
                        u.cached_input_tokens,
                        u.cache_creation_input_tokens,
                        u.cache_read_input_tokens,
                        u.reasoning_output_tokens,
                        u.total_tokens,
                    )
                    for u in session.token_usages
                ],
            )
        connection.commit()
    return source, matched, backup_path


def main() -> int:
    """Preview by default; write only with explicit ``--write``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sessions = [parser(path) for path, parser in _source_paths(args.home)]
    result = (
        migrate(args.database, sessions)
        if args.write
        else (*preview(args.database, sessions), None)
    )
    print(f"Source usage rows: {result[0]}")
    print(f"Indexed usage rows matched: {result[1]}")
    if result[2] is not None:
        print(f"Verified backup: {result[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
