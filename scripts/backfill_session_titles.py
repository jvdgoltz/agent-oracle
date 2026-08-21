"""Add and populate the session title column without re-indexing sessions."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from agent_oracle.models import Session
from agent_oracle.sources.claude import parse_claude_session
from agent_oracle.sources.codex import parse_codex_session
from agent_oracle.sources.factory import parse_factory_session
from agent_oracle.sources.omp import parse_omp_session

logger = logging.getLogger(__name__)


def _sessions(home: Path) -> Iterator[Session]:
    """Yield parsed sessions from every supported local agent directory."""
    sources = (
        (home / ".codex" / "sessions", parse_codex_session),
        (home / ".factory" / "sessions", parse_factory_session),
        (home / ".claude" / "projects", parse_claude_session),
        (home / ".omp" / "agent" / "sessions", parse_omp_session),
    )
    for root, parser in sources:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            yield parser(path)


def _backup_database(database: Path) -> Path:
    """Create a consistent SQLite backup beside the source database."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = database.with_name(f"{database.stem}.pre-title-{timestamp}{database.suffix}")
    with sqlite3.connect(database) as source, sqlite3.connect(backup) as destination:
        source.backup(destination)
    return backup


def migrate_titles(database: Path, home: Path) -> tuple[Path, int]:
    """Back up *database*, add ``sessions.title``, and populate matching rows."""
    backup = _backup_database(database)
    updated = 0
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        if "title" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        for session in _sessions(home):
            if session.title is None:
                continue
            cursor = connection.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (session.title, session.id)
            )
            updated += cursor.rowcount
    return backup, updated


def main() -> None:
    """Run the backup-first title migration for the local Agent Oracle database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path.home() / ".agent-oracle" / "index.db")
    parser.add_argument("--home", type=Path, default=Path.home())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    backup, updated = migrate_titles(args.database, args.home)
    logger.info("Backed up database to %s and populated %d session titles", backup, updated)


if __name__ == "__main__":
    main()
