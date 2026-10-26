"""Back up the archive and rebuild only message FTS; stop the backend before use."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

logger = logging.getLogger(__name__)


def repair(path: Path) -> Path:
    """Preserve a SQLite backup, then atomically repair searchable message entries."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.pre-fts-repair-{stamp}")
    with closing(sqlite3.connect(f"file:{path}?mode=rw", uri=True)) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        with closing(sqlite3.connect(backup)) as destination:
            conn.backup(destination)
        logger.info("Recoverable backup: %s", backup)
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError(
                "Structural corruption requires separate recovery; archive unchanged"
            )
        with conn:
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
            conn.execute(
                "INSERT INTO messages_fts(rowid, content) SELECT id, content FROM messages "
                "WHERE NOT is_thinking AND NOT is_system_instruction AND NOT is_injected"
            )
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('integrity-check')")
        logger.info("Repaired message FTS without reparsing, embedding, or enriching sessions")
    return backup


def main() -> None:
    """Repair the supplied offline database with a mandatory backup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    repair(args.database)


if __name__ == "__main__":
    main()
