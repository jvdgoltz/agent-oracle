"""Tests for the existing entity normalization migration."""

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import ModuleType


def _load_migration_module() -> ModuleType:
    """Load the standalone migration script for direct testing."""
    script = Path(__file__).parents[2] / "scripts" / "normalize_entity_values.py"
    spec = importlib.util.spec_from_file_location("normalize_entity_values", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_database(path: Path) -> None:
    """Create the minimal entities schema needed by the migration."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE entities (id INTEGER PRIMARY KEY, session_id TEXT, "
            "entity_type TEXT, entity_value TEXT)"
        )
        connection.executemany(
            "INSERT INTO entities VALUES (?, ?, ?, ?)",
            [
                (1, "one", "product", "SQLite"),
                (2, "one", "product", "sqlite"),
                (3, "two", "person", "Ada Lovelace"),
            ],
        )


def test_preview_reports_changes_and_collisions_without_writing(tmp_path: Path) -> None:
    """Preview reports normalization effects without changing the database."""
    database = tmp_path / "index.db"
    _create_database(database)
    migration = _load_migration_module()

    preview = migration.preview(database)

    assert preview.changed_rows == 2
    assert preview.collision_groups == 1
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT entity_value FROM entities WHERE id = 1").fetchone()
    assert row == ("SQLite",)


def test_migrate_creates_verified_backup_and_preserves_collision_rows(tmp_path: Path) -> None:
    """Migration writes normalized values while retaining rows without a unique constraint."""
    database = tmp_path / "index.db"
    _create_database(database)
    migration = _load_migration_module()

    result = migration.migrate(database)

    assert result.changed_rows == 2
    assert result.backup_path.exists()
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT entity_value FROM entities ORDER BY id").fetchall()
    assert rows == [("sqlite",), ("sqlite",), ("ada-lovelace",)]
