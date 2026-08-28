"""Tests for the provider token-usage migration."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from agent_oracle.models import AgentType, Session, TokenUsage


def _load_module() -> ModuleType:
    """Load the standalone migration script for direct testing."""
    script = Path(__file__).parents[2] / "scripts" / "backfill_token_usage.py"
    spec = importlib.util.spec_from_file_location("backfill_token_usage", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migrate_backs_up_and_is_idempotent(tmp_path: Path) -> None:
    """Migration backs up under its write lock and replaces matching usage rows."""
    database = tmp_path / "index.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO sessions VALUES ('session-1')")
    session = Session(
        id="session-1",
        agent=AgentType.CLAUDE,
        cwd="/work",
        started_at=datetime(2026, 8, 1, tzinfo=UTC),
        token_usages=[
            TokenUsage(
                timestamp=datetime(2026, 8, 1, tzinfo=UTC),
                model="claude-test",
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            )
        ],
    )
    migration = _load_module()

    first = migration.migrate(database, [session])
    second = migration.migrate(database, [session])

    assert first[:2] == (1, 1)
    assert first[2].exists()
    assert second[:2] == (1, 1)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0] == 1
