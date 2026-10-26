"""Tests for SQLite transaction isolation."""

import threading
from concurrent.futures import ThreadPoolExecutor

from agent_oracle.database import StoreDatabase, serialized
from agent_oracle.store import Store


class _Probe(StoreDatabase):
    """Exercise connection locking with an observable transaction boundary."""

    @serialized
    def hold(self, entered: threading.Event, release: threading.Event) -> None:
        """Keep one operation open until released by the test."""
        entered.set()
        assert release.wait(2)

    @serialized
    def observe(self, entered: threading.Event) -> None:
        """Signal when a competing operation enters the connection."""
        entered.set()


def test_operations_cannot_interleave(tmp_path) -> None:
    """A second thread cannot read or commit another operation's transaction."""
    db = _Probe(tmp_path / "index.db")
    entered, release, observed = (threading.Event() for _ in range(3))
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(db.hold, entered, release)
        assert entered.wait(2)
        second = executor.submit(db.observe, observed)
        try:
            assert not observed.wait(0.05)
        finally:
            release.set()
        first.result()
        second.result()
    assert observed.is_set()
    db.conn.close()


def test_init_creates_tables(store: Store) -> None:
    """All expected tables exist after initialization."""
    names = {
        r[0]
        for r in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
        ).fetchall()
    }
    assert "sessions" in names
    assert "messages" in names
    assert "messages_fts" in names
    assert "sessions_fts" in names
    assert "vec_sessions" in names
    assert "vec_messages" in names
    assert "entities" in names


def test_init_creates_indexes(store: Store) -> None:
    """Indexes on session_id exist for messages and entities."""
    names = {
        r[0]
        for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    }
    assert "idx_messages_session_id" in names
    assert "idx_entities_session_id" in names
