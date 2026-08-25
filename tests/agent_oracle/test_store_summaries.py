"""Store tests for summary indexing, entity batching, and backfill.

Split from ``test_store.py`` to keep module sizes under the pylint limit.
"""

from __future__ import annotations

import pytest

# Helpers shared with the main store test module.
from test_store import _make_embedding, _make_message, _make_session

from agent_oracle.store import Store

# --------------------------------------------------------------------------- #
# set_summary
# --------------------------------------------------------------------------- #


def test_set_summary_updates_session(store: Store) -> None:
    """set_summary stores the summary on the session row."""
    store.index_session(_make_session())
    store.set_summary("sess-001", "A session about testing")

    row = store.conn.execute("SELECT summary FROM sessions WHERE id = ?", ("sess-001",)).fetchone()
    assert row[0] == "A session about testing"


def test_set_summary_updates_fts_index(store: Store) -> None:
    """set_summary indexes the summary so it becomes text-searchable."""
    store.index_session(_make_session())
    store.set_summary("sess-001", "A session about database migrations")

    results = store.search_text("migrations")
    assert [r["session_id"] for r in results] == ["sess-001"]
    assert "database" in results[0]["snippet"]


def test_set_summary_replaces_fts_entry(store: Store) -> None:
    """Updating a summary removes the old terms from the FTS index."""
    store.index_session(_make_session())
    store.set_summary("sess-001", "A session about migrations")
    store.set_summary("sess-001", "A session about refactoring")

    assert store.search_text("migrations") == []
    assert [r["session_id"] for r in store.search_text("refactoring")] == ["sess-001"]


def test_upsert_session_embedding_makes_summary_vector_searchable(store: Store) -> None:
    """Summary embeddings stored in vec_sessions are returned by search_vector."""
    store.index_session(_make_session())
    store.upsert_session_embedding("sess-001", _make_embedding(1.0, index=0))

    results = store.search_vector(_make_embedding(1.0, index=0))
    assert results[0]["session_id"] == "sess-001"
    assert results[0]["distance"] == 0.0


def test_search_text_merges_message_and_summary_hits(store: Store) -> None:
    """Text search returns both message-level and summary-level matches."""
    store.index_session(
        _make_session(session_id="s-msg", messages=[_make_message("refactoring the parser")])
    )
    store.index_session(_make_session(session_id="s-sum"))
    store.set_summary("s-sum", "Large refactoring effort")

    results = store.search_text("refactoring")
    session_ids = {r["session_id"] for r in results}
    assert session_ids == {"s-msg", "s-sum"}


# --------------------------------------------------------------------------- #
# entities / enriched flag
# --------------------------------------------------------------------------- #


def test_list_entities_groups_by_session(store: Store) -> None:
    """list_entities returns entities keyed by session id for many sessions at once."""
    store.index_session(_make_session(session_id="s1"))
    store.index_session(_make_session(session_id="s2"))
    store.upsert_entities("s1", [{"type": "product", "value": "SQLite"}])
    store.upsert_entities("s2", [{"type": "person", "value": "Ada"}])

    grouped = store.list_entities(["s1", "s2", "unknown"])
    assert grouped == {
        "s1": [{"entity_type": "product", "entity_value": "SQLite"}],
        "s2": [{"entity_type": "person", "entity_value": "Ada"}],
    }
    assert store.list_entities([]) == {}


def test_upsert_entities_marks_session_enriched(store: Store) -> None:
    """upsert_entities flips the enriched flag on the session row."""
    store.index_session(_make_session())
    store.upsert_entities("sess-001", [{"type": "product", "value": "SQLite"}])

    row = store.conn.execute("SELECT enriched FROM sessions WHERE id = ?", ("sess-001",)).fetchone()
    assert row[0] == 1


# --------------------------------------------------------------------------- #
# backfill_summary_indexes
# --------------------------------------------------------------------------- #


def _legacy_session(store: Store, session_id: str, summary: str) -> None:
    """Index a session and set its summary without touching the new indexes."""
    store.index_session(_make_session(session_id=session_id))
    store.conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))
    store.conn.commit()


def test_backfill_indexes_missing_summaries(store: Store) -> None:
    """Summaries stored before the summary indexes existed become searchable."""
    _legacy_session(store, "s1", "A session about database migrations")
    _legacy_session(store, "s2", "A session about refactoring")

    count = store.backfill_summary_indexes(
        lambda texts: [_make_embedding(1.0, index=i) for i, _ in enumerate(texts)]
    )
    assert count == 4  # 2 FTS + 2 vector entries
    assert store.search_text("migrations")[0]["session_id"] == "s1"
    assert store.search_vector(_make_embedding(1.0, index=0))[0]["session_id"] == "s1"


def test_backfill_skips_already_indexed(store: Store) -> None:
    """Sessions already present in both summary indexes are left untouched."""
    _legacy_session(store, "s1", "A session about migrations")
    store.backfill_summary_indexes(lambda texts: [_make_embedding(1.0) for _ in texts])
    store.backfill_summary_indexes(
        lambda texts: pytest.fail("embed_batch must not be called again")
    )

    assert len(store.search_vector(_make_embedding(1.0))) == 1


def test_backfill_marks_sessions_enriched(store: Store) -> None:
    """Backfilled sessions get the enriched flag set."""
    _legacy_session(store, "s1", "A session about migrations")
    store.backfill_summary_indexes(lambda texts: [_make_embedding(1.0) for _ in texts])

    row = store.conn.execute("SELECT enriched FROM sessions WHERE id = ?", ("s1",)).fetchone()
    assert row[0] == 1


def test_backfill_returns_zero_when_nothing_missing(store: Store) -> None:
    """Backfill is a no-op when every summary is already indexed."""
    assert store.backfill_summary_indexes(lambda texts: []) == 0


def test_backfill_never_indexes_or_embeds_review_summaries(store: Store) -> None:
    """Review summaries remain absent from both summary indexes during backfill."""
    _legacy_session(store, "regular", "regular summary")
    _legacy_session(store, "review", "review summary")
    store.conn.execute("UPDATE sessions SET is_review_agent = 1 WHERE id = 'review'")
    store.conn.commit()

    embedded: list[list[str]] = []
    count = store.backfill_summary_indexes(
        lambda texts: embedded.append(texts) or [_make_embedding(1.0) for _ in texts]
    )

    assert count == 2
    assert embedded == [["regular summary"]]
    review_rowid = store.conn.execute("SELECT rowid FROM sessions WHERE id = 'review'").fetchone()[
        0
    ]
    assert (
        store.conn.execute("SELECT 1 FROM sessions_fts WHERE rowid = ?", (review_rowid,)).fetchone()
        is None
    )
    assert (
        store.conn.execute("SELECT 1 FROM vec_sessions WHERE rowid = ?", (review_rowid,)).fetchone()
        is None
    )
