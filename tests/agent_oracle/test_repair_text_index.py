"""Tests for recoverable repair of the derived message text index."""

import importlib.util
import sqlite3
from pathlib import Path

from agent_oracle.store import Store


def test_repair_preserves_archive_and_creates_backup(tmp_path: Path) -> None:
    """Repair removes stale text entries without changing source rows or vectors."""
    script = Path(__file__).parents[2] / "scripts" / "repair_text_index.py"
    spec = importlib.util.spec_from_file_location("repair_text_index", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "index.db"
    store = Store(path)
    store.conn.execute("INSERT INTO sessions(id) VALUES('session')")
    store.conn.execute(
        "INSERT INTO messages(id,session_id,role,content,is_thinking) "
        "VALUES(1,'session','user','visible',0),(2,'session','assistant','hidden',1)"
    )
    store.conn.execute("INSERT INTO messages_fts(rowid,content) VALUES(2,'hidden')")
    store.conn.commit()
    store.conn.close()
    backup = module.repair(path)
    with sqlite3.connect(backup) as old:
        assert old.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
        assert old.execute("SELECT id FROM messages_fts_docsize").fetchall() == [(2,)]
    restored = Store(path)
    assert restored.conn.execute("SELECT id FROM messages_fts_docsize").fetchall()[0][0] == 1
    assert restored.conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
    restored.conn.close()
