"""Tests for Pi archive discovery and watcher routing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_oracle import watcher as watcher_module
from agent_oracle.models import AgentType
from agent_oracle.sources.pi import parse_pi_session
from agent_oracle.watcher import SessionWatcher, _watched_dirs


def _pi_jsonl(directory: Path) -> Path:
    """Write a minimal Pi session JSONL."""
    directory.mkdir(parents=True)
    path = directory / "pi-sess.jsonl"
    records = [
        {
            "type": "session",
            "id": "pi-sess",
            "cwd": "/tmp",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "type": "message",
            "id": "u1",
            "timestamp": "2024-01-01T00:00:01Z",
            "message": {"role": "user", "content": "hi there"},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_pi_directory_and_parser_are_registered(tmp_path: Path) -> None:
    """Watcher discovery maps Pi archives to the Pi parser."""
    with patch.object(watcher_module, "_HOME", tmp_path):
        directory = _watched_dirs()[AgentType.PI]
        watcher = SessionWatcher(MagicMock(), MagicMock(), MagicMock())
        assert directory == tmp_path / ".pi" / "agent" / "sessions"
        assert watcher._detect_parser(directory / "a.jsonl") is parse_pi_session


def test_index_file_routes_to_pi_normalizer(tmp_path: Path) -> None:
    """A Pi JSONL file is normalized and stored."""
    source = _pi_jsonl(tmp_path / ".pi" / "agent" / "sessions")
    store = MagicMock()
    store.get_session.return_value = {"messages": []}
    watcher = SessionWatcher(store, MagicMock(), MagicMock())

    with patch.object(watcher_module, "_HOME", tmp_path):
        watcher._index_file(source)

    assert store.index_session.call_args.args[0].agent is AgentType.PI
