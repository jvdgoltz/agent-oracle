"""Tests for the session file watcher."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from watchdog.events import FileModifiedEvent

from agent_oracle import watcher as watcher_module
from agent_oracle.embed import Embedder
from agent_oracle.enrich import Enricher, EnrichmentResult, Entity
from agent_oracle.models import AgentType, Session
from agent_oracle.sources.claude import parse_claude_session
from agent_oracle.sources.codex import parse_codex_session
from agent_oracle.sources.factory import parse_factory_session
from agent_oracle.store import Store
from agent_oracle.watcher import _HOME, SessionWatcher


def _codex_jsonl(tmp_path: Path) -> Path:
    """Write a minimal Codex session JSONL and return its path."""
    path = tmp_path / "rollout-123.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "codex-sess", "cwd": "/tmp", "timestamp": "2024-01-01T00:00:00Z"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
                "timestamp": "2024-01-01T00:00:01Z",
            }
        )
        + "\n"
    )
    return path


def _factory_jsonl(tmp_path: Path) -> Path:
    """Write a minimal Factory session JSONL and return its path."""
    path = tmp_path / "fact-sess.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session_start",
                "id": "fact-sess",
                "cwd": "/tmp",
                "title": "Demo",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "message",
                "id": "msg-1",
                "timestamp": "2024-01-01T00:00:01Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            }
        )
        + "\n"
    )
    return path


def _claude_jsonl(tmp_path: Path) -> Path:
    """Write a minimal Claude session JSONL and return its path."""
    path = tmp_path / "claude-sess.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "user",
                "sessionId": "claude-sess",
                "cwd": "/tmp",
                "timestamp": "2024-01-01T00:00:01Z",
                "message": {"role": "user", "content": "hi there"},
            }
        )
        + "\n"
    )
    return path


def _make_watcher() -> tuple[SessionWatcher, MagicMock, MagicMock, MagicMock]:
    """Build a watcher with fully mocked store, embedder, and enricher."""
    store = MagicMock(spec=Store)
    embedder = MagicMock(spec=Embedder)
    enricher = MagicMock(spec=Enricher)
    watcher = SessionWatcher(store=store, embedder=embedder, enricher=enricher)
    return watcher, store, embedder, enricher


def _map_dirs(tmp_path: Path) -> dict[AgentType, Path]:
    """Point the watcher's watched directories at *tmp_path* subdirectories."""
    dirs = {
        AgentType.CODEX: tmp_path / ".codex" / "sessions",
        AgentType.FACTORY: tmp_path / ".factory" / "sessions",
        AgentType.CLAUDE: tmp_path / ".claude" / "projects",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def test_detect_parser_by_path() -> None:
    """_detect_parser selects the normalizer matching each watched directory."""
    watcher, _, _, _ = _make_watcher()
    codex_dir = _HOME / ".codex" / "sessions"
    factory_dir = _HOME / ".factory" / "sessions"
    claude_dir = _HOME / ".claude" / "projects"

    assert watcher._detect_parser(codex_dir / "a.jsonl") is parse_codex_session
    assert watcher._detect_parser(factory_dir / "a.jsonl") is parse_factory_session
    assert watcher._detect_parser(claude_dir / "a.jsonl") is parse_claude_session
    assert watcher._detect_parser(Path("/elsewhere/session.jsonl")) is None


@pytest.mark.parametrize(
    "path",
    ["/tmp/nonexistent", "/elsewhere/out.jsonl"],
    ids=["not-jsonl", "outside-watched-dirs"],
)
def test_index_file_skips_unwatched(path: str) -> None:
    """_index_file ignores non-JSONL files and files outside watched dirs."""
    watcher, store, embedder, enricher = _make_watcher()
    watcher._index_file(Path(path))
    store.index_session.assert_not_called()
    embedder.embed.assert_not_called()
    enricher.enrich.assert_not_called()


def test_index_file_routes_to_codex_normalizer(tmp_path: Path) -> None:
    """A .jsonl under the codex dir is parsed, indexed, embedded, and enriched."""
    codex_dir = tmp_path / ".codex" / "sessions"
    codex_dir.mkdir(parents=True)
    source = _codex_jsonl(codex_dir)
    watcher, store, embedder, enricher = _make_watcher()

    store.get_session.return_value = {"messages": [{"id": 1, "content": "hi"}]}
    embedder.embed_batch.return_value = [[0.1, 0.2]]
    enricher.enrich.return_value = EnrichmentResult(
        summary="s", entities=[Entity(type="product", value="SQLite")]
    )

    with patch.object(watcher_module, "_HOME", tmp_path):
        watcher._index_file(source)

    store.index_session.assert_called_once()
    session = store.index_session.call_args.args[0]
    assert isinstance(session, Session)
    assert session.agent == AgentType.CODEX
    embedder.embed_batch.assert_called_once_with(["hi"])
    store.upsert_embedding.assert_called_once_with(1, [0.1, 0.2])
    enricher.enrich.assert_called_once()
    store.set_summary.assert_called_once_with("codex-sess", "s")
    store.upsert_entities.assert_called_once_with(
        "codex-sess", [{"type": "product", "value": "SQLite"}]
    )


def test_index_file_routes_to_factory_normalizer(tmp_path: Path) -> None:
    """A .jsonl under the factory dir is parsed with the factory normalizer."""
    factory_dir = tmp_path / ".factory" / "sessions"
    factory_dir.mkdir(parents=True)
    source = _factory_jsonl(factory_dir)
    watcher, store, _embedder, enricher = _make_watcher()
    store.get_session.return_value = {"messages": []}
    enricher.enrich.return_value = EnrichmentResult(summary="", entities=[])

    with patch.object(watcher_module, "_HOME", tmp_path):
        watcher._index_file(source)

    session = store.index_session.call_args.args[0]
    assert isinstance(session, Session)
    assert session.agent == AgentType.FACTORY
    assert enricher.enrich.call_args.args[0].id == "fact-sess"


def test_index_file_routes_to_claude_normalizer(tmp_path: Path) -> None:
    """A .jsonl under the claude dir is parsed with the claude normalizer."""
    claude_dir = tmp_path / ".claude" / "projects"
    claude_dir.mkdir(parents=True)
    source = _claude_jsonl(claude_dir)
    watcher, store, _embedder, enricher = _make_watcher()
    store.get_session.return_value = {"messages": []}
    enricher.enrich.return_value = EnrichmentResult(summary="", entities=[])

    with patch.object(watcher_module, "_HOME", tmp_path):
        watcher._index_file(source)

    session = store.index_session.call_args.args[0]
    assert isinstance(session, Session)
    assert session.agent == AgentType.CLAUDE
    assert enricher.enrich.call_args.args[0].id == "claude-sess"


def test_index_file_continues_on_failure(tmp_path: Path) -> None:
    """A failed parse is logged and does not raise out of the watcher."""
    codex_dir = tmp_path / ".codex" / "sessions"
    codex_dir.mkdir(parents=True)
    bad = codex_dir / "bad.jsonl"
    bad.write_text("not valid json\n")
    watcher, store, _, _ = _make_watcher()

    with patch.object(watcher_module, "_HOME", tmp_path):
        watcher._index_file(bad)

    store.index_session.assert_not_called()


def test_index_existing_discovers_all_three_dirs(tmp_path: Path) -> None:
    """index_existing walks each watched directory for .jsonl files."""
    dirs = _map_dirs(tmp_path)
    codex = _codex_jsonl(dirs[AgentType.CODEX])
    factory = _factory_jsonl(dirs[AgentType.FACTORY])
    claude = _claude_jsonl(dirs[AgentType.CLAUDE])
    # A non-JSONL file must be skipped.
    (dirs[AgentType.CODEX] / "notes.txt").write_text("ignore")

    watcher, store, _embedder, enricher = _make_watcher()
    store.get_session.return_value = {"messages": []}
    enricher.enrich.return_value = EnrichmentResult(summary="", entities=[])

    with patch.object(watcher_module, "_HOME", tmp_path):
        watcher.index_existing()

    assert store.index_session.call_count == 3
    indexed_ids = {call.args[0].id for call in store.index_session.call_args_list}
    assert indexed_ids == {"codex-sess", "fact-sess", "claude-sess"}
    assert codex.exists()
    assert factory.exists()
    assert claude.exists()


def test_debounce_cancels_and_reschedules_timer() -> None:
    """Modify events for the same path cancel the prior timer and reschedule."""
    watcher, _, _, _ = _make_watcher()
    watcher.debounce_seconds = 10.0
    path = _HOME / ".codex" / "sessions" / "rollout-1.jsonl"

    created: list[MagicMock] = []
    real_timer = threading.Timer

    def fake_timer(_delay, _fn, args):
        timer = MagicMock(spec=real_timer)
        timer.args = args
        created.append(timer)
        return timer

    with patch.object(watcher_module.threading, "Timer", side_effect=fake_timer):
        watcher.debounce(path)
        watcher.debounce(path)

    assert len(created) == 2
    created[0].cancel.assert_called_once()
    created[1].cancel.assert_not_called()
    created[1].start.assert_called_once()
    assert watcher._timers[str(path)] is created[1]


def test_on_modified_uses_src_path(tmp_path: Path) -> None:
    """_on_modified wires the event src_path into the debounce schedule."""
    watcher, _, _, _ = _make_watcher()
    event = FileModifiedEvent("/tmp/rollout-abc.jsonl")
    with patch.object(watcher, "debounce") as debounce:
        watcher._on_modified(event)
    debounce.assert_called_once_with(Path("/tmp/rollout-abc.jsonl"))
