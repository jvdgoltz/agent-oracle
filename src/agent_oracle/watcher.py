"""File watcher for Agent Oracle.

Watches the session directories of Codex, Factory Droid, and Claude Code and
indexes new or modified session files into the store.  A debounce collapses the
burst of modify events emitted during a write into a single re-index per file.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from agent_oracle.embed import Embedder
from agent_oracle.enrich import Enricher, EnrichmentResult
from agent_oracle.models import AgentType, Session
from agent_oracle.sources.claude import parse_claude_session
from agent_oracle.sources.codex import parse_codex_session
from agent_oracle.sources.factory import parse_factory_session
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

#: The base directory used to locate agent session folders (patchable in tests).
_HOME = Path.home()

#: Maps each agent to the normalizer that parses its session files.
_PARSERS: dict[AgentType, Callable[[Path], Session]] = {
    AgentType.CODEX: parse_codex_session,
    AgentType.FACTORY: parse_factory_session,
    AgentType.CLAUDE: parse_claude_session,
}


def _watched_dirs() -> dict[AgentType, Path]:
    """Return the per-agent session directories below ``_HOME``."""
    return {
        AgentType.CODEX: _HOME / ".codex" / "sessions",
        AgentType.FACTORY: _HOME / ".factory" / "sessions",
        AgentType.CLAUDE: _HOME / ".claude" / "projects",
    }


class _Handler(FileSystemEventHandler):
    """Watchdog handler that forwards modify events to a SessionWatcher."""

    def __init__(self, watcher: SessionWatcher) -> None:
        """Remember the watcher that owns this handler."""
        self.watcher = watcher

    def on_modified(self, event: FileSystemEvent) -> None:
        """Forward a file modify event to the watcher's debounce path."""
        if event.is_directory:
            return
        self.watcher._on_modified(event)


class SessionWatcher:
    """Watches agent session directories and indexes new or changed files."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        enricher: Enricher,
        debounce_seconds: float = 2.0,
    ) -> None:
        """Store the dependencies and configure the debounce delay."""
        self.store = store
        self.embedder = embedder
        self.enricher = enricher
        self.debounce_seconds = debounce_seconds
        self._observer = Observer()
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Begin watching the session directories with a recursive observer."""
        for directory in _watched_dirs().values():
            directory.mkdir(parents=True, exist_ok=True)
            self._observer.schedule(_Handler(self), str(directory), recursive=True)
        self._observer.start()
        logger.info("Watching session directories for changes")

    def stop(self) -> None:
        """Stop the observer and wait for its threads to exit."""
        self._observer.stop()
        self._observer.join()
        logger.info("Stopped watching session directories")

    def index_existing(self) -> None:
        """Bulk-import all existing session files across the watched directories."""
        for directory in _watched_dirs().values():
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*.jsonl")):
                self._index_file(path)

    def _index_file(self, path: Path) -> None:
        """Parse, store, embed, and enrich a single session file."""
        if path.suffix != ".jsonl":
            return
        parser = self._detect_parser(path)
        if parser is None:
            logger.debug("Skipping file outside watched dirs: %s", path)
            return
        try:
            session = parser(path)
            self.store.index_session(session)
            self._embed_messages(session)
            self._enrich_session(session)
            logger.info("Indexed session %s (%s)", session.id, session.agent.value)
        except Exception:
            logger.error("Failed to index session file %s", path, exc_info=True)

    def _embed_messages(self, session: Session) -> None:
        """Batch-embed searchable messages in a session and store the vectors."""
        indexed = self.store.get_session(session.id)
        if indexed is None:
            logger.warning("No stored messages for session %s; skipping embeddings", session.id)
            return
        all_messages = indexed["messages"]
        searchable = [
            m
            for m in all_messages
            if not (m.get("is_thinking") or m.get("is_system_instruction") or m.get("is_injected"))
        ]
        if not searchable:
            return
        texts = [msg["content"] for msg in searchable]
        embeddings = self.embedder.embed_batch(texts)
        for msg, embedding in zip(searchable, embeddings, strict=True):
            self.store.upsert_embedding(msg["id"], embedding)

    def _enrich_session(self, session: Session) -> None:
        """Enrich the session with the LLM and persist summary and entities."""
        try:
            result: EnrichmentResult = self.enricher.enrich(session)
        except RuntimeError as exc:
            logger.warning("Skipping enrichment for %s: %s", session.id, exc)
            return
        self.store.set_summary(session.id, result.summary)
        if result.summary:
            embedding = self.embedder.embed_batch([result.summary])[0]
            self.store.upsert_session_embedding(session.id, embedding)
        entities = [{"type": entity.type, "value": entity.value} for entity in result.entities]
        self.store.upsert_entities(session.id, entities)

    def _detect_parser(self, path: Path) -> Callable[[Path], Session] | None:
        """Return the normalizer matching the file's location, or None."""
        text = str(path)
        for agent, directory in _watched_dirs().items():
            if str(directory) in text:
                return _PARSERS[agent]
        return None

    def _on_modified(self, event: FileSystemEvent) -> None:
        """Debounce and schedule re-indexing for the modified file path."""
        src = event.src_path
        if isinstance(src, bytes):
            src = src.decode("utf-8")
        self.debounce(Path(src))

    def debounce(self, path: Path) -> None:
        """Cancel any pending re-index for *path* and schedule a new one."""
        key = str(path)
        with self._lock:
            pending = self._timers.get(key)
            if pending is not None:
                pending.cancel()
            timer = threading.Timer(self.debounce_seconds, self._index_file, args=(path,))
            self._timers[key] = timer
        timer.start()
