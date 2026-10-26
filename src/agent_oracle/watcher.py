"""File watcher for Agent Oracle.

Watches the session directories of Codex, Factory Droid, Claude Code, Oh My Pi,
and Pi and indexes
indexes new or modified session files into the store.  A debounce collapses the
burst of modify events emitted during a write into a single re-index per file.
"""

from __future__ import annotations

import logging
import threading
import time
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
from agent_oracle.sources.omp import parse_omp_session
from agent_oracle.sources.pi import parse_pi_session
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

#: The base directory used to locate agent session folders (patchable in tests).
_HOME = Path.home()

#: Maps each agent to the normalizer that parses its session files.
_PARSERS: dict[AgentType, Callable[[Path], Session]] = {
    AgentType.CODEX: parse_codex_session,
    AgentType.FACTORY: parse_factory_session,
    AgentType.CLAUDE: parse_claude_session,
    AgentType.OMP: parse_omp_session,
    AgentType.PI: parse_pi_session,
}


def _watched_dirs() -> dict[AgentType, Path]:
    """Return the per-agent session directories below ``_HOME``."""
    return {
        AgentType.CODEX: _HOME / ".codex" / "sessions",
        AgentType.FACTORY: _HOME / ".factory" / "sessions",
        AgentType.CLAUDE: _HOME / ".claude" / "projects",
        AgentType.OMP: _HOME / ".omp" / "agent" / "sessions",
        AgentType.PI: _HOME / ".pi" / "agent" / "sessions",
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
        self._pending: dict[Path, float] = {}
        self._condition = threading.Condition()
        self._stopped = threading.Event()
        self._index_lock = threading.Lock()
        self._worker = threading.Thread(target=self._work, daemon=True, name="session-worker")

    def start(self) -> None:
        """Begin watching the session directories with a recursive observer."""
        if self._stopped.is_set():
            return
        self._worker.start()
        for directory in _watched_dirs().values():
            directory.mkdir(parents=True, exist_ok=True)
            self._observer.schedule(_Handler(self), str(directory), recursive=True)
        self._observer.start()
        logger.info("Watching session directories for changes")

    def stop(self) -> None:
        """Stop the observer and wait for its threads to exit."""
        with self._condition:
            self._stopped.set()
            self._pending.clear()
            self._condition.notify_all()
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
        if self._worker.is_alive():
            self._worker.join()
        logger.info("Stopped watching session directories")

    def index_existing(self) -> None:
        """Bulk-import all existing session files, skipping already-enriched ones."""
        for directory in _watched_dirs().values():
            if not directory.exists():
                continue
            for path in directory.rglob("*.jsonl"):
                if self._stopped.is_set():
                    return
                self._index_file(path, skip_if_enriched=True)

    def reindex_session(self, session_id: str, *, clear_existing: bool = False) -> bool:
        """Re-index the watched source file belonging to ``session_id``."""
        directories = [*_watched_dirs().values(), _HOME / ".codex" / "archived_sessions"]
        for directory in directories:
            if not directory.exists():
                continue
            for path in directory.rglob("*.jsonl"):
                parser = self._detect_parser(path)
                if parser is None:
                    continue
                try:
                    if parser(path).id == session_id:
                        if clear_existing:
                            self.store.clear_enrichment(session_id)
                        return self._index_file(path, raise_errors=True)
                except Exception as exc:
                    if session_id in path.name:
                        raise exc
                    logger.debug("Unable to inspect session file %s", path, exc_info=True)
        return False

    def _index_file(
        self, path: Path, skip_if_enriched: bool = False, *, raise_errors: bool = False
    ) -> bool:
        """Parse, store, embed, and enrich a single session file.

        When *skip_if_enriched* is True (used during bulk re-index on startup),
        files whose session is already fully enriched are skipped to avoid
        redundant re-processing after restarts.
        """
        if path.suffix != ".jsonl":
            return False
        parser = self._detect_parser(path)
        if parser is None:
            logger.debug("Skipping file outside watched dirs: %s", path)
            return False
        with self._index_lock:
            if self._stopped.is_set():
                return False
            return self._process_file(path, parser, skip_if_enriched, raise_errors=raise_errors)

    def _process_file(
        self,
        path: Path,
        parser: Callable[[Path], Session],
        skip_if_enriched: bool,
        *,
        raise_errors: bool,
    ) -> bool:
        """Run a complete indexing job while holding the indexing lock."""
        try:
            session = parser(path)
            if skip_if_enriched and self.store.is_session_indexed(session.id):
                logger.debug("Skipping already-enriched session %s", session.id)
                return False
            self.store.index_session(session)
            if session.is_review_agent:
                logger.info("Stored unindexed Codex review session %s", session.id)
                return False
            self._embed_messages(session)
            self._enrich_session(session, raise_errors=raise_errors)
            logger.info("Indexed session %s (%s)", session.id, session.agent.value)
            return True
        except Exception:
            if raise_errors:
                raise
            logger.error("Failed to index session file %s", path, exc_info=True)
            return False

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
            if m.get("role", "user") in ("user", "assistant")
            and not (m.get("is_thinking") or m.get("is_system_instruction") or m.get("is_injected"))
            and not m.get("has_embedding")
        ]
        if not searchable:
            return
        for start in range(0, len(searchable), 8):
            if self._stopped.is_set():
                return
            batch = searchable[start : start + 8]
            embeddings = self.embedder.embed_batch([msg["content"] for msg in batch])
            for msg, embedding in zip(batch, embeddings, strict=True):
                self.store.upsert_embedding(msg["id"], embedding)

    def _enrich_session(self, session: Session, *, raise_errors: bool = False) -> None:
        """Enrich the session with the LLM and persist summary and entities."""
        self.store.clear_enrichment(session.id)
        try:
            result: EnrichmentResult = self.enricher.enrich(session)
        except RuntimeError as exc:
            if raise_errors:
                raise
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
        if str(_HOME / ".codex" / "archived_sessions") in text:
            return parse_codex_session
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
        """Coalesce paths in a bounded queue, applying backpressure when full."""
        if path.suffix != ".jsonl":
            return
        with self._condition:
            while len(self._pending) >= 1024 and path not in self._pending:
                if self._stopped.is_set():
                    return
                self._condition.wait()
            if not self._stopped.is_set():
                self._pending[path] = time.monotonic() + self.debounce_seconds
                self._condition.notify_all()

    def _work(self) -> None:
        """Process one ready path at a time, keeping follow-up events coalesced."""
        while not self._stopped.is_set():
            with self._condition:
                if self._stopped.is_set():
                    return
                if not self._pending:
                    self._condition.wait()
                    continue
                path = min(self._pending, key=self._pending.__getitem__)
                delay = self._pending[path] - time.monotonic()
                if delay > 0:
                    self._condition.wait(delay)
                    continue
                del self._pending[path]
                self._condition.notify_all()
            self._index_file(path)
