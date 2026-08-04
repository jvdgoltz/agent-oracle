"""Main entry point for Agent Oracle.

Wires together the store, embedder, enricher, watcher, and FastAPI app.
Run with: ``uv run uvicorn agent_oracle.main:app --reload``
Or directly: ``uv run python -m agent_oracle.main``
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from agent_oracle.api import create_app
from agent_oracle.embed import Embedder
from agent_oracle.enrich import Enricher
from agent_oracle.store import Store
from agent_oracle.watcher import SessionWatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

#: Where Agent Oracle stores its database and copied sessions.
DATA_DIR = Path.home() / ".agent-oracle"


def _create_components() -> tuple[Store, Embedder, Enricher, SessionWatcher]:
    """Instantiate the store, embedder, enricher, and watcher."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(DATA_DIR / "index.db")
    embedder = Embedder()
    enricher = Enricher()
    watcher = SessionWatcher(store=store, embedder=embedder, enricher=enricher)
    return store, embedder, enricher, watcher


def _start_background_indexing(watcher: SessionWatcher) -> None:
    """Run the initial bulk index in a daemon thread, then start live watching."""

    def _run() -> None:
        """Index existing sessions, then start the file watcher."""
        logger.info("Starting bulk index of existing sessions...")
        watcher.index_existing()
        logger.info("Bulk index complete. Starting file watcher...")
        watcher.start()

    thread = threading.Thread(target=_run, daemon=True, name="agent-oracle-indexer")
    thread.start()


#: Lazy-initialised global app instance for uvicorn import.
app: FastAPI = create_app(*_create_components()[:2])

#: The watcher is started once on import so live changes are tracked.
_watcher_instance: SessionWatcher | None = None


def _ensure_watcher_running() -> None:
    """Start the background indexing and watcher (once per process)."""
    global _watcher_instance
    if _watcher_instance is not None:
        return
    _watcher_instance = _create_components()[3]
    _start_background_indexing(_watcher_instance)


_ensure_watcher_running()


if __name__ == "__main__":
    """Run the server directly when invoked as a module."""
    uvicorn.run(
        "agent_oracle.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
