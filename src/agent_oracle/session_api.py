"""Session-scoped API routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)


def register_enrich_route(app: FastAPI) -> None:
    """Register the session-scoped re-index and enrichment endpoint."""

    @app.post("/api/sessions/{session_id}/enrich")
    def enrich_session(request: Request, session_id: str) -> dict[str, str]:
        """Re-index and enrich one archived session from its source file."""
        store = request.app.state.store
        if store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        watcher = request.app.state.watcher
        if watcher is None:
            raise HTTPException(status_code=503, detail="Session watcher is unavailable")
        try:
            if not watcher.reindex_session(session_id, clear_existing=True):
                raise HTTPException(status_code=404, detail="Session source not found")
        except HTTPException:
            raise
        except (RuntimeError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Enrichment failed: {exc}") from exc
        return {"status": "enriched"}


def register_summary_route(app: FastAPI) -> None:
    """Register the search-result summary endpoint."""

    @app.post("/api/search/summary")
    def search_summary(request: Request, body: dict[str, Any]) -> dict[str, str]:
        """Summarize search results when a summarizer is configured."""
        summarizer = request.app.state.summarizer
        if summarizer is None or not body.get("results", []):
            return {"summary": ""}
        try:
            summary = summarizer.summarize(body.get("query", ""), body["results"])
        except Exception:
            logger.warning("Search summary generation failed", exc_info=True)
            summary = ""
        return {"summary": summary}
