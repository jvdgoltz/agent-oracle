"""FastAPI REST API for Agent Oracle.

Exposes the archived coding agent sessions over HTTP: listing, retrieval, and
text / vector / hybrid search with optional agent and entity filters. The store
and embedder are injected through :func:`create_app` and stored on ``app.state``
so route handlers stay thin and easy to test.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from agent_oracle.behavior import summarize_messages
from agent_oracle.embed import Embedder
from agent_oracle.search_summary import SearchSummarizer
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]


def create_app(
    store: Store, embedder: Embedder, summarizer: SearchSummarizer | None = None
) -> FastAPI:
    """Build and configure the Agent Oracle FastAPI application.

    The *store*, *embedder*, and optional *summarizer* are attached to
    ``app.state`` and CORS is enabled for all origins to support local
    development against the backend.
    """
    app = FastAPI(title="Agent Oracle API", version="0.1.0")
    app.state.store = store
    app.state.embedder = embedder
    app.state.summarizer = summarizer

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    """Attach every REST endpoint to *app*."""

    @app.get("/api/health")
    def health() -> dict[str, str]:
        """Return a simple liveness check for the service."""
        return {"status": "ok"}

    @app.get("/api/sessions")
    def list_sessions(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Return the most recent sessions with pagination metadata."""
        store = request.app.state.store
        sessions = store.list_sessions(limit=limit, offset=offset)
        entities = store.list_entities([s["id"] for s in sessions])
        for session in sessions:
            session["entities"] = _normalize_entities(entities.get(session["id"], []))
        return {"sessions": sessions, "total": len(sessions)}

    @app.get("/api/sessions/{session_id}")
    def get_session(request: Request, session_id: str) -> dict[str, Any]:
        """Return a single session with its messages, or 404 if unknown."""
        session = request.app.state.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session

    @app.get("/api/search")
    def search(
        request: Request,
        q: str = Query(min_length=1),
        mode: str = Query(default="hybrid"),
        limit: int = Query(default=20, ge=1, le=200),
        agent: str | None = Query(default=None),
        entity: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Search sessions by text, vector, or hybrid mode with optional filters.

        Returns results immediately without waiting for AI summary generation.
        Use ``POST /api/search/summary`` to fetch the AI summary separately.
        """
        store = request.app.state.store
        embedder = request.app.state.embedder
        raw_results = _run_search(store, embedder, q, mode, limit)
        raw_results = _filter_results(store, raw_results, agent=agent, entity=entity)
        grouped = _group_by_session(raw_results)
        entities = store.list_entities([r["session_id"] for r in grouped])
        payloads = [
            _payload(r, _normalize_entities(entities.get(r["session_id"], []))) for r in grouped
        ]
        return {"results": payloads}

    _register_summary_route(app)
    _register_behavior_route(app)

    @app.get("/api/entities")
    def get_entities(request: Request, session_id: str) -> dict[str, Any]:
        """Return the enriched entities recorded for *session_id*."""
        session_entities = request.app.state.store.get_entities(session_id)
        return {"entities": session_entities}


def _register_summary_route(app: FastAPI) -> None:
    """Register the POST /api/search/summary endpoint."""

    @app.post("/api/search/summary")
    def search_summary(
        request: Request,
        body: dict[str, Any],
    ) -> dict[str, str]:
        """Generate an AI summary of search results for a given query.

        Accepts a JSON body with ``query`` and ``results`` (the search payloads)
        and returns ``{"summary": "..."}``. Returns an empty summary if no
        summarizer is configured or the agent run fails.
        """
        summarizer = request.app.state.summarizer
        if summarizer is None:
            return {"summary": ""}
        query = body.get("query", "")
        results = body.get("results", [])
        if not results:
            return {"summary": ""}
        try:
            summary = summarizer.summarize(query, results)
        except Exception:
            logger.warning("Search summary generation failed", exc_info=True)
            summary = ""
        return {"summary": summary}


def _register_behavior_route(app: FastAPI) -> None:
    """Register the OMP-compatible user behavior statistics endpoint."""

    @app.get("/api/stats/behavior")
    def behavior(
        request: Request,
        agent: Annotated[str | None, Query(pattern="^(codex|factory|claude|omp)$")] = None,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        """Return query-time OMP behavior metrics for real user messages."""
        if start is not None and end is not None and start > end:
            raise HTTPException(status_code=422, detail="start must not be after end")
        messages = request.app.state.store.list_behavior_messages(
            agent=agent,
            start=start,
            end=end,
        )
        return summarize_messages(messages)


def _run_search(
    store: Store, embedder: Embedder, query: str, mode: str, limit: int
) -> list[SearchResult]:
    """Run the store search for *mode*, embedding the query when required."""
    if mode == "text":
        return store.search_text(query, limit=limit)
    query_embedding = embedder.embed_query(query)
    if mode == "vector":
        return store.search_vector(query_embedding, limit=limit)
    if mode == "hybrid":
        return store.search_hybrid(query, query_embedding, limit=limit)
    logger.warning("Unknown search mode %r, falling back to text", mode)
    return store.search_text(query, limit=limit)


def _filter_results(
    store: Store,
    results: list[SearchResult],
    *,
    agent: str | None,
    entity: str | None,
) -> list[SearchResult]:
    """Drop results whose session does not match the agent or entity filters."""
    if not agent and not entity:
        return results
    filtered: list[SearchResult] = []
    for result in results:
        session_id = result["session_id"]
        if agent is not None and _session_agent(store, session_id) != agent:
            continue
        if entity is not None and not _has_entity(store, session_id, entity):
            continue
        filtered.append(result)
    return filtered


def _session_agent(store: Store, session_id: str) -> str | None:
    """Return the agent type for *session_id*, or None if the session is missing."""
    session = store.get_session(session_id)
    return session.get("agent") if session else None


def _has_entity(store: Store, session_id: str, entity: str) -> bool:
    """Return True if any entity value for *session_id* equals *entity*."""
    return any(entry.get("entity_value") == entity for entry in store.get_entities(session_id))


def _group_by_session(results: list[SearchResult]) -> list[SearchResult]:
    """Merge results that share a session_id, collecting all message snippets.

    The store returns one row per matching message, so the same session can
    appear multiple times.  This groups them, keeping the best score and
    collecting all non-empty snippets into ``message_snippets``.
    """
    best: dict[str, SearchResult] = {}
    snippets: dict[str, list[str]] = {}
    for result in results:
        sid = result["session_id"]
        snippet = result.get("snippet", "")
        if snippet:
            snippets.setdefault(sid, []).append(snippet)
        if sid not in best:
            best[sid] = result
        else:
            score = result.get("score", result.get("rank", result.get("distance")))
            prev = best[sid]
            prev_score = prev.get("score", prev.get("rank", prev.get("distance")))
            if (
                isinstance(score, (int, float))
                and isinstance(prev_score, (int, float))
                and score > prev_score
            ):
                best[sid] = result
    return [{**r, "message_snippets": snippets.get(r["session_id"], [])} for r in best.values()]


def _normalize_entities(entities: list[dict]) -> list[dict]:
    """Convert store entity rows to the frontend ``type``/``value`` shape."""
    return [{"type": e["entity_type"], "value": e["entity_value"]} for e in entities]


def _payload(result: SearchResult, entities: list[dict]) -> SearchResult:
    """Normalize a store search result into the API response shape."""
    return {
        "session_id": result.get("session_id"),
        "agent": result.get("agent"),
        "cwd": result.get("cwd"),
        "started_at": result.get("started_at"),
        "summary": result.get("summary"),
        "entities": entities,
        "snippet": result.get("snippet", ""),
        "message_snippets": result.get("message_snippets", []),
        "score": result.get("score", result.get("rank", result.get("distance"))),
    }
