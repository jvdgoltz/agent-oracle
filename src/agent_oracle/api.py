"""FastAPI REST API for Agent Oracle.

Exposes the archived coding agent sessions over HTTP: listing, retrieval, and
text / vector / hybrid search with optional agent and entity filters. The store
and embedder are injected through :func:`create_app` and stored on ``app.state``
so route handlers stay thin and easy to test.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from agent_oracle.embed import Embedder
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]


def create_app(store: Store, embedder: Embedder) -> FastAPI:
    """Build and configure the Agent Oracle FastAPI application.

    The *store* and *embedder* are attached to ``app.state`` and CORS is enabled
    for all origins to support local development against the backend.
    """
    app = FastAPI(title="Agent Oracle API", version="0.1.0")
    app.state.store = store
    app.state.embedder = embedder

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
        sessions = request.app.state.store.list_sessions(limit=limit, offset=offset)
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
        """Search sessions by text, vector, or hybrid mode with optional filters."""
        store = request.app.state.store
        embedder = request.app.state.embedder
        results = _run_search(store, embedder, q, mode, limit)
        results = _filter_results(store, results, agent=agent, entity=entity)
        return {"results": [_payload(result) for result in results]}

    @app.get("/api/entities")
    def get_entities(request: Request, session_id: str) -> dict[str, Any]:
        """Return the enriched entities recorded for *session_id*."""
        session_entities = request.app.state.store.get_entities(session_id)
        return {"entities": session_entities}


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


def _payload(result: SearchResult) -> SearchResult:
    """Normalize a store search result into the API response shape."""
    return {
        "session_id": result.get("session_id"),
        "snippet": result.get("snippet", ""),
        "score": result.get("score", result.get("rank", result.get("distance"))),
    }
