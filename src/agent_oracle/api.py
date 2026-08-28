"""FastAPI REST API for Agent Oracle.

Exposes the archived coding agent sessions over HTTP: listing, retrieval, and
text / vector / hybrid search with optional agent and entity filters. The store
and embedder are injected through :func:`create_app` and stored on ``app.state``
so route handlers stay thin and easy to test.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent_oracle.agent_payload import source_messages as _source_messages
from agent_oracle.agent_payload import validated_image_data_url as _validated_image_data_url
from agent_oracle.agent_recovery import AgentRecoveryLease
from agent_oracle.agent_session import AgentSessionError, AgentSessionManager, encode_sse
from agent_oracle.behavior import summarize_messages
from agent_oracle.embed import Embedder
from agent_oracle.overview_summary import summarize_overview
from agent_oracle.search_summary import SearchSummarizer
from agent_oracle.session_api import register_enrich_route, register_summary_route
from agent_oracle.sources.codex import (
    archived_codex_session_ids,
    is_codex_session_archived,
    load_codex_session,
)
from agent_oracle.store import Store
from agent_oracle.token_usage_api import register_token_usage_route

if TYPE_CHECKING:
    from agent_oracle.watcher import SessionWatcher

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]

#: Browser origins served by the local Vite frontend.
LOCAL_FRONTEND_ORIGINS = ["http://localhost:8732", "http://127.0.0.1:8732"]


def create_app(
    store: Store,
    embedder: Embedder,
    summarizer: SearchSummarizer | None = None,
    agent_manager: AgentSessionManager | None = None,
    watcher: SessionWatcher | None = None,
) -> FastAPI:
    """Build and configure the Agent Oracle FastAPI application.

    The *store*, *embedder*, and optional *summarizer* are attached to
    ``app.state``. CORS accepts requests from the local frontend only.
    """
    app = FastAPI(title="Agent Oracle API", version="0.1.0")
    app.state.store = store
    app.state.embedder = embedder
    app.state.summarizer = summarizer
    app.state.agent_manager = agent_manager or AgentSessionManager(
        recovery_lease=AgentRecoveryLease(Path.home() / ".agent-oracle" / "agent-turn.json")
    )
    app.state.watcher = watcher

    app.add_middleware(
        CORSMiddleware,
        allow_origins=LOCAL_FRONTEND_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
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
        include_review_agents: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Return the most recent sessions with pagination metadata."""
        store = request.app.state.store
        sessions = store.list_sessions(
            limit=limit,
            offset=offset,
            include_review_agents=include_review_agents,
        )
        entities = store.list_entities([s["id"] for s in sessions])
        reviews = store.list_review_sessions([s["id"] for s in sessions])
        for session in sessions:
            session["entities"] = _normalize_entities(entities.get(session["id"], []))
            session["review_sessions"] = reviews.get(session["id"], [])
        return {"sessions": sessions, "total": len(sessions)}

    @app.get("/api/sessions/{session_id}")
    def get_session(request: Request, session_id: str) -> dict[str, Any]:
        """Return a single session with its messages, or 404 if unknown."""
        store = request.app.state.store
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session["entities"] = _normalize_entities(store.get_entities(session_id))
        session["review_sessions"] = store.list_review_sessions([session_id]).get(session_id, [])
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

    register_summary_route(app)
    register_enrich_route(app)
    _register_behavior_route(app)
    _register_overview_route(app)
    register_token_usage_route(app)
    _register_agent_routes(app)

    @app.get("/api/entities")
    def get_entities(request: Request, session_id: str) -> dict[str, Any]:
        """Return the enriched entities recorded for *session_id*."""
        session_entities = request.app.state.store.get_entities(session_id)
        return {"entities": session_entities}


def _register_behavior_route(app: FastAPI) -> None:
    """Register the OMP-compatible user behavior statistics endpoint."""

    @app.get("/api/stats/behavior")
    def behavior(
        request: Request,
        agent: Annotated[str | None, Query(pattern="^(codex|factory|claude|omp|pi)$")] = None,
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


def _register_overview_route(app: FastAPI) -> None:
    """Register the archive overview statistics endpoint."""

    @app.get("/api/stats/overview")
    def overview(
        request: Request,
        agent: Annotated[str | None, Query(pattern="^(codex|factory|claude|omp|pi)$")] = None,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        """Return query-time archive counts and session-length statistics."""
        if start is not None and end is not None and start > end:
            raise HTTPException(status_code=422, detail="start must not be after end")
        rows = request.app.state.store.list_overview_rows(
            agent=agent,
            start=start,
            end=end,
        )
        return summarize_overview(**rows)


def _register_agent_routes(app: FastAPI) -> None:
    """Register the start, event stream, and stop endpoints for Codex."""
    _register_agent_start_route(app)
    _register_agent_stream_route(app)
    _register_agent_control_routes(app)
    _register_archived_agent_route(app)


def _register_agent_start_route(app: FastAPI) -> None:
    """Register the routes that create or continue agent conversations."""

    @app.post("/api/agent/sessions", status_code=201)
    def start_agent(request: Request, body: dict[str, Any]) -> dict[str, str]:
        """Start a new or resumed agent thread and return its ID immediately."""
        resume_thread_id = body.get("resume_thread_id")
        if resume_thread_id is not None:
            _validate_resume_session(request.app.state.store, resume_thread_id)
        try:
            image_data_url = _validated_image_data_url(body)
            if image_data_url is None:
                state = request.app.state.agent_manager.start(
                    body.get("message", ""), resume_thread_id=resume_thread_id
                )
            else:
                state = request.app.state.agent_manager.start(
                    body.get("message", ""),
                    resume_thread_id=resume_thread_id,
                    image_data_url=image_data_url,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AgentSessionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"thread_id": state.thread_id}

    @app.get("/api/agent/sessions")
    def resumable_agent_sessions(request: Request) -> dict[str, list[dict[str, Any]]]:
        """List archived Codex sessions from this repository that can be resumed."""
        return {"sessions": _resumable_sessions(request.app.state.store)}


def _register_agent_stream_route(app: FastAPI) -> None:
    """Register the SSE route for one active agent turn."""

    @app.get("/api/agent/sessions/{thread_id}/events")
    async def agent_events(request: Request, thread_id: str) -> StreamingResponse:
        """Stream the active agent events as Server-Sent Events."""
        manager = request.app.state.agent_manager
        try:
            manager.stream_closed(thread_id)
        except AgentSessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StreamingResponse(
            _sse_events(request, manager, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _register_agent_control_routes(app: FastAPI) -> None:
    """Register message, stop, and New Session controls for Codex."""

    @app.post("/api/agent/sessions/{thread_id}/messages", status_code=202)
    def send_agent_message(
        request: Request, thread_id: str, body: dict[str, Any]
    ) -> dict[str, str]:
        """Send a follow-up message on an idle, existing Codex thread."""
        try:
            manager = request.app.state.agent_manager
            image_data_url = _validated_image_data_url(body)
            state = _send_agent_message(
                manager,
                request.app.state.store,
                thread_id,
                body.get("message", ""),
                image_data_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AgentSessionError as exc:
            status = 404 if "unknown" in str(exc) else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return {"thread_id": state.thread_id}

    @app.post("/api/agent/sessions/{thread_id}/stop", status_code=204)
    def stop_agent(request: Request, thread_id: str) -> None:
        """Request that Codex interrupts the currently active turn."""
        try:
            request.app.state.agent_manager.stop(thread_id)
        except AgentSessionError as exc:
            status = 404 if "unknown" in str(exc) else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.delete("/api/agent/sessions/{thread_id}", status_code=204)
    def new_agent_session(request: Request, thread_id: str) -> None:
        """Discard an idle session so the next start creates a fresh Codex thread."""
        try:
            request.app.state.agent_manager.new_session(thread_id)
        except AgentSessionError as exc:
            status = 404 if "unknown" in str(exc) else 409
            raise HTTPException(status_code=status, detail=str(exc)) from exc


def _register_archived_agent_route(app: FastAPI) -> None:
    """Register the read-only transcript route for archived Codex threads."""

    @app.get("/api/agent/sessions/{thread_id}")
    def get_archived_agent_session(request: Request, thread_id: str) -> dict[str, Any]:
        """Return an eligible archived Codex transcript without resuming Codex."""
        if request.app.state.agent_manager.is_running(thread_id) is True:
            raise HTTPException(status_code=409, detail="agent session is still running")
        _validate_agent_session(request.app.state.store, thread_id)
        session = request.app.state.store.get_session(thread_id)
        assert session is not None
        source_session = load_codex_session(thread_id)
        if source_session is not None:
            session["messages"] = _source_messages(source_session.messages, thread_id)
        return session


async def _sse_events(
    request: Request,
    manager: AgentSessionManager,
    thread_id: str,
) -> AsyncIterator[str]:
    """Encode agent events while leaving turns alive across client reconnects."""
    while not manager.stream_closed(thread_id):
        if await request.is_disconnected():
            return
        event = await asyncio.to_thread(manager.next_event, thread_id, 0.1)
        if event is not None:
            yield encode_sse(event)


def _send_agent_message(
    manager: AgentSessionManager,
    store: Store,
    thread_id: str,
    message: str,
    image_data_url: str | None,
) -> Any:
    """Send a text or image turn to the live thread, resuming it when needed."""
    if manager.has_thread(thread_id):
        if image_data_url is None:
            return manager.send_message(thread_id, message)
        return manager.send_message(thread_id, message, image_data_url=image_data_url)
    _validate_resume_session(store, thread_id)
    if image_data_url is None:
        return manager.resume_message(thread_id, message)
    return manager.resume_message(thread_id, message, image_data_url=image_data_url)


def _agent_repo_root() -> str:
    """Return the repository root used by the embedded Codex session."""
    from agent_oracle.agent_session import _REPO_ROOT

    return _REPO_ROOT


def _validate_resume_session(store: Store, thread_id: str) -> None:
    """Reject thread IDs that Codex cannot resume in this repository."""
    _validate_agent_session(store, thread_id)
    if is_codex_session_archived(thread_id):
        raise HTTPException(status_code=422, detail="Session cannot be resumed in this repository")


def _validate_agent_session(store: Store, thread_id: str) -> None:
    """Reject thread IDs that are not readable Codex sessions for this repository."""
    session = store.get_session(thread_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Resumable Codex session not found")
    if (
        session.get("agent") != "codex"
        or session.get("cwd") != _agent_repo_root()
        or session.get("is_review_agent", False)
    ):
        raise HTTPException(status_code=422, detail="Session cannot be resumed in this repository")


def _resumable_sessions(store: Store) -> list[dict[str, Any]]:
    """Read every paginated archive row and keep eligible Codex sessions."""
    candidates: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list_sessions(limit=200, offset=offset)
        candidates.extend(
            session
            for session in page
            if session.get("agent") == "codex" and session.get("cwd") == _agent_repo_root()
        )
        if len(page) < 200:
            archived_ids = archived_codex_session_ids({session["id"] for session in candidates})
            return [session for session in candidates if session["id"] not in archived_ids]
        offset += len(page)


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
        "title": result.get("title"),
        "started_at": result.get("started_at"),
        "summary": result.get("summary"),
        "entities": entities,
        "snippet": result.get("snippet", ""),
        "message_snippets": result.get("message_snippets", []),
        "score": result.get("score", result.get("rank", result.get("distance"))),
    }
