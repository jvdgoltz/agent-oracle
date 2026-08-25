"""MCP server exposing Agent Oracle tools to coding agents.

Builds a FastMCP server whose tools let a coding agent search archived
sessions (text / vector / hybrid), fetch a full session, and list the most
recent sessions.  The server is decoupled from storage and embedding details
by taking a :class:`Store` and an :class:`Embedder` as dependencies.
"""

from __future__ import annotations

import logging

from fastmcp.server import FastMCP

from agent_oracle.embed import Embedder
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

_SESSION_FIELDS = ("agent", "cwd", "title", "started_at")


def create_mcp_server(store: Store, embedder: Embedder) -> FastMCP:
    """Create a FastMCP server exposing Agent Oracle search and retrieval tools."""

    mcp = FastMCP("agent-oracle")

    @mcp.tool
    def search_sessions(query: str, mode: str = "hybrid", limit: int = 20) -> list[dict]:
        """Search archived sessions, returning result metadata per session.

        The *mode* selects text, vector, or hybrid (default) search; vector and
        hybrid modes embed *query* using the configured embedder.
        """
        if mode == "hybrid":
            embedding = embedder.embed_query(query)
            results = store.search_hybrid(query, embedding, limit=limit)
        elif mode == "text":
            results = store.search_text(query, limit=limit)
        elif mode == "vector":
            embedding = embedder.embed_query(query)
            results = store.search_vector(embedding, limit=limit)
        else:
            raise ValueError(f"unknown search mode: {mode!r}")
        return [_enrich(result, store) for result in results]

    @mcp.tool
    def get_session(session_id: str) -> dict | None:
        """Return the full session detail for *session_id*, or None if unknown."""
        return store.get_session(session_id)

    @mcp.tool
    def list_recent_sessions(limit: int = 20, offset: int = 0) -> list[dict]:
        """Return recent sessions ordered by start time, with pagination."""
        return store.list_sessions(limit=limit, offset=offset, include_review_agents=False)

    return mcp


def _enrich(result: dict, store: Store) -> dict:
    """Merge search *result* with its session metadata into a tool response."""
    session = store.get_session(result["session_id"]) or {}
    snippet = result.get("snippet") or session.get("summary")
    score = result.get("score", result.get("rank", result.get("distance")))
    return {
        "session_id": result["session_id"],
        "snippet": snippet,
        "score": score,
        **{field: session.get(field) for field in _SESSION_FIELDS},
    }
