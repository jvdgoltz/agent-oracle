"""MCP server exposing Agent Oracle tools to coding agents.

Builds a FastMCP server whose tools let a coding agent search archived
sessions (text / vector / hybrid), fetch a full session, and list the most
recent sessions.  The server is decoupled from storage and embedding details
by taking a :class:`Store` and an :class:`Embedder` as dependencies.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from fastmcp.server import FastMCP

from agent_oracle.embed import Embedder
from agent_oracle.store import Store

logger = logging.getLogger(__name__)

_SESSION_FIELDS = ("agent", "cwd", "title", "started_at")


def create_mcp_server(store: Store, embedder: Embedder) -> FastMCP:  # noqa: C901
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

    @mcp.tool
    def token_usage_stats(
        agent: str | None = None, start: str | None = None, end: str | None = None
    ) -> dict:
        """Return token usage grouped by agent, model, and agent-model pair."""
        if agent is not None and not re.fullmatch("codex|factory|claude|omp|pi", agent):
            raise ValueError("invalid agent")
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start must not be after end")
        rows = store.list_token_usage(agent=agent, start=start_date, end=end_date)
        grouped: dict[str, dict] = {}
        for row in rows:
            for key, label in ((row["agent"], "agent"), (row["model"], "model")):
                item = grouped.setdefault(f"{label}:{key}", dict.fromkeys(_TOKEN_FIELDS))
                item[label] = key
                item["responses"] = (item["responses"] or 0) + row["responses"]
                for field in _TOKEN_FIELDS[1:]:
                    if row[field] is not None:
                        item[field] = (item[field] or 0) + row[field]
        return {
            "agent_model": rows,
            "agents": [v for k, v in grouped.items() if k.startswith("agent:")],
            "models": [v for k, v in grouped.items() if k.startswith("model:")],
        }

    return mcp


_TOKEN_FIELDS = (
    "responses",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


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
