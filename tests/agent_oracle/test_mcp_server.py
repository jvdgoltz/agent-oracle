"""Tests for the MCP server (:mod:`agent_oracle.mcp_server`).

A fake :class:`Store` and :class:`Embedder` are injected so the tools can be
exercised hermetically without a real SQLite database or embedding model.
"""

from __future__ import annotations

import asyncio
from typing import cast

from agent_oracle.embed import Embedder
from agent_oracle.mcp_server import create_mcp_server
from agent_oracle.store import Store

SESSION = {
    "id": "s1",
    "agent": "codex",
    "cwd": "/tmp/project",
    "title": "Fix flaky test",
    "started_at": "2025-01-01T00:00:00",
    "summary": "Debugged the flaky test.",
    "enriched": 0,
    "messages": [
        {"id": 1, "session_id": "s1", "role": "user", "content": "help", "timestamp": "t", "seq": 0}
    ],
}

SEARCH_RESULTS = [
    {
        "session_id": "s1",
        "snippet": "Debugged the flaky test.",
        "score": 0.5,
        "agent": "codex",
        "cwd": "/tmp/project",
        "title": "Fix flaky test",
        "started_at": "2025-01-01T00:00:00",
    }
]


class _FakeStore:
    """A minimal stand-in for :class:`agent_oracle.store.Store`."""

    def __init__(self) -> None:
        """Record recent-session arguments for the visibility assertion."""
        self.recent_args: tuple[int, int, bool] | None = None

    def search_text(self, query: str, limit: int = 20) -> list[dict]:
        """Return a fake text search result for *query*."""
        return [{"session_id": "s1", "snippet": "text match", "rank": -1.0}]

    def search_vector(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        """Return a fake vector search result for *query_embedding*."""
        return [{"session_id": "s1", "distance": 0.1}]

    def search_hybrid(
        self, query: str, query_embedding: list[float], limit: int = 20
    ) -> list[dict]:
        """Return a fake hybrid search result for *query*."""
        return [{"session_id": "s1", "score": 0.5}]

    def get_session(self, session_id: str) -> dict | None:
        """Return the fake session when it is known, else None."""
        return SESSION if session_id == "s1" else None

    def list_sessions(
        self, limit: int = 50, offset: int = 0, *, include_review_agents: bool = False
    ) -> list[dict]:
        """Return a fake recent-sessions listing."""
        self.recent_args = (limit, offset, include_review_agents)
        keys = ("id", "agent", "cwd", "title", "started_at", "summary", "enriched")
        return [{k: SESSION[k] for k in keys}]


class _FakeEmbedder:
    """A minimal stand-in for :class:`agent_oracle.embed.Embedder`."""

    def embed_query(self, query: str) -> list[float]:
        """Return a deterministic fake query embedding."""
        return [0.1, 0.2, 0.3]


def _run(coro):
    """Run an awaitable coroutine to completion for a synchronous test."""
    return asyncio.run(coro)


def _make_server():
    """Build a server wired to fake store/embedder stand-ins."""
    store = cast(Store, _FakeStore())
    embedder = cast(Embedder, _FakeEmbedder())
    return create_mcp_server(store, embedder)


def _call(server, name: str, arguments: dict) -> list:
    """Invoke *name* on *server* and return the parsed result payload."""
    result = _run(server.call_tool(name, arguments))
    return result.structured_content["result"]


def _tool_names(server) -> list[str]:
    """Return the names of the tools registered on *server*."""
    tools = _run(server.list_tools())
    return [tool.name for tool in tools]


# --------------------------------------------------------------------------- #
# create_mcp_server
# --------------------------------------------------------------------------- #


def test_create_mcp_server_registers_all_tools() -> None:
    """All three tools are registered on the returned server."""
    server = _make_server()

    assert set(_tool_names(server)) == {
        "search_sessions",
        "get_session",
        "list_recent_sessions",
        "token_usage_stats",
    }


# --------------------------------------------------------------------------- #
# search_sessions
# --------------------------------------------------------------------------- #


def test_search_sessions_hybrid_mode() -> None:
    """hybrid mode embeds the query and returns enriched search results."""
    server = _make_server()

    results = _call(server, "search_sessions", {"query": "flaky test", "mode": "hybrid"})

    assert results == SEARCH_RESULTS


def test_search_sessions_text_mode() -> None:
    """text mode returns enriched snippets without embedding."""
    server = _make_server()

    results = _call(server, "search_sessions", {"query": "flaky", "mode": "text"})

    assert results[0]["session_id"] == "s1"
    assert results[0]["agent"] == "codex"
    assert results[0]["title"] == "Fix flaky test"


def test_search_sessions_vector_mode() -> None:
    """vector mode embeds the query and returns enriched results."""
    server = _make_server()

    results = _call(server, "search_sessions", {"query": "flaky", "mode": "vector"})

    assert results[0]["session_id"] == "s1"


def test_search_sessions_defaults_to_hybrid() -> None:
    """Searching without a mode uses hybrid search."""
    server = _make_server()

    results = _call(server, "search_sessions", {"query": "flaky"})

    assert results[0]["score"] == 0.5


# --------------------------------------------------------------------------- #
# get_session
# --------------------------------------------------------------------------- #


def test_get_session_returns_full_session() -> None:
    """get_session returns the session detail including messages."""
    server = _make_server()

    result = _call(server, "get_session", {"session_id": "s1"})

    assert result == SESSION


def test_get_session_missing_returns_none() -> None:
    """get_session returns None for an unknown session id."""
    server = _make_server()

    result = _call(server, "get_session", {"session_id": "nope"})

    assert result is None


# --------------------------------------------------------------------------- #
# list_recent_sessions
# --------------------------------------------------------------------------- #


def test_list_recent_sessions_returns_recent_sessions() -> None:
    """list_recent_sessions returns the recent sessions listing."""
    server = _make_server()

    result = _call(server, "list_recent_sessions", {"limit": 10, "offset": 0})

    assert result[0]["id"] == "s1"
    assert result[0]["agent"] == "codex"


def test_list_recent_sessions_excludes_review_agents() -> None:
    """The MCP overview always requests the default review-hidden listing."""
    store = _FakeStore()
    server = create_mcp_server(cast(Store, store), cast(Embedder, _FakeEmbedder()))

    _call(server, "list_recent_sessions", {"limit": 10, "offset": 3})

    assert store.recent_args == (10, 3, False)
