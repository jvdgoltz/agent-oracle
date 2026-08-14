"""Tests for the live Codex agent REST and SSE endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import cast
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agent_oracle.agent_session import AgentSessionManager, AgentSessionState
from agent_oracle.api import _sse_events, create_app
from agent_oracle.models import AgentType, Message, MessageRole, Session


class _SseManager:
    """Provide one deterministic event to a real TestClient SSE response."""

    def __init__(self) -> None:
        """Initialize the unread event count."""
        self.reads = 0

    def stream_closed(self, thread_id: str) -> bool:
        """Close the stream after its one event has been read."""
        return self.reads > 0

    def next_event(self, thread_id: str, timeout: float) -> dict[str, object] | None:
        """Return a stable assistant event."""
        self.reads += 1
        return {"type": "assistant", "data": {"delta": "Hello"}}

    def cancel_on_disconnect(self, thread_id: str) -> None:
        """Accept the disconnect callback for the test transport."""


def test_agent_start_returns_thread_id() -> None:
    """The agent endpoint immediately returns the started Codex thread ID."""
    manager = MagicMock()
    manager.start.return_value = AgentSessionState(thread_id="thread-1")
    app = create_app(MagicMock(), MagicMock(), agent_manager=manager)

    response = TestClient(app).post("/api/agent/sessions", json={"message": "Investigate"})

    assert response.status_code == 201
    assert response.json() == {"thread_id": "thread-1"}
    manager.start.assert_called_once_with("Investigate", resume_thread_id=None)


def test_agent_stop_returns_no_content() -> None:
    """The stop endpoint delegates to the manager."""
    manager = MagicMock()
    app = create_app(MagicMock(), MagicMock(), agent_manager=manager)

    response = TestClient(app).post("/api/agent/sessions/thread-1/stop")

    assert response.status_code == 204
    manager.stop.assert_called_once_with("thread-1")


def test_agent_follow_up_uses_message_endpoint() -> None:
    """An idle session receives its next prompt through the message endpoint."""
    manager = MagicMock()
    manager.send_message.return_value = AgentSessionState(thread_id="thread-1")
    app = create_app(MagicMock(), MagicMock(), agent_manager=manager)

    response = TestClient(app).post(
        "/api/agent/sessions/thread-1/messages", json={"message": "More"}
    )

    assert response.status_code == 202
    assert response.json() == {"thread_id": "thread-1"}
    manager.send_message.assert_called_once_with("thread-1", "More")


def test_agent_empty_message_is_unprocessable() -> None:
    """The API distinguishes invalid input from a session conflict."""
    manager = MagicMock()
    manager.start.side_effect = ValueError("message must not be empty")
    app = create_app(MagicMock(), MagicMock(), agent_manager=manager)

    response = TestClient(app).post("/api/agent/sessions", json={"message": ""})

    assert response.status_code == 422


def test_agent_resume_requires_a_matching_archived_codex_session() -> None:
    """Resume IDs are validated before the manager can invoke Codex."""
    store = MagicMock()
    store.get_session.return_value = {"agent": "claude", "cwd": "/tmp/other"}
    manager = MagicMock()
    app = create_app(store, MagicMock(), agent_manager=manager)

    response = TestClient(app).post(
        "/api/agent/sessions", json={"message": "Resume", "resume_thread_id": "wrong"}
    )

    assert response.status_code == 422
    manager.start.assert_not_called()


def test_agent_resume_rejects_unknown_session() -> None:
    """Unknown resume IDs return 404 rather than reaching Codex."""
    store = MagicMock()
    store.get_session.return_value = None
    manager = MagicMock()
    app = create_app(store, MagicMock(), agent_manager=manager)

    response = TestClient(app).post(
        "/api/agent/sessions", json={"message": "Resume", "resume_thread_id": "missing"}
    )

    assert response.status_code == 404
    manager.start.assert_not_called()


def test_archived_agent_session_loads_source_transcript_without_starting_codex(
    monkeypatch,
) -> None:
    """An eligible archived thread can be displayed without creating a live turn."""
    archived = {
        "id": "saved-thread",
        "agent": "codex",
        "cwd": "/Users/jvdgoltz/Projects/agent-oracle",
        "messages": [{"role": "user", "content": "Earlier work"}],
    }
    store = MagicMock()
    store.get_session.return_value = archived
    manager = MagicMock()
    source = Session(
        id="saved-thread",
        agent=AgentType.CODEX,
        cwd="/Users/jvdgoltz/Projects/agent-oracle",
        started_at=datetime(2026, 8, 14),
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content="Stored reply",
                timestamp=datetime(2026, 8, 14),
            )
        ],
    )
    monkeypatch.setattr("agent_oracle.api.load_codex_session", lambda _: source)
    app = create_app(store, MagicMock(), agent_manager=manager)

    response = TestClient(app).get("/api/agent/sessions/saved-thread")

    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "Stored reply"
    manager.start.assert_not_called()


def test_archived_agent_follow_up_resumes_after_a_server_reload() -> None:
    """A follow-up resumes an archived eligible thread when no live state exists."""
    store = MagicMock()
    store.get_session.return_value = {
        "id": "saved-thread",
        "agent": "codex",
        "cwd": "/Users/jvdgoltz/Projects/agent-oracle",
    }
    manager = MagicMock()
    manager.has_thread.return_value = False
    manager.resume_message.return_value = AgentSessionState(thread_id="saved-thread")
    app = create_app(store, MagicMock(), agent_manager=manager)

    response = TestClient(app).post(
        "/api/agent/sessions/saved-thread/messages", json={"message": "Continue"}
    )

    assert response.status_code == 202
    assert response.json() == {"thread_id": "saved-thread"}
    manager.resume_message.assert_called_once_with("saved-thread", "Continue")


def test_live_agent_follow_up_does_not_require_archive_lookup() -> None:
    """An active new thread can receive its next message before archive indexing."""
    store = MagicMock()
    manager = MagicMock()
    manager.has_thread.return_value = True
    manager.send_message.return_value = AgentSessionState(thread_id="thread-1")
    app = create_app(store, MagicMock(), agent_manager=manager)

    response = TestClient(app).post(
        "/api/agent/sessions/thread-1/messages", json={"message": "More"}
    )

    assert response.status_code == 202
    manager.send_message.assert_called_once_with("thread-1", "More")
    store.get_session.assert_not_called()


def test_resume_candidates_read_past_the_first_page() -> None:
    """All archived pages are inspected instead of silently truncating at 200."""
    store = MagicMock()
    first_page = [{"id": str(index), "agent": "claude", "cwd": "/tmp"} for index in range(200)]
    second_page = [{"id": "keep", "agent": "codex", "cwd": "/Users/jvdgoltz/Projects/agent-oracle"}]
    store.list_sessions.side_effect = [first_page, second_page]
    app = create_app(store, MagicMock(), agent_manager=MagicMock())

    response = TestClient(app).get("/api/agent/sessions")

    assert response.status_code == 200
    assert [session["id"] for session in response.json()["sessions"]] == ["keep"]
    assert store.list_sessions.call_args_list[1].kwargs == {"limit": 200, "offset": 200}


def test_sse_disconnect_cancels_agent() -> None:
    """A disconnected SSE client requests turn cancellation without blocking the loop."""
    request = MagicMock()

    async def disconnected() -> bool:
        """Report a disconnected request."""
        return True

    request.is_disconnected = disconnected
    manager = MagicMock()
    manager.stream_closed.return_value = False

    async def consume() -> list[str]:
        """Consume the SSE generator until it ends."""
        return [event async for event in _sse_events(request, manager, "thread-1")]

    assert asyncio.run(consume()) == []
    manager.cancel_on_disconnect.assert_called_with("thread-1")


def test_agent_events_return_a_real_sse_response() -> None:
    """The event endpoint emits browser-compatible event and data lines."""
    app = create_app(
        MagicMock(), MagicMock(), agent_manager=cast(AgentSessionManager, _SseManager())
    )

    response = TestClient(app).get("/api/agent/sessions/thread-1/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: assistant" in response.text
    assert '"delta": "Hello"' in response.text


def test_agent_resume_candidates_only_include_this_codex_repository() -> None:
    """Only archived Codex threads from the current repository are resumable."""
    store = MagicMock()
    store.list_sessions.return_value = [
        {"id": "keep", "agent": "codex", "cwd": "/Users/jvdgoltz/Projects/agent-oracle"},
        {"id": "other-agent", "agent": "claude", "cwd": "/Users/jvdgoltz/Projects/agent-oracle"},
        {"id": "other-repo", "agent": "codex", "cwd": "/tmp/other"},
    ]
    app = create_app(store, MagicMock(), agent_manager=MagicMock())

    response = TestClient(app).get("/api/agent/sessions")

    assert response.status_code == 200
    assert [session["id"] for session in response.json()["sessions"]] == ["keep"]
