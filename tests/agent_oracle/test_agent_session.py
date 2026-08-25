"""Tests for the live Codex agent session manager."""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock

import pytest
from openai_codex import ImageInput, TextInput

from agent_oracle.agent_session import (
    _REPO_ROOT,
    AgentSessionError,
    AgentSessionManager,
    AgentSessionState,
    _authenticate_api_key_if_needed,
    _event_from_notification,
)


class _Stream:
    """Return a finite stream of fake Codex notifications."""

    def __iter__(self):
        """Yield the configured events."""
        yield MagicMock(
            method="item/agentMessage/delta",
            payload=MagicMock(model_dump=lambda **_: {"delta": "Hi"}),
        )
        yield MagicMock(
            method="turn/completed",
            payload=MagicMock(model_dump=lambda **_: {"turn": {"status": "completed"}}),
        )


class _BlockingStream:
    """Block turn streaming until the test allows the worker to finish."""

    def __init__(self, release: threading.Event) -> None:
        """Keep the completion event for the iterator."""
        self.release = release

    def __iter__(self):
        """Wait until the caller permits one completion event."""
        self.release.wait()
        yield MagicMock(
            method="turn/completed",
            payload=MagicMock(model_dump=lambda **_: {"turn": {"status": "completed"}}),
        )


def test_start_streams_user_and_sdk_events() -> None:
    """Starting a run publishes the user message and normalized SDK events."""
    thread = MagicMock(id="thread-1")
    turn = MagicMock(id="turn-1")
    turn.stream.return_value = _Stream()
    thread.turn.return_value = turn
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    state = manager.start("Investigate this")
    events = list(manager.events(state.thread_id))

    assert state.thread_id == "thread-1"
    assert [event["type"] for event in events] == ["auth", "user", "assistant", "completed"]
    assert events[2]["data"] == {"delta": "Hi"}


def test_start_passes_image_data_url_to_codex() -> None:
    """An image data URL becomes a typed Codex image input alongside the prompt."""
    thread = MagicMock(id="thread-1")
    thread.turn.return_value = MagicMock()
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    manager.start("Inspect this", image_data_url="data:image/png;base64,AAAA")

    turn_input = thread.turn.call_args.args[0]
    assert isinstance(turn_input[0], TextInput)
    assert turn_input[0].text == "Inspect this"
    assert isinstance(turn_input[1], ImageInput)
    assert turn_input[1].url == "data:image/png;base64,AAAA"


def test_start_rejects_a_second_active_session() -> None:
    """Only one active session can run at once."""
    manager = AgentSessionManager(codex_factory=MagicMock())
    manager._active = MagicMock(running=True)

    with pytest.raises(AgentSessionError):
        manager.start("Investigate this")


def test_stop_interrupts_the_running_turn() -> None:
    """Stopping a session forwards the request to Codex."""
    manager = AgentSessionManager(codex_factory=MagicMock())
    turn = MagicMock()
    manager._active = MagicMock(thread_id="thread-1", running=True, turn=turn)

    manager.stop("thread-1")

    turn.interrupt.assert_called_once_with()


def test_start_assigns_the_turn_before_control_can_interrupt() -> None:
    """Stop can always reach the TurnHandle immediately after start returns."""
    turn = MagicMock(id="turn-1")
    release = threading.Event()
    turn.stream.return_value = _BlockingStream(release)
    thread = MagicMock(id="thread-1")
    thread.turn.return_value = turn
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    state = manager.start("Investigate")
    manager.stop(state.thread_id)

    assert state.turn is turn
    turn.interrupt.assert_called_once_with()
    release.set()
    list(manager.events(state.thread_id))


def test_controls_wait_for_turn_setup_then_interrupt_the_new_handle() -> None:
    """Stop, disconnect, and New Session cannot act on an unassigned turn."""
    setup_entered = threading.Event()
    allow_setup = threading.Event()
    release_stream = threading.Event()
    turn = MagicMock(id="turn-1")
    turn.stream.return_value = _BlockingStream(release_stream)
    thread = MagicMock(id="thread-1")

    def create_turn(*args, **kwargs):
        """Hold synchronous setup while concurrent controls attempt to run."""
        setup_entered.set()
        allow_setup.wait()
        return turn

    thread.turn.side_effect = create_turn
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)
    started: list[AgentSessionState] = []
    start_thread = threading.Thread(target=lambda: started.append(manager.start("Investigate")))
    start_thread.start()
    assert setup_entered.wait(timeout=1)

    def stop_without_propagating_retirement_races() -> None:
        """Allow a concurrent New Session to retire the state first."""
        try:
            manager.stop("thread-1")
        except AgentSessionError:
            return

    controls = [
        threading.Thread(target=stop_without_propagating_retirement_races),
        threading.Thread(target=lambda: manager.new_session("thread-1")),
    ]
    for control in controls:
        control.start()
    assert turn.interrupt.call_count == 0

    allow_setup.set()
    start_thread.join(timeout=1)
    release_stream.set()
    for control in controls:
        control.join(timeout=1)

    assert started[0].turn is turn
    assert turn.interrupt.call_count >= 1
    assert manager._active is None


def test_follow_up_reuses_the_idle_thread() -> None:
    """An idle session accepts the next message without creating a new thread."""
    first_turn = MagicMock(id="turn-1")
    second_turn = MagicMock(id="turn-2")
    first_turn.stream.return_value = _Stream()
    second_turn.stream.return_value = _Stream()
    thread = MagicMock(id="thread-1")
    thread.turn.side_effect = [first_turn, second_turn]
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    state = manager.start("First")
    list(manager.events(state.thread_id))
    manager.send_message(state.thread_id, "Second")
    list(manager.events(state.thread_id))

    codex.thread_start.assert_called_once()
    assert thread.turn.call_args_list[1].args == ("Second",)
    codex.close.assert_not_called()


def test_resume_uses_the_requested_thread_and_accepts_follow_up() -> None:
    """A resumed non-ephemeral Codex thread remains conversational."""
    first_turn = MagicMock(id="turn-1")
    second_turn = MagicMock(id="turn-2")
    first_turn.stream.return_value = _Stream()
    second_turn.stream.return_value = _Stream()
    thread = MagicMock(id="saved-thread")
    thread.turn.side_effect = [first_turn, second_turn]
    codex = MagicMock()
    codex.thread_resume.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    state = manager.start("Resume", resume_thread_id="saved-thread")
    list(manager.events(state.thread_id))
    manager.send_message(state.thread_id, "Continue")
    list(manager.events(state.thread_id))

    codex.thread_resume.assert_called_once()
    assert codex.thread_resume.call_args.args == ("saved-thread",)
    assert codex.thread_resume.call_args.kwargs == manager._thread_options()
    assert thread.turn.call_args_list[1].args == ("Continue",)


def test_resume_message_reopens_an_archived_thread_for_a_follow_up() -> None:
    """An archived thread is resumed only when its first follow-up is sent."""
    turn = MagicMock(id="turn-1")
    turn.stream.return_value = _Stream()
    thread = MagicMock(id="saved-thread")
    thread.turn.return_value = turn
    codex = MagicMock()
    codex.thread_resume.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    state = manager.resume_message("saved-thread", "Continue")
    list(manager.events(state.thread_id))

    codex.thread_resume.assert_called_once()
    assert thread.turn.call_args.args == ("Continue",)


def test_resume_message_reuses_the_matching_live_thread() -> None:
    """A matching idle thread is reused without a second SDK resume call."""
    turn = MagicMock(id="turn-1")
    turn.stream.return_value = _Stream()
    thread = MagicMock(id="saved-thread")
    thread.turn.return_value = turn
    codex = MagicMock()
    manager = AgentSessionManager(codex_factory=lambda: codex)
    state = AgentSessionState(thread_id="saved-thread", codex=codex, thread=thread, running=False)
    state.completed.set()
    manager._active = state

    manager.resume_message("saved-thread", "Continue")

    codex.thread_resume.assert_not_called()
    assert thread.turn.call_args.args == ("Continue",)


def test_start_configures_codex_for_the_repository() -> None:
    """New threads use the requested model, sandbox, MCP, and safety prompt."""
    thread = MagicMock(id="thread-1")
    turn = MagicMock(id="turn-1")
    turn.stream.return_value = _Stream()
    thread.turn.return_value = turn
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex, mcp_url="http://mcp.test")

    state = manager.start("Investigate")
    list(manager.events(state.thread_id))

    options = codex.thread_start.call_args.kwargs
    assert options["ephemeral"] is False
    assert options["cwd"] == _REPO_ROOT
    assert options["model"] == "gpt-5.6-luna"
    assert options["sandbox"].value == "workspace-write"
    assert options["config"]["mcp_servers"]["agent-oracle"]["url"] == "http://mcp.test"
    assert options["config"]["web_search"] == "live"
    assert "untrusted" in options["base_instructions"]
    turn_options = thread.turn.call_args.kwargs
    assert turn_options["effort"] == "medium"


def test_default_codex_configuration_uses_mounted_agent_oracle_mcp() -> None:
    """New threads default to the MCP route mounted by the FastAPI app."""
    thread = MagicMock(id="thread-1")
    turn = MagicMock(id="turn-1")
    turn.stream.return_value = _Stream()
    thread.turn.return_value = turn
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    state = manager.start("Investigate")
    list(manager.events(state.thread_id))

    options = codex.thread_start.call_args.kwargs
    assert options["config"]["mcp_servers"]["agent-oracle"]["url"] == ("http://127.0.0.1:8731/mcp/")


def test_new_session_closes_an_idle_client() -> None:
    """New Session releases an idle client before a later replacement."""
    codex = MagicMock()
    manager = AgentSessionManager(codex_factory=lambda: codex)
    state = AgentSessionState(thread_id="thread-1", running=False, codex=codex)
    state.completed.set()
    manager._active = state

    manager.new_session("thread-1")

    codex.close.assert_called_once_with()


def test_api_key_fallback_is_used_only_when_local_auth_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback reports its mode without placing the key in any event data."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    codex = MagicMock()
    codex.account.return_value = MagicMock(account=None, requires_openai_auth=True)

    assert _authenticate_api_key_if_needed(codex) == "api_key_fallback"
    codex.login_api_key.assert_called_once_with("secret")


def test_failed_auth_closes_the_new_codex_client() -> None:
    """A failed setup cannot leak a partially initialized Codex process."""
    codex = MagicMock()
    codex.account.side_effect = RuntimeError("not logged in")
    manager = AgentSessionManager(codex_factory=lambda: codex)

    with pytest.raises(RuntimeError, match="not logged in"):
        manager.start("Investigate")

    codex.close.assert_called_once_with()


def test_failed_turn_setup_closes_the_new_codex_client() -> None:
    """A failed first turn cannot leave a live client or active session behind."""
    thread = MagicMock(id="thread-1")
    thread.turn.side_effect = RuntimeError("turn setup failed")
    codex = MagicMock()
    codex.thread_start.return_value = thread
    manager = AgentSessionManager(codex_factory=lambda: codex)

    with pytest.raises(RuntimeError, match="turn setup failed"):
        manager.start("Investigate")

    codex.close.assert_called_once_with()


def test_old_worker_cannot_finish_a_new_turn_queue() -> None:
    """An old worker leaves the newer turn running and its queue untouched."""
    old_queue: queue.Queue = queue.Queue()
    new_queue: queue.Queue = queue.Queue()
    turn = MagicMock(id="old")
    turn.stream.return_value = _Stream()
    thread = MagicMock()
    thread.turn.return_value = turn
    state = AgentSessionState(
        thread_id="thread-1",
        thread=thread,
        turn=turn,
        running=True,
        generation=2,
        events=new_queue,
    )
    manager = AgentSessionManager(codex_factory=MagicMock())
    manager._active = state

    manager._run_turn(state, 1, old_queue)

    assert state.running is True
    assert new_queue.empty()
    assert old_queue.get_nowait()["type"] == "assistant"


def test_new_session_retires_an_active_turn_without_blocking_replacement() -> None:
    """New Session interrupts work and makes a fresh start available immediately."""
    old_turn = MagicMock()
    old = AgentSessionState(thread_id="old", codex=MagicMock(), turn=old_turn, running=True)
    old.completed.set()
    new_thread = MagicMock(id="new")
    new_turn = MagicMock()
    new_turn.stream.return_value = _Stream()
    new_thread.turn.return_value = new_turn
    codex = MagicMock()
    codex.thread_start.return_value = new_thread
    manager = AgentSessionManager(codex_factory=lambda: codex)
    manager._active = old

    manager.new_session("old")
    state = manager.start("Fresh")
    list(manager.events(state.thread_id))

    old_turn.interrupt.assert_called_once_with()
    assert state.thread_id == "new"


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("item/agentMessage/delta", "assistant"),
        ("item/completed", "assistant"),
        ("item/reasoning/summaryTextDelta", "reasoning"),
        ("item/reasoning/textDelta", "reasoning"),
        ("item/commandExecution/outputDelta", "command"),
        ("item/mcpToolCall/progress", "item"),
        ("item/fileChange/outputDelta", "file"),
        ("thread/tokenUsage/updated", "usage"),
        ("item/started", "item"),
        ("item/completed", "item"),
        ("turn/completed", "completed"),
        ("error", "error"),
        ("error", "retry"),
    ],
)
def test_notification_mapping_retains_method_and_payload(method: str, expected: str) -> None:
    """Every SDK event category is preserved for the browser details pane."""
    payload = {"ok": True}
    if method == "item/completed" and expected == "assistant":
        payload = {"item": {"type": "agentMessage", "text": "Final answer"}}
    if method == "error" and expected == "retry":
        payload = {"willRetry": True}
    notification = MagicMock(method=method, payload=MagicMock(model_dump=lambda **_: payload))

    event = _event_from_notification(notification)

    assert event == {"type": expected, "method": method, "data": payload}
