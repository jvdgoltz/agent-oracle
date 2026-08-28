"""Tests for persisted live-agent recovery state."""

from __future__ import annotations

from pathlib import Path

from agent_oracle.agent_recovery import AgentRecoveryLease


def test_recovery_lease_round_trips_and_clears(tmp_path: Path) -> None:
    """The active turn identity survives a process restart until cleared."""
    path = tmp_path / "agent-turn.json"
    lease = AgentRecoveryLease(path)

    lease.mark_running("thread-1", "turn-1")

    assert AgentRecoveryLease(path).read() == {
        "state": "running",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
    }
    lease.clear()
    assert lease.read() is None


def test_invalid_recovery_lease_is_not_resumed(tmp_path: Path) -> None:
    """Malformed transient state fails closed instead of starting agent work."""
    path = tmp_path / "agent-turn.json"
    path.write_text("not json")

    assert AgentRecoveryLease(path).claim_running() is None
    assert not path.exists()
