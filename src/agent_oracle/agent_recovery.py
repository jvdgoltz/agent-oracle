"""Persist the live Codex turn identity across backend reloads."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class RecoveryRecord(TypedDict):
    """Identify one physical Codex turn eligible for reload recovery."""

    state: str
    thread_id: str
    turn_id: str


class AgentRecoveryLease:
    """Store one transient live-turn lease in an atomically replaced JSON file."""

    def __init__(self, path: Path) -> None:
        """Keep the recovery path and serialize access within this process."""
        self.path = path
        self._lock = threading.Lock()

    def read(self) -> RecoveryRecord | None:
        """Return a valid recovery record, removing malformed state."""
        with self._lock:
            return self._read_unlocked()

    def claim_running(self) -> RecoveryRecord | None:
        """Atomically claim a running lease for startup reconciliation."""
        with self._lock:
            record = self._read_unlocked()
            if record is None or record["state"] != "running":
                return None
            self._write_unlocked({**record, "state": "recovering"})
            return record

    def mark_running(self, thread_id: str, turn_id: str) -> None:
        """Persist a newly started physical turn as reload-recoverable."""
        with self._lock:
            self._write_unlocked({"state": "running", "thread_id": thread_id, "turn_id": turn_id})

    def clear(self, thread_id: str | None = None, turn_id: str | None = None) -> None:
        """Remove the lease when it matches the optional physical-turn identity."""
        with self._lock:
            record = self._read_unlocked()
            if record is None:
                return
            if thread_id is not None and record["thread_id"] != thread_id:
                return
            if turn_id is not None and record["turn_id"] != turn_id:
                return
            self.path.unlink(missing_ok=True)

    def _read_unlocked(self) -> RecoveryRecord | None:
        """Read and validate state while the caller holds the lease lock."""
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Discarding invalid agent recovery lease at %s", self.path)
            self.path.unlink(missing_ok=True)
            return None
        if not (
            isinstance(value, dict)
            and value.get("state") in {"running", "recovering"}
            and isinstance(value.get("thread_id"), str)
            and isinstance(value.get("turn_id"), str)
        ):
            logger.warning("Discarding invalid agent recovery lease at %s", self.path)
            self.path.unlink(missing_ok=True)
            return None
        return RecoveryRecord(
            state=value["state"],
            thread_id=value["thread_id"],
            turn_id=value["turn_id"],
        )

    def _write_unlocked(self, record: RecoveryRecord) -> None:
        """Replace the lease atomically while the caller holds the lease lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, separators=(",", ":")))
        temporary.replace(self.path)
