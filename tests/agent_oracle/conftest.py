"""Shared pytest fixtures for the agent_oracle test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_oracle.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """Create a :class:`Store` backed by a temporary database file."""
    return Store(tmp_path / "test.db")
