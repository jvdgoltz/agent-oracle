"""Smoke tests for the agent_oracle package."""

import agent_oracle


def test_package_is_importable() -> None:
    """The package imports and exposes a docstring."""
    assert agent_oracle.__doc__ is not None
