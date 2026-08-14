"""Tests for portable embedding evaluation datasets."""

from __future__ import annotations

import pytest

from agent_oracle.eval_dataset import dataset_fingerprint, validate_dataset


def _dataset() -> dict:
    """Return one valid portable dataset."""
    dataset = {
        "schema_version": 1,
        "questions": [{"id": "q-0", "question": "find setup", "session_id": "s1"}],
        "corpus": [{"id": "m1", "session_id": "s1", "content": "setup details"}],
    }
    dataset["fingerprint"] = dataset_fingerprint(dataset)
    return dataset


def test_validate_dataset_accepts_matching_portable_content() -> None:
    """A matching fingerprint and gold corpus session are valid."""
    validate_dataset(_dataset())


def test_validate_dataset_rejects_tampered_content() -> None:
    """Fingerprint validation detects changed evaluation passages."""
    dataset = _dataset()
    dataset["corpus"][0]["content"] = "changed"

    with pytest.raises(ValueError, match="fingerprint"):
        validate_dataset(dataset)
