"""Validate and fingerprint portable embedding evaluation datasets."""

from __future__ import annotations

import hashlib
import json

SCHEMA_VERSION = 1


def dataset_fingerprint(dataset: dict) -> str:
    """Return a stable fingerprint of evaluation inputs, not construction state."""
    payload = {
        "schema_version": dataset["schema_version"],
        "questions": dataset["questions"],
        "corpus": dataset["corpus"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_dataset(dataset: dict) -> None:
    """Require a schema-versioned corpus containing every gold session."""
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported eval dataset schema version")
    if not dataset.get("fingerprint"):
        raise ValueError("Eval dataset has no fingerprint")
    questions = dataset.get("questions")
    corpus = dataset.get("corpus")
    if not isinstance(questions, list) or not isinstance(corpus, list):
        raise ValueError("Eval dataset requires questions and corpus lists")
    if dataset["fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("Eval dataset fingerprint does not match its contents")
    corpus_sessions = {passage.get("session_id") for passage in corpus}
    for question in questions:
        if question.get("session_id") not in corpus_sessions:
            raise ValueError(f"Question {question.get('id')} has no gold session passage in corpus")
