"""Regression tests for the portable embedding and reranking eval skill."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

SKILL_PYTHON = Path(__file__).parents[2] / ".agents/skills/eval-embedding-reranking/python"


def load_script(name: str):
    """Load one skill script as a module without making it a package."""
    spec = importlib.util.spec_from_file_location(name, SKILL_PYTHON / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_fingerprint_changes_with_corpus_or_questions() -> None:
    """A portable dataset fingerprint covers both its corpus and gold labels."""
    generator = load_script("generate_eval_dataset")
    dataset = {
        "schema_version": 1,
        "questions": [{"id": "q-0", "question": "find setup", "session_id": "s1"}],
        "corpus": [{"id": "m1", "session_id": "s1", "content": "setup details"}],
    }

    fingerprint = generator.dataset_fingerprint(dataset)
    changed = {**dataset, "corpus": [{**dataset["corpus"][0], "content": "other"}]}

    assert fingerprint != generator.dataset_fingerprint(changed)


def test_dataset_validation_rejects_missing_gold_passages() -> None:
    """Evaluators require every gold session to be carried in the dataset corpus."""
    generator = load_script("generate_eval_dataset")
    evaluator = load_script("eval_embeddings")
    dataset = {
        "schema_version": 1,
        "questions": [{"id": "q-0", "question": "find setup", "session_id": "missing"}],
        "corpus": [{"id": "m1", "session_id": "s1", "content": "setup details"}],
    }
    dataset["fingerprint"] = generator.dataset_fingerprint(dataset)

    with pytest.raises(ValueError, match="gold session"):
        evaluator.validate_dataset(dataset)


@pytest.mark.parametrize("script_name", ["eval_embeddings", "eval_reranker"])
def test_evaluators_reject_tampered_dataset_fingerprints(script_name: str) -> None:
    """Evaluators reject a portable dataset whose declared contents changed."""
    generator = load_script("generate_eval_dataset")
    evaluator = load_script(script_name)
    dataset = {
        "schema_version": 1,
        "questions": [{"id": "q-0", "question": "find setup", "session_id": "s1"}],
        "corpus": [{"id": "m1", "session_id": "s1", "content": "setup details"}],
    }
    dataset["fingerprint"] = generator.dataset_fingerprint(dataset)
    dataset["corpus"][0]["content"] = "tampered"

    with pytest.raises(ValueError, match="fingerprint"):
        evaluator.validate_dataset(dataset)


def test_query_text_preparation_is_outside_embedding_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query construction completes before the model-only timer starts."""
    evaluator = load_script("eval_embeddings")
    prepared = False

    class Question(dict):
        def __getitem__(self, key: str):
            nonlocal prepared
            prepared = True
            return super().__getitem__(key)

    model = Mock()
    model.query_embed.return_value = [[0.1]]
    monkeypatch.setattr(
        evaluator.time,
        "perf_counter",
        lambda: 0.0 if prepared else pytest.fail("query text was not prepared before timing"),
    )

    evaluator.embed_queries(model, [Question(question="find setup")])


def test_reset_discards_existing_questions_and_checkpoint(tmp_path: Path) -> None:
    """Reset makes a fresh dataset rather than appending duplicate questions."""
    generator = load_script("generate_eval_dataset")
    out = tmp_path / "dataset.json"
    checkpoint = tmp_path / ".checkpoint.json"
    out.write_text('{"questions": [{"id": "q-0"}]}')
    checkpoint.write_text('{"done": ["s1"]}')

    generator.reset_generation_state(out, checkpoint)

    assert not out.exists()
    assert not checkpoint.exists()


def test_dataset_is_authoritative_when_checkpoint_is_missing() -> None:
    """Existing portable content prevents duplicate generation without a checkpoint."""
    generator = load_script("generate_eval_dataset")
    dataset = {
        "questions": [{"session_id": "question-session"}],
        "corpus": [{"session_id": "corpus-session"}],
    }

    assert generator.processed_session_ids(dataset) == {
        "question-session",
        "corpus-session",
    }
