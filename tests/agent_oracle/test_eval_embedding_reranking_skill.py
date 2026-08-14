"""Regression tests for the portable embedding and reranking eval skill."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SKILL_PYTHON = Path(__file__).parents[2] / ".agents/skills/eval-embedding-reranking/python"
sys.path.insert(0, str(SKILL_PYTHON))


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


def test_corpus_only_sessions_remain_pending() -> None:
    """Copied corpus passages alone do not mark a session as generated."""
    generator = load_script("generate_eval_dataset")
    dataset = {
        "questions": [],
        "corpus": [{"session_id": "corpus-session"}],
    }

    assert generator.processed_session_ids(dataset) == set()


def test_legacy_question_sessions_are_processed() -> None:
    """Legacy datasets derive completed sessions from their question labels."""
    generator = load_script("generate_eval_dataset")
    dataset = {
        "questions": [{"session_id": "question-session"}],
        "corpus": [{"session_id": "corpus-session"}],
    }

    assert generator.processed_session_ids(dataset) == {"question-session"}


def test_explicit_processed_sessions_do_not_change_eval_fingerprint() -> None:
    """Construction state does not make model evaluation results incomparable."""
    generator = load_script("generate_eval_dataset")
    dataset = {
        "schema_version": 1,
        "questions": [],
        "corpus": [],
        "processed_session_ids": ["zero-question-session"],
    }

    assert generator.processed_session_ids(dataset) == {"zero-question-session"}
    assert generator.dataset_fingerprint(dataset) == generator.dataset_fingerprint(
        {key: value for key, value in dataset.items() if key != "processed_session_ids"}
    )


def test_generator_persists_state_and_ignores_poisoned_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dataset state survives zero-question sessions and wins over a stale checkpoint."""
    generator = load_script("generate_eval_dataset")
    out = tmp_path / "dataset.json"
    checkpoint = tmp_path / ".dataset.checkpoint.json"
    checkpoint.write_text('{"done": ["poisoned-session"]}')
    out.write_text(
        '{"schema_version": 1, "questions": [{"id": "q-0", '
        '"question": "old", "topic": "old", "session_id": "done-session"}], '
        '"corpus": [], "processed_session_ids": ["done-session"]}'
    )

    class FakeStore:
        """Provide a small deterministic archive for generator testing."""

        def __init__(self, _db: Path) -> None:
            self.sessions = ["done-session", "poisoned-session", "zero-question-session"]

        def list_session_summary_embeddings(self) -> list[dict]:
            return [
                {"id": session_id, "embedding": [float(index), 1.0]}
                for index, session_id in enumerate(self.sessions)
            ]

        def list_all_entities(self) -> list[dict]:
            return []

        def get_session(self, session_id: str) -> dict:
            return {
                "id": session_id,
                "messages": [
                    {
                        "id": f"message-{session_id}",
                        "content": session_id,
                        "is_thinking": False,
                        "is_system_instruction": False,
                        "is_injected": False,
                    }
                ],
            }

    generated_for: list[str] = []

    def fake_generate(_client: Mock, _model: str, session: dict, _context: str) -> list[dict]:
        generated_for.append(session["id"])
        if session["id"] == "zero-question-session":
            return []
        return [{"question": "find poisoned session", "topic": "test"}]

    monkeypatch.setattr(generator, "Store", FakeStore)
    monkeypatch.setattr(generator, "OpenAI", Mock)
    monkeypatch.setattr(generator, "generate_session_questions", fake_generate)
    monkeypatch.setattr(sys, "argv", ["generate_eval_dataset.py", "--out", str(out)])

    generator.main()

    written = json.loads(out.read_text())
    assert written["processed_session_ids"] == [
        "done-session",
        "poisoned-session",
        "zero-question-session",
    ]
    assert generated_for == ["poisoned-session", "zero-question-session"]
    assert len(written["questions"]) == 2

    generator.main()

    assert generated_for == ["poisoned-session", "zero-question-session"]
    assert len(json.loads(out.read_text())["questions"]) == 2


def test_checkpoint_does_not_override_dataset_construction_state(tmp_path: Path) -> None:
    """Resumption disregards checkpoint state that contradicts the dataset."""
    generator = load_script("generate_eval_dataset")
    checkpoint = tmp_path / ".dataset.checkpoint.json"
    checkpoint.write_text('{"done": ["poisoned-session"]}')
    dataset = {
        "questions": [],
        "corpus": [],
        "processed_session_ids": ["authoritative-session"],
    }

    assert generator.resume_session_ids(dataset, checkpoint) == {"authoritative-session"}


def test_select_sessions_is_bounded_and_covers_entities_when_possible() -> None:
    """Selection combines summary clusters and entity coverage without duplicates."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide stored summary vectors and entity rows without a database."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [
                {"id": "done", "embedding": [0.0, 0.0]},
                {"id": "cluster-a", "embedding": [0.1, 0.0]},
                {"id": "cluster-b", "embedding": [10.0, 10.0]},
                {"id": "entity-only", "embedding": [10.1, 10.0]},
            ]

        def list_all_entities(self) -> list[dict]:
            return [
                {"session_id": "done", "entity_type": "product", "entity_value": "done"},
                {"session_id": "cluster-a", "entity_type": "product", "entity_value": "alpha"},
                {"session_id": "entity-only", "entity_type": "product", "entity_value": "beta"},
            ]

    selection = generator.select_sessions(
        FakeStore(), done={"done"}, max_sessions=3, cluster_samples=2, sessions_per_entity=1, seed=7
    )

    assert len(selection.session_ids) <= 3
    assert len(selection.session_ids) == len(set(selection.session_ids))
    assert "done" not in selection.session_ids
    assert {"product:alpha", "product:beta"} <= set(selection.covered_entities)
    assert selection.uncovered_entities == []


def test_select_sessions_reports_entity_coverage_truncated_by_hard_limit() -> None:
    """Entity cardinality cannot make synthetic LLM calls exceed the selected cap."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide more entities than the requested global selection capacity."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [
                {"id": "s1", "embedding": [0.0, 0.0]},
                {"id": "s2", "embedding": [1.0, 0.0]},
                {"id": "s3", "embedding": [2.0, 0.0]},
            ]

        def list_all_entities(self) -> list[dict]:
            return [
                {"session_id": "s1", "entity_type": "product", "entity_value": "alpha"},
                {"session_id": "s2", "entity_type": "product", "entity_value": "beta"},
                {"session_id": "s3", "entity_type": "product", "entity_value": "gamma"},
            ]

    selection = generator.select_sessions(
        FakeStore(), done=set(), max_sessions=2, cluster_samples=1, sessions_per_entity=1, seed=1
    )

    assert len(selection.session_ids) == 2
    assert len(selection.uncovered_entities) == 1


def test_dry_run_does_not_create_openai_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preview prints the bounded selection before any LLM client is constructed."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide one candidate session for the dry-run preview."""

        def __init__(self, _db: Path) -> None:
            pass

        def list_session_summary_embeddings(self) -> list[dict]:
            return [{"id": "s1", "embedding": [0.0, 0.0]}]

        def list_all_entities(self) -> list[dict]:
            return []

    monkeypatch.setattr(generator, "Store", FakeStore)
    monkeypatch.setattr(generator, "OpenAI", lambda: pytest.fail("must not construct client"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_eval_dataset.py", "--out", str(tmp_path / "dataset.json"), "--dry-run"],
    )

    generator.main()
