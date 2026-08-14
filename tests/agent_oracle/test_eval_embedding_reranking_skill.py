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


def test_question_generation_uses_pydantic_response_schema() -> None:
    """Question generation supplies its Pydantic model to the LLM API."""
    generator = load_script("generate_eval_dataset")
    parsed = generator.GeneratedQuestionsOutput(
        questions=[generator.GeneratedQuestion(question="How was search built?", topic="Search")]
    )
    client = Mock()
    client.chat.completions.parse.return_value = Mock(
        choices=[Mock(message=Mock(parsed=parsed))]
    )

    result = generator.generate_session_questions(
        client, "test-model", {"id": "session-1"}, "The agent built hybrid retrieval."
    )

    assert result == [{"question": "How was search built?", "topic": "search"}]
    assert client.chat.completions.parse.call_args.kwargs["response_format"] is (
        generator.GeneratedQuestionsOutput
    )


def test_question_generation_returns_empty_without_parsed_output() -> None:
    """Question generation skips a completion with no parsed payload."""
    generator = load_script("generate_eval_dataset")
    client = Mock()
    client.chat.completions.parse.return_value = Mock(
        choices=[Mock(message=Mock(parsed=None))]
    )

    assert (
        generator.generate_session_questions(
            client, "test-model", {"id": "session-1"}, "Some context"
        )
        == []
    )


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


def test_select_sessions_takes_five_nearest_sessions_per_cluster() -> None:
    """Each cluster contributes its five closest stored-vector sessions."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide two well-separated, equally sized vector clusters."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [
                {"id": f"a-{index}", "embedding": [float(index), 0.0]} for index in range(5)
            ] + [
                {"id": f"b-{index}", "embedding": [100.0 + float(index), 0.0]} for index in range(5)
            ]

        def list_all_entities(self) -> list[dict]:
            return []

    selection = generator.select_sessions(
        FakeStore(),
        done=set(),
        clusters=2,
        sessions_per_cluster=5,
        min_sessions=10,
        max_sessions=200,
        sessions_per_entity=1,
        seed=7,
    )

    assert len(selection.cluster_session_ids) == 10
    assert selection.backfill_session_ids == []
    assert set(selection.session_ids) == {f"a-{index}" for index in range(5)} | {
        f"b-{index}" for index in range(5)
    }


def test_select_sessions_backfills_cluster_shortfall_to_minimum() -> None:
    """Small clusters do not make a 100-session archive sample smaller than its minimum."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Give one cluster only one member and enough remaining eligible sessions."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [{"id": f"s{index}", "embedding": [float(index), 0.0]} for index in range(12)]

        def list_all_entities(self) -> list[dict]:
            return []

    selection = generator.select_sessions(
        FakeStore(),
        done=set(),
        clusters=10,
        sessions_per_cluster=1,
        min_sessions=12,
        max_sessions=200,
        sessions_per_entity=1,
        seed=0,
    )

    assert len(selection.cluster_session_ids) == 10
    assert len(selection.backfill_session_ids) == 2
    assert len(selection.session_ids) == 12


def test_select_sessions_warns_when_archive_is_smaller_than_minimum() -> None:
    """All eligible sessions are selected when the minimum cannot be reached."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide fewer embedded sessions than the requested minimum."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [{"id": f"s{index}", "embedding": [float(index), 0.0]} for index in range(3)]

        def list_all_entities(self) -> list[dict]:
            return []

    selection = generator.select_sessions(
        FakeStore(),
        done=set(),
        clusters=20,
        sessions_per_cluster=5,
        min_sessions=100,
        max_sessions=200,
        sessions_per_entity=1,
        seed=0,
    )

    assert len(selection.session_ids) == 3
    assert selection.minimum_shortfall == 97


def test_select_sessions_deduplicates_entity_additions() -> None:
    """Entity candidates already selected from clusters do not consume the cap twice."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide one clustered session and one distinct entity session."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [
                {"id": "cluster", "embedding": [0.0, 0.0]},
                {"id": "entity", "embedding": [10.0, 0.0]},
                {"id": "other", "embedding": [-10.0, 0.0]},
            ]

        def list_all_entities(self) -> list[dict]:
            return [
                {"session_id": "cluster", "entity_type": "product", "entity_value": "alpha"},
                {"session_id": "entity", "entity_type": "product", "entity_value": "beta"},
            ]

    selection = generator.select_sessions(
        FakeStore(),
        done=set(),
        clusters=1,
        sessions_per_cluster=1,
        min_sessions=1,
        max_sessions=200,
        sessions_per_entity=1,
        seed=0,
    )

    assert len(selection.session_ids) == len(set(selection.session_ids))
    assert selection.entity_session_ids == ["entity"]
    assert selection.covered_entities == ["product:beta"]
    assert selection.already_covered_entities == ["product:alpha"]


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
        FakeStore(),
        done=set(),
        clusters=1,
        sessions_per_cluster=1,
        min_sessions=1,
        max_sessions=2,
        sessions_per_entity=1,
        seed=1,
    )

    assert len(selection.session_ids) == 2
    assert len(selection.uncovered_entities) == 1


def test_select_sessions_never_exceeds_hard_maximum() -> None:
    """Cluster and entity selection together cannot exceed the 200-session limit."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Provide enough vectors and distinct entity values to pressure the cap."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [{"id": f"s{index}", "embedding": [float(index), 0.0]} for index in range(250)]

        def list_all_entities(self) -> list[dict]:
            return [
                {"session_id": f"s{index}", "entity_type": "topic", "entity_value": str(index)}
                for index in range(250)
            ]

    selection = generator.select_sessions(
        FakeStore(),
        done=set(),
        clusters=20,
        sessions_per_cluster=5,
        min_sessions=100,
        max_sessions=200,
        sessions_per_entity=1,
        seed=0,
    )

    assert len(selection.session_ids) == 200


def test_select_sessions_prioritizes_entity_values_with_more_sessions() -> None:
    """Entity selection uses eligible-session count before its stable tie-breaker."""
    generator = load_script("generate_eval_dataset")

    class FakeStore:
        """Leave space for only one entity addition after the cluster sample."""

        def list_session_summary_embeddings(self) -> list[dict]:
            return [
                {"id": "cluster", "embedding": [0.0, 0.0]},
                {"id": "large-a", "embedding": [10.0, 0.0]},
                {"id": "large-b", "embedding": [-10.0, 0.0]},
                {"id": "small", "embedding": [20.0, 0.0]},
                {"id": "other", "embedding": [-20.0, 0.0]},
            ]

        def list_all_entities(self) -> list[dict]:
            return [
                {"session_id": "large-a", "entity_type": "topic", "entity_value": "large"},
                {"session_id": "large-b", "entity_type": "topic", "entity_value": "large"},
                {"session_id": "small", "entity_type": "topic", "entity_value": "small"},
            ]

    selection = generator.select_sessions(
        FakeStore(),
        done=set(),
        clusters=1,
        sessions_per_cluster=1,
        min_sessions=1,
        max_sessions=2,
        sessions_per_entity=1,
        seed=0,
    )

    assert selection.covered_entities == ["topic:large"]
    assert selection.uncovered_entities == ["topic:small"]


def test_dry_run_does_not_create_openai_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
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

    output = capsys.readouterr().out
    assert "cluster-derived" in output
    assert "backfill" in output
    assert "entity-added" in output
    assert "Maximum LLM calls" in output
