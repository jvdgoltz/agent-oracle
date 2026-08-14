"""Tests for clustered evaluation-session selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SKILL_PYTHON = Path(__file__).parents[2] / ".agents/skills/eval-embedding-reranking/python"
sys.path.insert(0, str(SKILL_PYTHON))


def load_generator():
    """Load the dataset generator without making the skill a package."""
    path = SKILL_PYTHON / "generate_eval_dataset.py"
    spec = importlib.util.spec_from_file_location("eval_generator_selection", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_sessions_takes_five_nearest_sessions_per_cluster() -> None:
    """Each cluster contributes its five closest stored-vector sessions."""
    generator = load_generator()

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
    """Small clusters do not make an archive sample smaller than its minimum."""
    generator = load_generator()

    class FakeStore:
        """Give enough remaining eligible sessions for deterministic backfill."""

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
