"""Tests for query-time OMP behavior aggregations."""

from __future__ import annotations

from agent_oracle.behavior_summary import summarize_messages


def test_summary_counts_each_omp_signal_without_derived_friction() -> None:
    """Expose six OMP signal totals and rates without Agent Oracle additions."""
    report = summarize_messages(
        [
            {
                "content": "Wrong, you missed it",
                "timestamp": "2026-08-01T12:00:00+00:00",
                "agent": "codex",
                "cwd": "/work/a",
                "is_injected": 0,
            }
        ]
    )

    assert report["totals"]["user_messages"] == 1
    assert report["totals"]["negation"] == 1
    assert report["totals"]["blame"] == 1
    assert "total_frustration" not in report["totals"]


def test_summary_groups_metrics_by_inferred_model() -> None:
    """Aggregate each user message once under its inferred assistant model."""
    report = summarize_messages(
        [
            {
                "content": "Wrong file",
                "timestamp": "2026-08-01T12:00:00+00:00",
                "agent": "codex",
                "cwd": "/work/a",
                "model": "gpt-5.6",
            },
            {
                "content": "hello",
                "timestamp": "2026-08-01T12:01:00+00:00",
                "agent": "codex",
                "cwd": "/work/a",
                "model": "unknown",
            },
        ]
    )

    assert [
        (row["model"], row["user_messages"], row["negation"], row["negation_rate"])
        for row in report["models"]
    ] == [("gpt-5.6", 1, 1, 100.0), ("unknown", 1, 0, 0.0)]
