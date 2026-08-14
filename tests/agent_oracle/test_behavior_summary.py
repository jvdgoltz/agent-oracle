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
    assert report["totals"]["detected_messages"] == 1
    assert report["totals"]["detection_rate"] == 100.0


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


def test_summary_counts_multi_signal_messages_once_and_sorts_breakdowns() -> None:
    """Count detected prompts once and rank all breakdowns by detection rate."""
    report = summarize_messages(
        [
            {
                "content": "Wrong, you missed it",
                "timestamp": "2026-08-02T12:00:00+00:00",
                "agent": "zeta",
                "cwd": "/zeta",
                "model": "zeta-model",
            },
            {
                "content": "hello",
                "timestamp": "2026-08-01T12:00:00+00:00",
                "agent": "alpha",
                "cwd": "/alpha",
                "model": "alpha-model",
            },
            {
                "content": "Wrong file",
                "timestamp": "2026-08-03T12:00:00+00:00",
                "agent": "beta",
                "cwd": "/beta",
                "model": "beta-model",
            },
        ]
    )

    assert report["totals"]["detected_messages"] == 2
    assert report["totals"]["detection_rate"] == 2 / 3 * 100
    assert [row["date"] for row in report["daily"]] == ["2026-08-02", "2026-08-03", "2026-08-01"]
    assert [row["agent"] for row in report["agents"]] == ["beta", "zeta", "alpha"]
    assert [row["cwd"] for row in report["projects"]] == ["/beta", "/zeta", "/alpha"]
    assert [row["model"] for row in report["models"]] == ["beta-model", "zeta-model", "alpha-model"]


def test_summary_has_zero_detection_rate_without_messages() -> None:
    """Avoid division by zero when a selected archive scope has no messages."""
    report = summarize_messages([])

    assert report["totals"]["detected_messages"] == 0
    assert report["totals"]["detection_rate"] == 0.0


def test_summary_counts_an_interruption_as_one_detected_message() -> None:
    """Count an interruption once even when lexical signals also match."""
    report = summarize_messages(
        [
            {
                "content": "Wrong, you missed it",
                "timestamp": "2026-08-01T12:00:00+00:00",
                "agent": "codex",
                "cwd": "/work/a",
                "model": "gpt-5.6",
                "is_interrupted": True,
            }
        ]
    )

    assert report["totals"]["interruptions"] == 1
    assert report["totals"]["interruption_rate"] == 100.0
    assert report["totals"]["detected_messages"] == 1
    assert report["models"][0]["interruptions"] == 1
    assert report["models"][0]["interruption_rate"] == 100.0
