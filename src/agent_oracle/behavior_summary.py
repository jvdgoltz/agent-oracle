"""Query-time aggregations for OMP-compatible user behavior metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from agent_oracle.behavior import UserMessageMetrics, compute_user_message_metrics


def _empty_summary() -> dict[str, int | float]:
    """Return a zeroed serializable behavior aggregate."""
    return dict.fromkeys(
        (
            "user_messages",
            "chars",
            "words",
            "yelling",
            "profanity",
            "anguish",
            "negation",
            "repetition",
            "blame",
        ),
        0,
    )


def _add_metrics(summary: dict[str, int | float], metrics: UserMessageMetrics) -> None:
    """Accumulate one user message's metrics into *summary*."""
    summary["user_messages"] += 1
    for key, value in asdict(metrics).items():
        summary[key] += value


def _with_rates(summary: dict[str, int | float]) -> dict[str, int | float]:
    """Add OMP's per-user-message percentage rates to an aggregate."""
    result = dict(summary)
    messages = int(summary["user_messages"])
    for key in ("yelling", "profanity", "anguish", "negation", "repetition", "blame"):
        result[f"{key}_rate"] = (float(summary[key]) / messages * 100) if messages else 0.0
    return result


def _stats_date(timestamp: str) -> str:
    """Return the UTC calendar date SQLite's ``date()`` uses for timestamps."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    return parsed.astimezone(UTC).date().isoformat()


def summarize_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate OMP behavior statistics by day, agent, project, and model."""
    totals = _empty_summary()
    daily: dict[str, dict[str, int | float]] = defaultdict(_empty_summary)
    agents: dict[str, dict[str, int | float]] = defaultdict(_empty_summary)
    projects: dict[str, dict[str, int | float]] = defaultdict(_empty_summary)
    models: dict[str, dict[str, int | float]] = defaultdict(_empty_summary)
    for message in messages:
        if message.get("is_injected"):
            continue
        metrics = compute_user_message_metrics(message["content"])
        _add_metrics(totals, metrics)
        date = _stats_date(str(message["timestamp"]))
        agent = str(message["agent"])
        cwd = str(message["cwd"])
        model = str(message.get("model") or "unknown")
        _add_metrics(daily[date], metrics)
        _add_metrics(agents[agent], metrics)
        _add_metrics(projects[cwd], metrics)
        _add_metrics(models[model], metrics)
    return {
        "totals": _with_rates(totals),
        "daily": [{"date": key, **_with_rates(value)} for key, value in sorted(daily.items())],
        "agents": [{"agent": key, **_with_rates(value)} for key, value in sorted(agents.items())],
        "projects": [{"cwd": key, **_with_rates(value)} for key, value in sorted(projects.items())],
        "models": [{"model": key, **_with_rates(value)} for key, value in sorted(models.items())],
    }
