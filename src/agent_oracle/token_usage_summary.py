"""Summarize provider token usage and cache effectiveness."""

from __future__ import annotations

from typing import Any

TOKEN_FIELDS = (
    "responses",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def summarize_token_usage(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return agent-model rows plus token-weighted agent and model aggregates."""
    return {
        "agent_model": [_with_cache_hit_rate(row) for row in rows],
        "agents": _roll_up(rows, "agent"),
        "models": _roll_up(rows, "model"),
    }


def _roll_up(rows: list[dict[str, Any]], grouping: str) -> list[dict[str, Any]]:
    """Aggregate token rows by one dimension while preserving unknown metrics."""
    grouped: dict[str, dict[str, Any]] = {}
    cache_totals: dict[str, tuple[int, int]] = {}
    for row in rows:
        key = row[grouping]
        if key not in grouped:
            grouped[key] = dict.fromkeys(TOKEN_FIELDS)
            grouped[key]["responses"] = 0
            grouped[key]["agent"] = row["agent"] if grouping == "agent" else None
            grouped[key]["model"] = row["model"] if grouping == "model" else None
        for field in TOKEN_FIELDS:
            if row[field] is not None:
                grouped[key][field] = (grouped[key][field] or 0) + row[field]
        fraction = _cache_fraction(row)
        if fraction is not None:
            hits, eligible = cache_totals.get(key, (0, 0))
            cache_totals[key] = (hits + fraction[0], eligible + fraction[1])
    for key, row in grouped.items():
        hits, eligible = cache_totals.get(key, (0, 0))
        row["cache_hit_rate"] = hits / eligible * 100 if eligible else None
    return list(grouped.values())


def _with_cache_hit_rate(row: dict[str, Any]) -> dict[str, Any]:
    """Return one usage row with its provider-aware cache hit rate."""
    result = dict(row)
    fraction = _cache_fraction(row)
    result["cache_hit_rate"] = fraction[0] / fraction[1] * 100 if fraction else None
    return result


def _cache_fraction(row: dict[str, Any]) -> tuple[int, int] | None:
    """Return cache-hit and eligible-input tokens using provider field semantics."""
    if row["cached_input_tokens"] is not None:
        eligible = row["input_tokens"] or 0
        return (row["cached_input_tokens"], eligible) if eligible else None
    if row["cache_read_input_tokens"] is not None:
        hits = row["cache_read_input_tokens"]
        eligible = hits + (row["input_tokens"] or 0) + (row["cache_creation_input_tokens"] or 0)
        return (hits, eligible) if eligible else None
    return None
