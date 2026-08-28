"""REST route for provider-reported token statistics."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request

_TOKEN_FIELDS = (
    "responses",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def register_token_usage_route(app: FastAPI) -> None:
    """Register token consumption statistics grouped by agent and model."""

    @app.get("/api/stats/tokens")
    def tokens(
        request: Request,
        agent: Annotated[str | None, Query(pattern="^(codex|factory|claude|omp|pi)$")] = None,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, Any]:
        """Return provider-reported token usage for the selected archive scope."""
        if start is not None and end is not None and start > end:
            raise HTTPException(status_code=422, detail="start must not be after end")
        rows = request.app.state.store.list_token_usage(agent=agent, start=start, end=end)
        return {
            "agent_model": rows,
            "agents": _roll_up(rows, "agent"),
            "models": _roll_up(rows, "model"),
        }


def _roll_up(rows: list[dict[str, Any]], grouping: str) -> list[dict[str, Any]]:
    """Aggregate token rows by one dimension while preserving unknown metrics."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row[grouping]
        if key not in grouped:
            grouped[key] = dict.fromkeys(_TOKEN_FIELDS)
            grouped[key]["responses"] = 0
            grouped[key]["agent"] = row["agent"] if grouping == "agent" else None
            grouped[key]["model"] = row["model"] if grouping == "model" else None
        for field in _TOKEN_FIELDS:
            if row[field] is not None:
                grouped[key][field] = (grouped[key][field] or 0) + row[field]
    return list(grouped.values())
