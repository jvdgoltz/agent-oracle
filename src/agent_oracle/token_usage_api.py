"""REST route for provider-reported token statistics."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request

from agent_oracle.token_usage_summary import summarize_token_usage


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
        return summarize_token_usage(rows)
