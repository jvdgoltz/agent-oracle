"""Tests for provider-aware token usage summaries."""

from agent_oracle.token_usage_summary import summarize_token_usage


def _row(agent: str, **tokens: int | None) -> dict:
    """Build one complete token usage aggregate row."""
    row = {
        "agent": agent,
        "model": "shared-model",
        "responses": 1,
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "reasoning_output_tokens": None,
        "total_tokens": None,
    }
    row.update(tokens)
    return row


def test_cache_hit_rate_respects_provider_field_semantics() -> None:
    """Codex subset fields and separate cache fields produce comparable rates."""
    report = summarize_token_usage(
        [
            _row("codex", input_tokens=100, cached_input_tokens=80),
            _row(
                "omp",
                input_tokens=20,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=70,
            ),
        ]
    )

    assert report["agent_model"][0]["cache_hit_rate"] == 80
    assert report["agent_model"][1]["cache_hit_rate"] == 70
    assert [row["cache_hit_rate"] for row in report["agents"]] == [80, 70]
    assert report["models"][0]["cache_hit_rate"] == 75


def test_cache_hit_rate_remains_unknown_without_eligible_input() -> None:
    """Missing cache telemetry does not become a zero-percent hit rate."""
    report = summarize_token_usage([_row("factory", input_tokens=100)])

    assert report["agent_model"][0]["cache_hit_rate"] is None
    assert report["agents"][0]["cache_hit_rate"] is None
