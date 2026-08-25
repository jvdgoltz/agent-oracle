"""Tests for session-scoped API routes."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from agent_oracle.api import create_app


def test_enrich_session_reindexes_only_requested_session() -> None:
    """The enrich endpoint delegates the requested session to the watcher."""
    store = MagicMock()
    store.get_session.return_value = {"id": "s1"}
    watcher = MagicMock()
    watcher.reindex_session.return_value = True
    client = TestClient(create_app(store, MagicMock(), watcher=watcher))
    response = client.post("/api/sessions/s1/enrich")
    assert response.status_code == 200
    assert response.json() == {"status": "enriched"}
    watcher.reindex_session.assert_called_once_with("s1", clear_existing=True)


def test_enrich_session_reports_unavailable_watcher() -> None:
    """The enrich endpoint reports when background indexing is unavailable."""
    store = MagicMock()
    store.get_session.return_value = {"id": "s1"}
    response = TestClient(create_app(store, MagicMock())).post("/api/sessions/s1/enrich")
    assert response.status_code == 503


def test_enrich_session_reports_missing_source() -> None:
    """The enrich endpoint reports a missing source file distinctly."""
    store = MagicMock()
    store.get_session.return_value = {"id": "s1"}
    watcher = MagicMock()
    watcher.reindex_session.return_value = False
    response = TestClient(create_app(store, MagicMock(), watcher=watcher)).post(
        "/api/sessions/s1/enrich"
    )
    assert response.status_code == 404
    store.clear_enrichment.assert_not_called()


def test_enrich_session_reports_processing_failure() -> None:
    """The enrich endpoint surfaces source or enrichment failures."""
    store = MagicMock()
    store.get_session.return_value = {"id": "s1"}
    watcher = MagicMock()
    watcher.reindex_session.side_effect = RuntimeError("OPENAI_API_KEY is not set")
    response = TestClient(create_app(store, MagicMock(), watcher=watcher)).post(
        "/api/sessions/s1/enrich"
    )
    assert response.status_code == 422
    assert "OPENAI_API_KEY" in response.json()["detail"]


def _summary_client(
    summarizer: MagicMock | None = None,
) -> tuple[TestClient, MagicMock]:
    """Build a client and summarizer for summary-route tests."""
    summarizer = summarizer or MagicMock()
    return TestClient(create_app(MagicMock(), MagicMock(), summarizer=summarizer)), summarizer


def test_search_summary_returns_ai_summary() -> None:
    """The summary endpoint calls the summarizer and returns its text."""
    client, summarizer = _summary_client()
    summarizer.summarize.return_value = "These sessions cover SQLite usage."
    response = client.post(
        "/api/search/summary", json={"query": "sqlite", "results": [{"snippet": "a"}]}
    )
    assert response.json() == {"summary": "These sessions cover SQLite usage."}


def test_search_summary_without_summarizer_returns_empty() -> None:
    """When no summarizer is configured, the summary endpoint returns empty."""
    response = TestClient(create_app(MagicMock(), MagicMock())).post(
        "/api/search/summary", json={"query": "sqlite", "results": [{"snippet": "a"}]}
    )
    assert response.json() == {"summary": ""}


def test_search_summary_with_no_results_returns_empty() -> None:
    """When no results are provided, the summary endpoint returns empty."""
    client, summarizer = _summary_client()
    response = client.post("/api/search/summary", json={"query": "x", "results": []})
    assert response.json() == {"summary": ""}
    summarizer.summarize.assert_not_called()


def test_search_summary_fails_open_on_summarizer_error() -> None:
    """A summarizer failure logs and returns an empty summary with 200."""
    summarizer = MagicMock()
    summarizer.summarize.side_effect = RuntimeError("codex down")
    client, _ = _summary_client(summarizer)
    response = client.post(
        "/api/search/summary", json={"query": "sqlite", "results": [{"snippet": "a"}]}
    )
    assert response.json() == {"summary": ""}
