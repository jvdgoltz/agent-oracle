"""Tests for the LLM enrichment module."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAI

from agent_oracle.enrich import (
    Enricher,
    EnrichmentOutput,
    EnrichmentResult,
    Entity,
    EntityOutput,
    normalize_entity_value,
)
from agent_oracle.models import AgentType, Message, MessageRole, Session


def _make_session(contents: list[str]) -> Session:
    """Build a Session with one message per given content string."""
    ts = datetime.now(UTC)
    messages = [
        Message(
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=c,
            timestamp=ts,
        )
        for i, c in enumerate(contents)
    ]
    return Session(
        id="sess-1",
        agent=AgentType.CODEX,
        cwd="/tmp/p",
        started_at=ts,
        messages=messages,
    )


def test_build_prompt_contains_message_content() -> None:
    """The prompt embeds the session message content."""
    session = _make_session(["hello there", "fix the bug"])
    enricher = Enricher(api_key="test-key")
    prompt = enricher._build_prompt(session)
    assert "hello there" in prompt
    assert "fix the bug" in prompt


def test_enrich_with_mocked_client() -> None:
    """Enrichment supplies a Pydantic schema and maps the parsed result."""
    session = _make_session(["let's build a search engine"])
    parsed = EnrichmentOutput(
        summary="Session built a search engine.",
        entities=[EntityOutput(type="product", value="SQLite")],
    )
    enricher = Enricher(api_key="test-key")
    fake = FakeClient([parsed])
    from typing import cast

    enricher._client = cast("OpenAI", fake)

    result = enricher.enrich(session)

    assert isinstance(result, EnrichmentResult)
    assert result.summary == "Session built a search engine."
    assert result.entities == [Entity(type="product", value="sqlite")]
    assert fake.chat.completions.calls[0]["response_format"] is EnrichmentOutput


def test_enrich_returns_empty_result_without_parsed_output() -> None:
    """Enrichment degrades safely when the API supplies no parsed payload."""
    enricher = Enricher(api_key="test-key")
    fake = FakeClient([None])
    from typing import cast

    enricher._client = cast("OpenAI", fake)

    assert enricher.enrich(_make_session(["hello"])) == EnrichmentResult(
        summary="", entities=[]
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("SQLite", "sqlite"),
        ("Agent Oracle", "agent-oracle"),
        ("  OpenAI\tAPI\n", "openai-api"),
    ],
)
def test_normalize_entity_value(value: str, expected: str) -> None:
    """Entity values are lower-case and use hyphens for whitespace."""
    assert normalize_entity_value(value) == expected


def test_enrich_skips_entity_that_normalizes_to_empty() -> None:
    """Whitespace-only entity values are not persisted as empty strings."""
    enricher = Enricher(api_key="test-key")
    fake = FakeClient(
        [EnrichmentOutput(summary="Summary.", entities=[EntityOutput(type="product", value=" ")])]
    )
    from typing import cast

    enricher._client = cast("OpenAI", fake)

    assert enricher.enrich(_make_session(["hello"])).entities == []


def test_enrich_creates_openai_client_by_default() -> None:
    """With no api_key, the client falls back to the OPENAI_API_KEY env var."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENAI_API_KEY", "env-key")
        enricher = Enricher()
        assert isinstance(enricher.client, OpenAI)


class FakeClient:
    """Minimal stand-in mirroring the OpenAI client's chat.completions chain."""

    def __init__(self, responses: list[EnrichmentOutput | None]) -> None:
        self.chat = FakeChat(responses)


class FakeChat:
    """Fake chat namespace exposing a completions attribute."""

    def __init__(self, responses: list[EnrichmentOutput | None]) -> None:
        self.completions = FakeCompletions(responses)


class FakeCompletions:
    """Fake completions endpoint returning canned attribute-style responses."""

    def __init__(self, responses: list[EnrichmentOutput | None]) -> None:
        self._responses = responses
        self._index = 0
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        """Return the next canned completion as a nested SimpleNamespace."""
        self.calls.append(kwargs)
        response = self._responses[self._index]
        self._index += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=response))])


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts/lists to attribute-accessible namespaces."""
    if isinstance(value, dict):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value
