"""Tests for the LLM enrichment module."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAI

from agent_oracle.enrich import Enricher, EnrichmentResult, Entity
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


def test_parse_response_valid_json() -> None:
    """Valid JSON with known entity types parses into an EnrichmentResult."""
    raw = (
        '{"summary": "A session about fixing bugs.", "entities": '
        '[{"type": "product", "value": "SQLite"}, {"type": "person", "value": "Ada"}]}'
    )
    result = Enricher(api_key="test-key")._parse_response(raw)
    assert result.summary == "A session about fixing bugs."
    assert result.entities == [
        Entity(type="product", value="SQLite"),
        Entity(type="person", value="Ada"),
    ]


def test_parse_response_skips_unknown_entity_types() -> None:
    """Entities with types outside the fixed list are filtered out."""
    raw = (
        '{"summary": "Summary.", "entities": ['
        '{"type": "product", "value": "SQLite"}, '
        '{"type": "bogus", "value": "Nope"}, '
        '{"type": "place", "value": "Berlin"}]}'
    )
    result = Enricher(api_key="test-key")._parse_response(raw)
    assert result.entities == [
        Entity(type="product", value="SQLite"),
        Entity(type="place", value="Berlin"),
    ]


def test_enrich_with_mocked_client() -> None:
    """enrich calls the client and returns entities and a summary."""
    session = _make_session(["let's build a search engine"])
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"summary": "Session built a search engine.", '
                            '"entities": [{"type": "product", "value": "SQLite"}]}'
                        )
                    }
                }
            ]
        }
    ]
    enricher = Enricher(api_key="test-key")
    with pytest.MonkeyPatch.context() as mp:
        fake = FakeClient(responses)
        mp.setattr(enricher, "client", fake)

        result = enricher.enrich(session)

    assert isinstance(result, EnrichmentResult)
    assert result.summary == "Session built a search engine."
    assert result.entities == [Entity(type="product", value="SQLite")]


def test_enrich_creates_openai_client_by_default() -> None:
    """With no api_key, the client falls back to the OPENAI_API_KEY env var."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPENAI_API_KEY", "env-key")
        enricher = Enricher()
    assert isinstance(enricher.client, OpenAI)


class FakeClient:
    """Minimal stand-in mirroring the OpenAI client's chat.completions chain."""

    def __init__(self, responses: list[dict]) -> None:
        self.chat = FakeChat(responses)


class FakeChat:
    """Fake chat namespace exposing a completions attribute."""

    def __init__(self, responses: list[dict]) -> None:
        self.completions = FakeCompletions(responses)


class FakeCompletions:
    """Fake completions endpoint returning canned attribute-style responses."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = responses
        self._index = 0

    def create(self, **_kwargs: object) -> SimpleNamespace:
        """Return the next canned completion as a nested SimpleNamespace."""
        response = self._responses[self._index]
        self._index += 1
        return _to_namespace(response)


def _to_namespace(value: Any) -> Any:
    """Recursively convert dicts/lists to attribute-accessible namespaces."""
    if isinstance(value, dict):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value
