"""LLM enrichment of coding sessions.

Calls the OpenAI chat completions API to extract entities from a fixed list of
types (``product``, ``person``, ``organization``, ``place``) and to write a
short session summary. Used by the store to enrich archived sessions.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from openai import OpenAI

from agent_oracle.models import Session

logger = logging.getLogger(__name__)

#: Fixed vocabulary of entity types the model may emit.
ENTITY_TYPES = ("product", "person", "organization", "place")

#: Maximum number of characters of session content included in the prompt.
_MAX_PROMPT_CHARS = 50_000


@dataclass(frozen=True, slots=True)
class Entity:
    """A single extracted entity with a type and value."""

    type: str
    value: str


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """The output of enriching a session: a summary plus extracted entities."""

    summary: str
    entities: list[Entity]


class Enricher:
    """Extracts entities and a summary from a session via the OpenAI API."""

    def __init__(self, model: str = "gpt-5.6-luna", api_key: str | None = None) -> None:
        """Store config; the OpenAI client is created lazily on first use."""
        self.model = model
        self._api_key = api_key
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Return the OpenAI client, creating it on first access."""
        if self._client is None:
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set; enrichment is unavailable.")
            self._client = OpenAI(api_key=key)
        return self._client

    def enrich(self, session: Session) -> EnrichmentResult:
        """Build a prompt from the session, call the API, and parse the result."""
        prompt = self._build_prompt(session)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        if raw is None:
            logger.warning("Enricher received an empty completion for session %s", session.id)
            return EnrichmentResult(summary="", entities=[])
        return self._parse_response(raw)

    def _build_prompt(self, session: Session) -> str:
        """Concatenate message contents, truncated, into an extraction prompt."""
        transcript = "\n".join(message.content for message in session.messages)
        transcript = transcript[:_MAX_PROMPT_CHARS]
        types = ", ".join(ENTITY_TYPES)
        return (
            f"Analyze the following coding agent session transcript.\n"
            f"Extract entities of these types only: {types}.\n"
            "Write a 2-3 sentence summary of what the session accomplished.\n"
            'Respond with JSON: {"summary": "...", '
            '"entities": [{"type": "...", "value": "..."}]}.\n\n'
            f"TRANSCRIPT:\n{transcript}"
        )

    def _parse_response(self, raw: str) -> EnrichmentResult:
        """Parse the JSON response, validating types against the fixed list."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Enricher could not parse model response as JSON", exc_info=True)
            return EnrichmentResult(summary="", entities=[])

        summary = data.get("summary", "")
        entities = [
            Entity(type=item["type"], value=item["value"])
            for item in data.get("entities", [])
            if item.get("type") in ENTITY_TYPES and item.get("value")
        ]
        return EnrichmentResult(summary=summary, entities=entities)
