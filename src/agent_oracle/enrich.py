"""LLM enrichment of coding sessions.

Calls the OpenAI chat completions API to extract entities from a fixed list of
types (``product``, ``person``, ``organization``, ``place``) and to write a
short session summary. Used by the store to enrich archived sessions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

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


class EntityOutput(BaseModel):
    """Define one entity in the LLM structured output."""

    type: Literal["product", "person", "organization", "place"]
    value: str


class EnrichmentOutput(BaseModel):
    """Define the structured output requested from the enrichment LLM."""

    summary: str
    entities: list[EntityOutput]


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
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=EnrichmentOutput,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.warning("Enricher received an empty completion for session %s", session.id)
            return EnrichmentResult(summary="", entities=[])
        entities = []
        for item in parsed.entities:
            value = normalize_entity_value(item.value)
            if value:
                entities.append(Entity(type=item.type, value=value))
        return EnrichmentResult(summary=parsed.summary, entities=entities)

    def _build_prompt(self, session: Session) -> str:
        """Concatenate message contents, truncated, into an extraction prompt."""
        transcript = "\n".join(message.content for message in session.messages)
        transcript = transcript[:_MAX_PROMPT_CHARS]
        types = ", ".join(ENTITY_TYPES)
        return (
            f"Analyze the following coding agent session transcript.\n"
            f"Extract entities of these types only: {types}.\n"
            "Write a 2-3 sentence summary of what the session accomplished.\n"
            "Return the summary and entities in the requested structure.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )

def normalize_entity_value(value: str) -> str:
    """Return *value* in lower-case with whitespace replaced by hyphens."""
    return "-".join(value.lower().split())
