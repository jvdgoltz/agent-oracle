"""FastEmbed wrapper for Agent Oracle.

Loads a dense embedding model once (lazily, on first use) and exposes a small
interface for embedding passages, batches, and search queries as lists of
floats.  Wrapping FastEmbed keeps the rest of the backend decoupled from the
underlying vector library.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol, cast

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class _Model(Protocol):
    """The slice of the FastEmbed model interface that Embedder relies on."""

    def embed(self, texts: list[str], **_kwargs: object) -> Iterable[np.ndarray]: ...
    def query_embed(self, query: str, **_kwargs: object) -> Iterable[np.ndarray]: ...

    @property
    def embedding_size(self) -> int: ...


class Embedder:
    """Lazily-loaded wrapper around a FastEmbed text embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        _model: _Model | None = None,
    ) -> None:
        """Configure the embedder without downloading the model yet.

        The *model_name* selects the FastEmbed model used on first use, unless
        a ready "_model" instance is injected (used by tests to avoid
        downloading the ~130MB model).
        """
        self.model_name = model_name
        self._model: _Model | None = _model

    @property
    def model(self) -> _Model:
        """Return the backing FastEmbed model, loading it on first access."""
        if self._model is None:
            self._model = cast("_Model", TextEmbedding(model_name=self.model_name))
            logger.info("Loaded FastEmbed model %s", self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return its dense vector as floats."""
        vector = next(iter(self.model.embed([text])))
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one pass and return their vectors."""
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query, semantically distinct from passages."""
        vector = next(iter(self.model.query_embed(query)))
        return vector.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.embedding_size
