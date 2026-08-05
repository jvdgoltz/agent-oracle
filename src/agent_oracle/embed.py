"""Text embedding via FastEmbed (ONNX runtime, no PyTorch).

Loads a BGE-small model once (lazily) and exposes a small interface for
embedding passages, batches, and search queries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Protocol, cast

import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class _ModelProtocol(Protocol):
    """Minimal protocol for the FastEmbed model interface."""

    def embed(self, texts: list[str], **_kwargs: object) -> Iterator[np.ndarray]: ...

    def query_embed(self, queries: list[str], **_kwargs: object) -> Iterator[np.ndarray]: ...


class Embedder:
    """Lazily-loaded wrapper around a FastEmbed ONNX model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        _model: _ModelProtocol | None = None,
    ) -> None:
        """Configure the embedder without loading the model yet.

        The *model_name* selects the HuggingFace model used on first use.
        Tests can inject a *_model* to avoid downloading weights.
        """
        self.model_name = model_name
        self._model: _ModelProtocol | None = _model

    @property
    def model(self) -> _ModelProtocol:
        """Return the backing model, loading it on first access."""
        if self._model is None:
            self._model = cast("_ModelProtocol", TextEmbedding(model_name=self.model_name))
            logger.info("Loaded FastEmbed model %s", self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single string and return its dense vector as floats."""
        vector = next(iter(self.model.embed([text])))
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in one pass and return their vectors."""
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query using FastEmbed's query_embed method."""
        vector = next(iter(self.model.query_embed([query])))
        return vector.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return len(self.embed("test"))
