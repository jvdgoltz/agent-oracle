"""Text embedding via FastEmbed (ONNX runtime, no PyTorch).

Loads a BGE-small model once (lazily) and exposes a small interface for
embedding passages, batches, and search queries.
"""

from __future__ import annotations

import logging
from typing import Any

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Lazily-loaded wrapper around a FastEmbed ONNX model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        _model: Any = None,
    ) -> None:
        """Configure the embedder without loading the model yet.

        The *model_name* selects the HuggingFace model used on first use.
        Tests can inject a *_model* to avoid downloading weights.
        """
        self.model_name = model_name
        self._model: Any = _model

    @property
    def model(self) -> Any:
        """Return the backing model, loading it on first access."""
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
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
        vectors = self.model.embed(texts)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query using FastEmbed query_embed method."""
        vector = next(iter(self.model.query_embed([query])))
        return vector.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return len(self.embed("test"))
