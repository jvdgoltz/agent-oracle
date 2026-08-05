"""Tests for the FastEmbed embedding wrapper (:mod:`agent_oracle.embed`).

The real model is not downloaded during tests; a stub model stand-in is injected
to keep tests fast and hermetic.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from agent_oracle.embed import DEFAULT_MODEL, Embedder, _ModelProtocol

_DIM = 384


class _FakeModel(_ModelProtocol):
    """A stand-in for ``TextEmbedding`` returning deterministic vectors."""

    def __init__(self, model_name: str | None = None, **_kwargs: object) -> None:
        """Record the requested model name and the call count."""
        self.model_name = model_name
        self.calls = 0

    def embed(self, texts: list[str], **_kwargs: object) -> Iterator[np.ndarray]:
        """Return fake embeddings as a numpy array iterator."""
        self.calls += 1
        yield from np.zeros((len(texts), _DIM), dtype=np.float32)

    def query_embed(self, queries: list[str], **_kwargs: object) -> Iterator[np.ndarray]:
        """Return fake query embeddings as a numpy array iterator."""
        self.calls += 1
        yield from np.zeros((len(queries), _DIM), dtype=np.float32)


def test_embed_returns_float_list_of_expected_dimension() -> None:
    """A single embedding is a list of floats with the model dimension."""
    embedder = Embedder(_model=_FakeModel())
    vector = embedder.embed("hello world")

    assert isinstance(vector, list)
    assert len(vector) == _DIM
    assert all(isinstance(v, float) for v in vector)


def test_embed_accepts_empty_or_short_text() -> None:
    """Embedding handles a one-word string without error."""
    embedder = Embedder(_model=_FakeModel())
    assert len(embedder.embed("hi")) == _DIM


def test_embed_batch_returns_one_embedding_per_text() -> None:
    """Batch embedding yields as many vectors as input texts."""
    embedder = Embedder(_model=_FakeModel())
    vectors = embedder.embed_batch(["one", "two", "three"])

    assert len(vectors) == 3
    assert all(len(v) == _DIM for v in vectors)


def test_embed_batch_empty_list() -> None:
    """Batch embedding of no texts returns no vectors."""
    embedder = Embedder(_model=_FakeModel())
    assert embedder.embed_batch([]) == []


def test_embed_query_returns_float_list() -> None:
    """A query embedding is a list of floats with the model dimension."""
    embedder = Embedder(_model=_FakeModel())
    vector = embedder.embed_query("search query")

    assert isinstance(vector, list)
    assert len(vector) == _DIM
    assert all(isinstance(v, float) for v in vector)


def test_dimension_matches_model() -> None:
    """dimension exposes the underlying model embedding size."""
    embedder = Embedder(_model=_FakeModel())
    assert embedder.dimension == _DIM


def test_model_loaded_lazily_and_cached() -> None:
    """The model is constructed once and reused across calls."""
    model = _FakeModel()
    embedder = Embedder(_model=model)

    assert model.calls == 0

    embedder.embed("first")
    embedder.embed_batch(["a", "b"])
    embedder.embed_query("q")

    assert model.calls == 3
    assert embedder._model is model


def test_default_model_constant_is_bge_small() -> None:
    """The default model name is the BAAI bge-small-en-v1.5."""
    assert DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"
