"""Pluggable text-embedding backends.

The app defaults to a real local model (fastembed / ONNX, no GPU required). Tests
and CI use the dependency-free ``FakeEmbedder`` so they never download a model.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from app.config import settings


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words hashing embeddings — no downloads, good for tests."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FastEmbedEmbedder(Embedder):
    """Real embeddings via fastembed (ONNX). Model is downloaded on first use."""

    def __init__(self, model_name: str, dim: int) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(texts)]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the process-wide embedder singleton."""
    global _embedder
    if _embedder is None:
        if settings.use_fake_embeddings:
            _embedder = FakeEmbedder(settings.embed_dim)
        else:
            _embedder = FastEmbedEmbedder(settings.embed_model, settings.embed_dim)
    return _embedder
