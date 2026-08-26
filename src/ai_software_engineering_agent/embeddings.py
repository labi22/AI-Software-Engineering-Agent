"""Embedding client abstraction, OpenAI implementation, and offline test mocks."""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any, Protocol

from .config import Settings


class EmbeddingError(RuntimeError):
    """Base error for embedding operations."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when the embedding client is misconfigured."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the embedding provider fails."""


class EmbeddingClient(Protocol):
    """Protocol for generating vector embeddings from text."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of texts."""

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding vector for a single query text."""


class OpenAIEmbeddingClient:
    """OpenAI Embeddings adapter using the official client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise EmbeddingConfigurationError(
                    "The OpenAI SDK is not installed. Install the project dependencies first."
                ) from error
            client = OpenAI(api_key=api_key, timeout=timeout_seconds)

        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Batch texts to prevent hitting single-request payload limits
        batch_size = 100
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = await asyncio.to_thread(
                    self._client.embeddings.create,
                    model=self._model,
                    input=batch,
                )
            except Exception as error:
                raise EmbeddingProviderError(
                    f"OpenAI embedding request failed: {error}"
                ) from error

            # Sort by index to maintain ordering
            sorted_data = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed([text])
        if not embeddings:
            raise EmbeddingProviderError("No embedding was returned for query.")
        return embeddings[0]


class FakeEmbeddingClient:
    """Deterministic offline embedding client for tests and offline runs."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dimension

        # Generate deterministic pseudo-vector using token frequencies and sha256
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            words = [text.lower()]

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            for idx in range(min(len(digest), self.dimension)):
                slot = (idx * 37 + int(digest[idx])) % self.dimension
                vec[slot] += float(digest[idx]) / 255.0

        # L2-normalize the vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Factory creating the configured embedding client."""
    if settings.embedding_provider == "fake":
        return FakeEmbeddingClient(dimension=settings.embedding_dimension)

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise EmbeddingConfigurationError(
                "OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai."
            )
        return OpenAIEmbeddingClient(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimension,
            timeout_seconds=settings.request_timeout_seconds,
        )

    raise EmbeddingConfigurationError(
        f"Unsupported embedding provider: {settings.embedding_provider}."
    )
