"""Unit tests for the Hugging Face local embeddings adapter."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    HuggingFaceEmbeddingClient,
    create_embedding_client,
)
from ai_software_engineering_agent.models import CodeChunk, RepositorySpec
from ai_software_engineering_agent.rag import RAGService
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


def test_create_embedding_client_huggingface():
    """Factory creates HuggingFaceEmbeddingClient when provider is huggingface."""
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key=None,
        request_timeout_seconds=30,
        allowed_repository_roots=(),
        embedding_provider="huggingface",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
    )
    client = create_embedding_client(settings)
    assert isinstance(client, HuggingFaceEmbeddingClient)
    assert client._dimension == 384
    assert client._model_name == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.asyncio
async def test_huggingface_embedding_client_with_mock_model():
    """HuggingFaceEmbeddingClient encodes texts and normalizes vectors."""
    mock_model = MagicMock()
    # 2 sample 4-dimensional vectors
    import numpy as np

    fake_vectors = np.array([[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=float)
    mock_model.encode.return_value = fake_vectors

    client = HuggingFaceEmbeddingClient(
        model_name="mock-model",
        dimension=4,
        client=mock_model,
    )

    embeddings = await client.embed(["hello world", "test"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 4
    # Check L2 normalization
    norm = math.sqrt(sum(v * v for v in embeddings[0]))
    assert abs(norm - 1.0) < 1e-5

    # Check query embedding
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    query_vec = await client.embed_query("query")
    assert len(query_vec) == 4
    assert query_vec[0] == 1.0


@pytest.mark.asyncio
async def test_huggingface_embedding_client_empty_input():
    """Empty text list returns empty embeddings without calling model."""
    mock_model = MagicMock()
    client = HuggingFaceEmbeddingClient(client=mock_model)
    res = await client.embed([])
    assert res == []
    mock_model.encode.assert_not_called()


@pytest.mark.asyncio
async def test_huggingface_embedding_client_missing_package():
    """Raises EmbeddingConfigurationError if sentence_transformers is not installed."""
    client = HuggingFaceEmbeddingClient()
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(EmbeddingConfigurationError):
            client._get_model()


@pytest.mark.asyncio
async def test_huggingface_embedding_client_error_handling():
    """Raises EmbeddingProviderError if model encoding fails."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("CUDA out of memory")
    client = HuggingFaceEmbeddingClient(client=mock_model)

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await client.embed(["text"])
    assert "Hugging Face embedding request failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_huggingface_integration_with_vector_store():
    """VectorStore correctly indexes and queries 384-dimensional embeddings."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones((1, 384), dtype=float) / math.sqrt(384)

    client = HuggingFaceEmbeddingClient(dimension=384, client=mock_model)
    vstore = InMemoryVectorStore()

    vec = await client.embed_query("find function")
    chunk = CodeChunk(
        chunk_id="chk_1",
        repo_id="test_repo",
        file_path="engine.py",
        start_line=1,
        end_line=10,
        content="def price_bond(): pass",
        language="python",
        symbol_name="price_bond",
    )

    await vstore.store_chunks([chunk], [vec])
    results = await vstore.search(vec, limit=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chk_1"
