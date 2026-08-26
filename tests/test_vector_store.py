"""Tests for VectorStore abstractions and similarity search."""

import pytest

from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.models import CodeChunk
from ai_software_engineering_agent.vector_store import (
    InMemoryVectorStore,
    VectorStoreConfigurationError,
    create_vector_store,
)


@pytest.mark.asyncio
async def test_in_memory_vector_store_stores_and_searches_chunks():
    store = InMemoryVectorStore()
    await store.initialize()

    c1 = CodeChunk(
        chunk_id="c1",
        repo_id="repo-a",
        file_path="src/math.py",
        start_line=1,
        end_line=10,
        symbol_name="add",
        language="python",
        content="def add(a, b): return a + b",
    )
    c2 = CodeChunk(
        chunk_id="c2",
        repo_id="repo-a",
        file_path="src/calc.py",
        start_line=1,
        end_line=10,
        symbol_name="multiply",
        language="python",
        content="def multiply(a, b): return a * b",
    )
    c3 = CodeChunk(
        chunk_id="c3",
        repo_id="repo-b",
        file_path="src/other.py",
        start_line=1,
        end_line=5,
        symbol_name="other",
        language="python",
        content="def other(): pass",
    )

    # Synthetic embeddings (orthogonal / distinct)
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.8, 0.6, 0.0]
    emb3 = [0.0, 0.0, 1.0]

    await store.store_chunks([c1, c2, c3], [emb1, emb2, emb3])
    assert await store.count_chunks() == 3
    assert await store.count_chunks("repo-a") == 2

    # Query close to emb1
    query_emb = [0.99, 0.1, 0.0]
    results = await store.search(query_emb, limit=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score > results[1].score

    # Search with repo filter
    results_b = await store.search(query_emb, limit=5, repo_id="repo-b")
    assert len(results_b) == 1
    assert results_b[0].chunk.chunk_id == "c3"


@pytest.mark.asyncio
async def test_in_memory_vector_store_deletes_repository():
    store = InMemoryVectorStore()
    c1 = CodeChunk(
        chunk_id="c1",
        repo_id="repo-a",
        file_path="a.py",
        start_line=1,
        end_line=5,
        symbol_name=None,
        language="python",
        content="x = 1",
    )
    await store.store_chunks([c1], [[1.0, 0.0]])
    assert await store.count_chunks() == 1

    await store.delete_repository("repo-a")
    assert await store.count_chunks() == 0


def test_create_vector_store_factory_validates_pgvector_settings():
    settings_no_db = Settings(
        llm_provider="openai",
        llm_model="gpt-5.2",
        openai_api_key="key",
        request_timeout_seconds=30,
        allowed_repository_roots=(),
        vector_store_type="pgvector",
        database_url=None,
    )
    with pytest.raises(VectorStoreConfigurationError, match="DATABASE_URL must be set"):
        create_vector_store(settings_no_db)
