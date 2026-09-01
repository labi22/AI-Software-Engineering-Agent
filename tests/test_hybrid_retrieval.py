"""Tests for Reciprocal Rank Fusion, SymbolBoostReranker, and HybridRetriever."""

import pytest

from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.lexical import BM25Index
from ai_software_engineering_agent.models import (
    CodeChunk,
    MetadataFilter,
    RetrievalResult,
    RetrievalStrategy,
)
from ai_software_engineering_agent.retrieval import (
    HybridRetriever,
    QueryExpander,
    SymbolBoostReranker,
    reciprocal_rank_fusion,
)
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


def test_reciprocal_rank_fusion_merges_and_scores_correctly():
    c1 = CodeChunk(
        chunk_id="c1",
        repo_id="r1",
        file_path="a.py",
        start_line=1,
        end_line=10,
        symbol_name="foo",
        language="python",
        content="def foo(): pass",
    )
    c2 = CodeChunk(
        chunk_id="c2",
        repo_id="r1",
        file_path="b.py",
        start_line=1,
        end_line=10,
        symbol_name="bar",
        language="python",
        content="def bar(): pass",
    )
    c3 = CodeChunk(
        chunk_id="c3",
        repo_id="r1",
        file_path="c.py",
        start_line=1,
        end_line=10,
        symbol_name="baz",
        language="python",
        content="def baz(): pass",
    )

    # Dense ranked: [c1 (rank 1), c2 (rank 2)]
    dense_results = [RetrievalResult(chunk=c1, score=0.9), RetrievalResult(chunk=c2, score=0.7)]
    # BM25 ranked: [c2 (rank 1), c3 (rank 2)]
    bm25_results = [RetrievalResult(chunk=c2, score=15.0), RetrievalResult(chunk=c3, score=8.0)]

    # Fused RRF: c2 has rank 2 in dense + rank 1 in BM25 -> highest combined score
    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60, limit=3)
    assert len(fused) == 3
    assert fused[0].chunk.chunk_id == "c2"
    assert fused[0].debug_info is not None
    assert fused[0].debug_info.dense_rank == 2
    assert fused[0].debug_info.bm25_rank == 1


def test_symbol_boost_reranker_boosts_matching_symbols():
    reranker = SymbolBoostReranker(symbol_exact_boost=0.3)
    c1 = CodeChunk(
        chunk_id="c1",
        repo_id="r1",
        file_path="utils.py",
        start_line=1,
        end_line=10,
        symbol_name="compute_something",
        language="python",
        content="def compute_something(): pass",
    )
    c2 = CodeChunk(
        chunk_id="c2",
        repo_id="r1",
        file_path="pricing.py",
        start_line=1,
        end_line=10,
        symbol_name="price_bond",
        language="python",
        content="def price_bond(): pass",
    )

    # Initial ranking where c1 has higher baseline score than c2
    candidates = [
        RetrievalResult(chunk=c1, score=0.8),
        RetrievalResult(chunk=c2, score=0.6),
    ]

    reranked = reranker.rerank("How does price_bond work?", candidates)
    # c2 receives exact symbol boost and surpasses c1
    assert reranked[0].chunk.chunk_id == "c2"
    assert reranked[0].score > reranked[1].score


def test_query_expander_adds_domain_synonyms():
    expanded = QueryExpander.expand("Calculate YTM for zero coupon bond")
    assert "yield_to_maturity" in expanded or "yield" in expanded


@pytest.mark.asyncio
async def test_hybrid_retriever_dense_bm25_and_hybrid_strategies():
    vector_store = InMemoryVectorStore()
    bm25_index = BM25Index()
    embedding_client = FakeEmbeddingClient(dimension=32)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_client=embedding_client,
    )

    c1 = CodeChunk(
        chunk_id="c1",
        repo_id="bond-repo",
        file_path="models/curve.py",
        start_line=1,
        end_line=20,
        symbol_name="YieldCurve",
        language="python",
        content="class YieldCurve:\n    def zero_rate(self, t): return 0.05",
    )
    c2 = CodeChunk(
        chunk_id="c2",
        repo_id="bond-repo",
        file_path="models/pricing.py",
        start_line=1,
        end_line=25,
        symbol_name="bond_price",
        language="python",
        content="def bond_price(rate, coupon): return 102.5",
    )

    # Store in both indices
    bm25_index.index_chunks([c1, c2])
    embeddings = await embedding_client.embed([c1.content, c2.content])
    await vector_store.store_chunks([c1, c2], embeddings)

    # 1. Test BM25 strategy
    bm25_res = await retriever.retrieve("zero_rate", strategy=RetrievalStrategy.BM25, limit=2)
    assert len(bm25_res) >= 1
    assert bm25_res[0].chunk.chunk_id == "c1"

    # 2. Test Dense strategy
    dense_res = await retriever.retrieve("discounting cash flows", strategy=RetrievalStrategy.DENSE, limit=2)
    assert len(dense_res) >= 1

    # 3. Test Hybrid strategy
    hybrid_res = await retriever.retrieve("YieldCurve zero_rate", strategy=RetrievalStrategy.HYBRID, limit=2)
    assert len(hybrid_res) >= 1
    assert hybrid_res[0].chunk.chunk_id == "c1"
    assert hybrid_res[0].debug_info is not None
