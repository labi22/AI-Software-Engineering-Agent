"""Unit and integration tests for the evaluation pipeline and IR metrics."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.evaluation import (
    AnswerEvalMetrics,
    EvalQuestion,
    GenerationBenchmarkResult,
    PerQueryGenerationResult,
    RetrievalMetrics,
    StrategyBenchmarkResult,
    compute_hit_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_citations_faithfulness,
    evaluate_generation,
    evaluate_retrieval_strategy,
    format_benchmark_markdown_table,
    format_generation_markdown_table,
    load_eval_dataset,
    normalize_file_path,
    run_comparative_retrieval_benchmark,
)
from ai_software_engineering_agent.lexical import BM25Index
from ai_software_engineering_agent.models import Citation, CodeChunk, RetrievalStrategy
from ai_software_engineering_agent.retrieval import HybridRetriever
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Metric Math Unit Tests
# ---------------------------------------------------------------------------


def test_normalize_file_path() -> None:
    """Path normalizer removes backslashes, leading slashes, and relative dots."""
    assert normalize_file_path("src\\calc.py") == "src/calc.py"
    assert normalize_file_path("./src/calc.py") == "src/calc.py"
    assert normalize_file_path("/src/calc.py") == "src/calc.py"
    assert normalize_file_path("calc.py") == "calc.py"


def test_compute_hit_at_k() -> None:
    """Hit@K correctly checks presence in the top K retrieved results."""
    retrieved = ["utils/helpers.py", "bond_engine/pricer.py", "yield_curve/rates.py"]
    expected = ["bond_engine/pricer.py"]

    # Rank 2 is within top 2, 3, 5, but NOT top 1
    assert compute_hit_at_k(retrieved, expected, k=1) == 0.0
    assert compute_hit_at_k(retrieved, expected, k=2) == 1.0
    assert compute_hit_at_k(retrieved, expected, k=3) == 1.0
    assert compute_hit_at_k(retrieved, expected, k=5) == 1.0

    # Not found in retrieved
    assert compute_hit_at_k(retrieved, ["missing/file.py"], k=5) == 0.0
    # Empty expected
    assert compute_hit_at_k(retrieved, [], k=5) == 1.0


def test_compute_mrr() -> None:
    """MRR computes reciprocal rank 1/rank for the first relevant document."""
    retrieved = ["a.py", "b.py", "c.py", "d.py"]

    # Rank 1 -> 1.0
    assert compute_mrr(retrieved, ["a.py"]) == 1.0
    # Rank 2 -> 0.5
    assert compute_mrr(retrieved, ["b.py"]) == 0.5
    # Rank 4 -> 0.25
    assert compute_mrr(retrieved, ["d.py"]) == 0.25
    # Not found -> 0.0
    assert compute_mrr(retrieved, ["z.py"]) == 0.0
    # Multiple expected: first occurrence wins
    assert compute_mrr(retrieved, ["c.py", "b.py"]) == 0.5


def test_compute_recall_at_k() -> None:
    """Recall@K calculates fraction of expected files in top K."""
    retrieved = ["a.py", "b.py", "c.py", "d.py"]
    expected = ["b.py", "c.py"]

    # In top 1: only 'a.py' retrieved -> 0/2 = 0.0
    assert compute_recall_at_k(retrieved, expected, k=1) == 0.0
    # In top 2: 'b.py' retrieved -> 1/2 = 0.5
    assert compute_recall_at_k(retrieved, expected, k=2) == 0.5
    # In top 3: 'b.py' and 'c.py' retrieved -> 2/2 = 1.0
    assert compute_recall_at_k(retrieved, expected, k=3) == 1.0


def test_compute_precision_at_k() -> None:
    """Precision@K calculates fraction of top K retrieved files that are relevant."""
    retrieved = ["a.py", "b.py", "c.py", "d.py", "e.py"]
    expected = ["b.py", "d.py"]

    # In top 2: ['a.py', 'b.py'], 1 relevant -> 1/2 = 0.5
    assert compute_precision_at_k(retrieved, expected, k=2) == 0.5
    # In top 4: ['a.py', 'b.py', 'c.py', 'd.py'], 2 relevant -> 2/4 = 0.5
    assert compute_precision_at_k(retrieved, expected, k=4) == 0.5
    # In top 5: 2 relevant -> 2/5 = 0.4
    assert compute_precision_at_k(retrieved, expected, k=5) == 0.4


# ---------------------------------------------------------------------------
# Dataset Loading Tests
# ---------------------------------------------------------------------------


def test_load_eval_dataset() -> None:
    """Evaluation dataset loads and validates the 25 target benchmark questions."""
    dataset_path = Path("data/eval_questions.json")
    assert dataset_path.exists(), "data/eval_questions.json must exist"

    questions = load_eval_dataset(dataset_path)
    assert len(questions) == 25

    for q in questions:
        assert q.id.startswith("q")
        assert len(q.question.strip()) > 10
        assert len(q.expected_files) > 0
        assert len(q.expected_symbols) > 0


def test_load_eval_dataset_missing_file() -> None:
    """Loader raises FileNotFoundError when path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_eval_dataset("data/non_existent_dataset.json")


# ---------------------------------------------------------------------------
# Benchmark Runner Integration Tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def mock_retriever() -> HybridRetriever:
    """Create a populated HybridRetriever fixture with predictable chunks."""
    vector_store = InMemoryVectorStore()
    embedding_client = FakeEmbeddingClient(dimension=8)
    bm25_index = BM25Index()

    chunks = [
        CodeChunk(
            chunk_id="c1",
            repo_id="repo-1",
            file_path="bond_engine/bond_pricer.py",
            start_line=1,
            end_line=20,
            symbol_name="price_bond",
            language="python",
            content="def price_bond(bond, yield_curve):\n    return sum(cashflow * discount_factor)",
        ),
        CodeChunk(
            chunk_id="c2",
            repo_id="repo-1",
            file_path="bond_engine/key_rate_shock.py",
            start_line=1,
            end_line=25,
            symbol_name="key_rate_duration",
            language="python",
            content="class KeyRateShockedCurve:\n    def key_rate_duration(self, bond): pass",
        ),
        CodeChunk(
            chunk_id="c3",
            repo_id="repo-1",
            file_path="yield_curve/curve_builder.py",
            start_line=1,
            end_line=30,
            symbol_name="YieldCurve",
            language="python",
            content="class YieldCurve:\n    def discount_factor(self, t): return exp(-r * t)",
        ),
    ]

    bm25_index.index_chunks(chunks)
    # Dense vector store populated with fake embeddings
    embeddings = [[0.1 * i for i in range(8)] for _ in chunks]
    await vector_store.store_chunks(chunks, embeddings)

    return HybridRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        bm25_index=bm25_index,
    )


@pytest.mark.asyncio
async def test_evaluate_retrieval_strategy(mock_retriever: HybridRetriever) -> None:
    """Strategy evaluation executes queries and calculates valid aggregated metrics."""
    questions = [
        EvalQuestion(
            id="q1",
            question="How is a bond priced using discount factors?",
            expected_files=["bond_engine/bond_pricer.py"],
            expected_symbols=["price_bond"],
        ),
        EvalQuestion(
            id="q2",
            question="Where is the YieldCurve class implemented?",
            expected_files=["yield_curve/curve_builder.py"],
            expected_symbols=["YieldCurve"],
        ),
    ]

    result = await evaluate_retrieval_strategy(
        retriever=mock_retriever,
        questions=questions,
        strategy=RetrievalStrategy.BM25,
        k=5,
    )

    assert result.strategy_name == "Bm25"
    assert result.metrics.total_queries == 2
    assert 0.0 <= result.metrics.hit_at_1 <= 1.0
    assert 0.0 <= result.metrics.mrr <= 1.0
    assert len(result.query_results) == 2


@pytest.mark.asyncio
async def test_run_comparative_retrieval_benchmark(mock_retriever: HybridRetriever) -> None:
    """Comparative benchmark runs across all 4 strategy variants and formats table."""
    questions = [
        EvalQuestion(
            id="q1",
            question="How is a bond priced in price_bond?",
            expected_files=["bond_engine/bond_pricer.py"],
            expected_symbols=["price_bond"],
        ),
    ]

    results = await run_comparative_retrieval_benchmark(
        retriever=mock_retriever,
        questions=questions,
        k=3,
    )

    assert "dense" in results
    assert "bm25" in results
    assert "hybrid_rrf" in results
    assert "hybrid_rrf_boost" in results

    table_md = format_benchmark_markdown_table(results)
    assert "| Retrieval Strategy |" in table_md
    assert "| Hit@1 |" in table_md
    assert "| MRR |" in table_md
    assert "Hybrid" in table_md


# ---------------------------------------------------------------------------
# Answer Fidelity Tests
# ---------------------------------------------------------------------------


def test_evaluate_citations_faithfulness() -> None:
    """Citation evaluation flags unverified line citations and scores faithfulness."""
    c1 = Citation(file_path="a.py", start_line=1, end_line=10, is_verified=True)
    c2 = Citation(file_path="b.py", start_line=11, end_line=20, is_verified=False)

    metrics = evaluate_citations_faithfulness([c1, c2])
    assert metrics.total_citations == 2
    assert metrics.verified_citations == 1
    assert metrics.unverified_citations == 1
    assert metrics.faithfulness_score == 0.5

    # 100% verified citations
    metrics_all_verified = evaluate_citations_faithfulness([c1, c1])
    assert metrics_all_verified.faithfulness_score == 1.0

    # Empty citations fallback
    empty_metrics = evaluate_citations_faithfulness([])
    assert empty_metrics.faithfulness_score == 1.0


# ---------------------------------------------------------------------------
# Generation Evaluation Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_generation_with_fake_llm(mock_retriever: HybridRetriever) -> None:
    """Generation evaluation runs end-to-end with FakeLLMClient and produces valid metrics."""
    from ai_software_engineering_agent.config import Settings
    from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
    from ai_software_engineering_agent.llm import FakeLLMClient
    from ai_software_engineering_agent.rag import RAGService

    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key=None,
        request_timeout_seconds=30.0,
        allowed_repository_roots=(),
        embedding_provider="fake",
        vector_store_type="memory",
    )
    fake_llm = FakeLLMClient(
        response_text="The price_bond function in bond_engine/bond_pricer.py computes discounted cashflows using the YieldCurve."
    )
    rag_service = RAGService(
        vector_store=mock_retriever.vector_store,
        embedding_client=mock_retriever.embedding_client,
        llm_client=fake_llm,
        retriever=mock_retriever,
        settings=settings,
    )

    questions = [
        EvalQuestion(
            id="q1",
            question="How is a bond priced using discount factors?",
            expected_files=["bond_engine/bond_pricer.py"],
            expected_symbols=["price_bond", "YieldCurve"],
        ),
        EvalQuestion(
            id="q2",
            question="Where is key rate duration computed?",
            expected_files=["bond_engine/key_rate_shock.py"],
            expected_symbols=["key_rate_duration"],
        ),
    ]

    result = await evaluate_generation(
        rag_service=rag_service,
        questions=questions,
        max_questions=2,
    )

    assert isinstance(result, GenerationBenchmarkResult)
    assert result.total_queries == 2
    assert 0.0 <= result.mean_faithfulness <= 1.0
    assert 0.0 <= result.mean_symbol_recall <= 1.0
    assert result.duration_seconds > 0.0
    assert len(result.query_results) == 2

    # q1 answer contains both "price_bond" and "YieldCurve" → symbol recall = 1.0
    q1_result = result.query_results[0]
    assert q1_result.question_id == "q1"
    assert q1_result.symbol_recall == 1.0
    assert "price_bond" in q1_result.matched_symbols
    assert "YieldCurve" in q1_result.matched_symbols

    # q2 answer does NOT contain "key_rate_duration" → symbol recall = 0.0
    q2_result = result.query_results[1]
    assert q2_result.question_id == "q2"
    assert q2_result.symbol_recall == 0.0


def test_format_generation_markdown_table() -> None:
    """Generation table formatter produces valid markdown with expected columns."""
    result = GenerationBenchmarkResult(
        mean_faithfulness=0.85,
        mean_symbol_recall=0.70,
        total_citations=10,
        verified_citations=8,
        total_queries=5,
        duration_seconds=2.5,
        query_results=[],
    )

    table_md = format_generation_markdown_table(result)
    assert "| Total Queries |" in table_md
    assert "| Mean Faithfulness |" in table_md
    assert "| Mean Symbol Recall |" in table_md
    assert "85.0%" in table_md
    assert "70.0%" in table_md
    assert "| 5 " in table_md
