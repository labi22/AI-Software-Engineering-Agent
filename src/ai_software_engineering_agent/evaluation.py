"""Empirical evaluation pipeline for retrieval quality, hybrid ranking, and answer fidelity."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .models import RetrievalStrategy
from .retrieval import HybridRetriever


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuestion:
    """Golden benchmark question with expected relevant files and symbols."""

    id: str
    question: str
    expected_files: list[str]
    expected_symbols: list[str] = field(default_factory=list)
    category: str = "general"


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate information retrieval metrics for a strategy across a test set."""

    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    total_queries: int


@dataclass(frozen=True)
class PerQueryRetrievalResult:
    """Detailed retrieval metrics for an individual benchmark question."""

    question_id: str
    question: str
    expected_files: list[str]
    retrieved_files: list[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    mrr: float
    recall_at_5: float
    precision_at_5: float


@dataclass(frozen=True)
class StrategyBenchmarkResult:
    """Benchmark results for a single retrieval strategy."""

    strategy_name: str
    metrics: RetrievalMetrics
    duration_seconds: float
    query_results: list[PerQueryRetrievalResult]


@dataclass(frozen=True)
class AnswerEvalMetrics:
    """Answer fidelity metrics based on citation interval overlap and relevance."""

    faithfulness_score: float
    citation_precision: float
    verified_citations: int
    unverified_citations: int
    total_citations: int


@dataclass(frozen=True)
class PerQueryGenerationResult:
    """Detailed generation metrics for an individual benchmark question."""

    question_id: str
    question: str
    answer: str
    citations: list[Any]
    faithfulness_score: float
    symbol_recall: float
    expected_symbols: list[str]
    matched_symbols: list[str]
    duration_seconds: float


@dataclass(frozen=True)
class GenerationBenchmarkResult:
    """Aggregate benchmark results for answer generation fidelity and grounding."""

    mean_faithfulness: float
    mean_symbol_recall: float
    total_citations: int
    verified_citations: int
    total_queries: int
    duration_seconds: float
    query_results: list[PerQueryGenerationResult]


# ---------------------------------------------------------------------------
# Metric Calculation Functions
# ---------------------------------------------------------------------------


def normalize_file_path(path: str | Path) -> str:
    """Normalize file path for uniform comparison across platforms."""
    p_str = str(path).replace("\\", "/").strip()
    if p_str.startswith("./"):
        p_str = p_str[2:]
    return p_str.lstrip("/")


def _file_matches(candidate: str, expected_set: set[str]) -> bool:
    """Check if candidate path matches any expected path, allowing relative suffix matches."""
    cand = normalize_file_path(candidate)
    for exp in expected_set:
        norm_exp = normalize_file_path(exp)
        if cand == norm_exp or cand.endswith("/" + norm_exp) or norm_exp.endswith("/" + cand):
            return True
    return False


def compute_hit_at_k(retrieved_files: Sequence[str], expected_files: Sequence[str], k: int) -> float:
    """Return 1.0 if at least one expected file is in top-k retrieved files, else 0.0."""
    if not expected_files:
        return 1.0
    expected_set = {normalize_file_path(f) for f in expected_files}
    top_k = retrieved_files[:k]
    for rf in top_k:
        if _file_matches(rf, expected_set):
            return 1.0
    return 0.0


def compute_mrr(retrieved_files: Sequence[str], expected_files: Sequence[str]) -> float:
    """Compute Reciprocal Rank (1/rank) for the first relevant file retrieved."""
    if not expected_files:
        return 1.0
    expected_set = {normalize_file_path(f) for f in expected_files}
    for rank, rf in enumerate(retrieved_files, start=1):
        if _file_matches(rf, expected_set):
            return 1.0 / rank
    return 0.0


def compute_recall_at_k(retrieved_files: Sequence[str], expected_files: Sequence[str], k: int) -> float:
    """Compute Recall@K: proportion of expected files present in top-k retrieved files."""
    if not expected_files:
        return 1.0
    top_k = retrieved_files[:k]
    matched_expected = 0
    for exp in expected_files:
        norm_exp = normalize_file_path(exp)
        if any(_file_matches(rf, {norm_exp}) for rf in top_k):
            matched_expected += 1
    return matched_expected / len(expected_files)


def compute_precision_at_k(retrieved_files: Sequence[str], expected_files: Sequence[str], k: int) -> float:
    """Compute Precision@K: proportion of top-k retrieved files that are relevant."""
    if k <= 0:
        return 0.0
    if not expected_files:
        return 0.0
    top_k = retrieved_files[:k]
    expected_set = {normalize_file_path(f) for f in expected_files}
    matched_retrieved = sum(1 for rf in top_k if _file_matches(rf, expected_set))
    return matched_retrieved / k


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------


def load_eval_dataset(dataset_path: str | Path) -> list[EvalQuestion]:
    """Load benchmark questions from a JSON file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    questions: list[EvalQuestion] = []
    for item in raw_data:
        questions.append(
            EvalQuestion(
                id=str(item.get("id", "")),
                question=str(item.get("question", "")),
                expected_files=[str(f) for f in item.get("expected_files", [])],
                expected_symbols=[str(s) for s in item.get("expected_symbols", [])],
                category=str(item.get("category", "general")),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Benchmark Execution Engine
# ---------------------------------------------------------------------------


async def evaluate_retrieval_strategy(
    retriever: HybridRetriever,
    questions: list[EvalQuestion],
    strategy: RetrievalStrategy | str,
    k: int = 5,
    symbol_boost: bool = True,
    repo_id: str | None = None,
) -> StrategyBenchmarkResult:
    """Evaluate a single retrieval strategy across all benchmark questions."""
    start_time = time.perf_counter()
    query_results: list[PerQueryRetrievalResult] = []

    hit_1_sum = 0.0
    hit_3_sum = 0.0
    hit_5_sum = 0.0
    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0

    strat_enum = RetrievalStrategy(strategy) if isinstance(strategy, str) else strategy
    strategy_label = f"{strat_enum.value.capitalize()}"
    if strat_enum == RetrievalStrategy.HYBRID:
        strategy_label += " (RRF k=60)"
        if symbol_boost:
            strategy_label += " + Symbol Boost"

    original_reranker = getattr(retriever, "reranker", None)
    if not symbol_boost:
        class PassthroughReranker:
            def rerank(self, q: str, results: list[Any]) -> list[Any]:
                return results
        retriever.reranker = PassthroughReranker()

    from .models import MetadataFilter

    try:
        for q in questions:
            retrieval_res = await retriever.retrieve(
                query=q.question,
                strategy=strat_enum,
                limit=max(k, 5),
                filter=MetadataFilter(repo_id=repo_id) if repo_id else None,
            )

            retrieved_files = [item.chunk.file_path for item in retrieval_res]

            h1 = compute_hit_at_k(retrieved_files, q.expected_files, k=1)
            h3 = compute_hit_at_k(retrieved_files, q.expected_files, k=3)
            h5 = compute_hit_at_k(retrieved_files, q.expected_files, k=5)
            rec = compute_recall_at_k(retrieved_files, q.expected_files, k=k)
            prec = compute_precision_at_k(retrieved_files, q.expected_files, k=k)
            mrr = compute_mrr(retrieved_files, q.expected_files)

            hit_1_sum += h1
            hit_3_sum += h3
            hit_5_sum += h5
            recall_sum += rec
            precision_sum += prec
            mrr_sum += mrr

            query_results.append(
                PerQueryRetrievalResult(
                    question_id=q.id,
                    question=q.question,
                    expected_files=q.expected_files,
                    retrieved_files=retrieved_files,
                    hit_at_1=bool(h1 > 0),
                    hit_at_3=bool(h3 > 0),
                    hit_at_5=bool(h5 > 0),
                    mrr=mrr,
                    recall_at_5=rec,
                    precision_at_5=prec,
                )
            )
    finally:
        if original_reranker is not None:
            retriever.reranker = original_reranker

    n = max(len(questions), 1)
    duration = time.perf_counter() - start_time

    metrics = RetrievalMetrics(
        hit_at_1=round(hit_1_sum / n, 4),
        hit_at_3=round(hit_3_sum / n, 4),
        hit_at_5=round(hit_5_sum / n, 4),
        recall_at_5=round(recall_sum / n, 4),
        precision_at_5=round(precision_sum / n, 4),
        mrr=round(mrr_sum / n, 4),
        total_queries=len(questions),
    )

    return StrategyBenchmarkResult(
        strategy_name=strategy_label,
        metrics=metrics,
        duration_seconds=round(duration, 3),
        query_results=query_results,
    )


async def run_comparative_retrieval_benchmark(
    retriever: HybridRetriever,
    questions: list[EvalQuestion],
    k: int = 5,
    repo_id: str | None = None,
) -> dict[str, StrategyBenchmarkResult]:
    """Run all retrieval configurations side-by-side to benchmark ranking enhancements."""
    configs = [
        ("dense", RetrievalStrategy.DENSE, False),
        ("bm25", RetrievalStrategy.BM25, False),
        ("hybrid_rrf", RetrievalStrategy.HYBRID, False),
        ("hybrid_rrf_boost", RetrievalStrategy.HYBRID, True),
    ]

    results: dict[str, StrategyBenchmarkResult] = {}
    for key, strategy, boost in configs:
        res = await evaluate_retrieval_strategy(
            retriever=retriever,
            questions=questions,
            strategy=strategy,
            k=k,
            symbol_boost=boost,
            repo_id=repo_id,
        )
        results[key] = res

    return results


def format_benchmark_markdown_table(benchmark_results: dict[str, StrategyBenchmarkResult]) -> str:
    """Format comparative benchmark results into a clean GitHub-flavored markdown table."""
    headers = [
        "Retrieval Strategy",
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Recall@5",
        "Precision@5",
        "MRR",
        "Latency / Query",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for key, res in benchmark_results.items():
        m = res.metrics
        avg_latency_ms = (res.duration_seconds / max(m.total_queries, 1)) * 1000.0
        row = [
            res.strategy_name,
            f"{m.hit_at_1 * 100:.1f}%",
            f"{m.hit_at_3 * 100:.1f}%",
            f"{m.hit_at_5 * 100:.1f}%",
            f"{m.recall_at_5 * 100:.1f}%",
            f"{m.precision_at_5 * 100:.1f}%",
            f"{m.mrr:.3f}",
            f"{avg_latency_ms:.1f} ms",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer Evaluation
# ---------------------------------------------------------------------------


def evaluate_citations_faithfulness(citations: Sequence[Any]) -> AnswerEvalMetrics:
    """Evaluate citation faithfulness and precision using verification flags."""
    if not citations:
        return AnswerEvalMetrics(
            faithfulness_score=1.0,
            citation_precision=1.0,
            verified_citations=0,
            unverified_citations=0,
            total_citations=0,
        )

    verified_count = sum(1 for c in citations if getattr(c, "is_verified", False))
    total_count = len(citations)
    unverified_count = total_count - verified_count

    faithfulness = verified_count / total_count if total_count > 0 else 1.0

    return AnswerEvalMetrics(
        faithfulness_score=round(faithfulness, 4),
        citation_precision=round(faithfulness, 4),
        verified_citations=verified_count,
        unverified_citations=unverified_count,
        total_citations=total_count,
    )


async def evaluate_generation(
    rag_service: Any,
    questions: Sequence[EvalQuestion],
    strategy: RetrievalStrategy | str = RetrievalStrategy.HYBRID,
    max_questions: int | None = None,
    repo_id: str | None = None,
) -> GenerationBenchmarkResult:
    """Evaluate end-to-end RAG answer generation, citation faithfulness, and symbol grounding."""
    from .models import MetadataFilter

    eval_subset = list(questions)[:max_questions] if max_questions else list(questions)
    strat_enum = RetrievalStrategy(strategy) if isinstance(strategy, str) else strategy

    query_results: list[PerQueryGenerationResult] = []
    faithfulness_sum = 0.0
    symbol_recall_sum = 0.0
    total_citations_count = 0
    verified_citations_count = 0

    overall_start = time.perf_counter()

    for q in eval_subset:
        t0 = time.perf_counter()
        rag_response = await rag_service.answer_query(
            query=q.question,
            strategy=strat_enum,
            filter=MetadataFilter(repo_id=repo_id) if repo_id else None,
        )
        duration = time.perf_counter() - t0

        citation_metrics = evaluate_citations_faithfulness(rag_response.citations)

        # Evaluate symbol grounding
        answer_text = rag_response.answer.lower()
        matched_symbols = [s for s in q.expected_symbols if s.lower() in answer_text]
        sym_recall = len(matched_symbols) / len(q.expected_symbols) if q.expected_symbols else 1.0

        faithfulness_sum += citation_metrics.faithfulness_score
        symbol_recall_sum += sym_recall
        total_citations_count += citation_metrics.total_citations
        verified_citations_count += citation_metrics.verified_citations

        query_results.append(
            PerQueryGenerationResult(
                question_id=q.id,
                question=q.question,
                answer=rag_response.answer,
                citations=rag_response.citations,
                faithfulness_score=citation_metrics.faithfulness_score,
                symbol_recall=round(sym_recall, 4),
                expected_symbols=q.expected_symbols,
                matched_symbols=matched_symbols,
                duration_seconds=round(duration, 3),
            )
        )

    n = max(len(eval_subset), 1)
    total_duration = time.perf_counter() - overall_start

    return GenerationBenchmarkResult(
        mean_faithfulness=round(faithfulness_sum / n, 4),
        mean_symbol_recall=round(symbol_recall_sum / n, 4),
        total_citations=total_citations_count,
        verified_citations=verified_citations_count,
        total_queries=len(eval_subset),
        duration_seconds=round(total_duration, 3),
        query_results=query_results,
    )


def format_generation_markdown_table(result: GenerationBenchmarkResult) -> str:
    """Format generation evaluation metrics into a clean markdown table."""
    avg_latency = (result.duration_seconds / max(result.total_queries, 1)) * 1000.0
    headers = [
        "Total Queries",
        "Mean Faithfulness",
        "Mean Symbol Recall",
        "Verified Citations",
        "Total Citations",
        "Avg Latency / Query",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        (
            f"| {result.total_queries} "
            f"| {result.mean_faithfulness * 100:.1f}% "
            f"| {result.mean_symbol_recall * 100:.1f}% "
            f"| {result.verified_citations} "
            f"| {result.total_citations} "
            f"| {avg_latency:.1f} ms |"
        ),
    ]
    return "\n".join(lines)
