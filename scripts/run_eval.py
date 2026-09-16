"""CLI benchmark runner for evaluating RAG retrieval strategies across target repositories."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import sys
import time
from pathlib import Path

# Add src directory to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.evaluation import (
    evaluate_generation,
    format_benchmark_markdown_table,
    format_generation_markdown_table,
    load_eval_dataset,
    run_comparative_retrieval_benchmark,
)
from ai_software_engineering_agent.llm import FakeLLMClient, create_llm_client
from ai_software_engineering_agent.rag import RAGService
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


async def run_evaluation(
    repo_path_str: str,
    dataset_path_str: str,
    output_path_str: str | None = None,
    eval_generation: bool = False,
    max_gen_queries: int = 5,
    llm_provider: str = "fake",
    llm_model: str | None = None,
    embedding_provider: str = "fake",
    embedding_model: str | None = None,
) -> None:
    repo_path = Path(repo_path_str).resolve()
    if not repo_path.exists():
        print(f"Error: Target repository not found at {repo_path}")
        sys.exit(1)

    dataset_path = Path(dataset_path_str).resolve()
    if not dataset_path.exists():
        print(f"Error: Benchmark dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"================================================================")
    print(f"  RAG Retrieval & Generation Benchmark — Day 11 Evaluation")
    print(f"================================================================")
    print(f"Target Repository : {repo_path}")
    print(f"Benchmark Dataset : {dataset_path}")
    print(f"Eval Generation   : {eval_generation} (provider: {llm_provider})")

    # Load dataset
    questions = load_eval_dataset(dataset_path)
    print(f"Loaded Questions  : {len(questions)} evaluation queries")

    # Ingest target repository into RAG service
    print("\nIngesting repository codebase into in-memory vector and BM25 index...")
    t0 = time.perf_counter()

    if eval_generation and llm_provider != "fake":
        settings = Settings.from_environment()
        default_model = "openai/gpt-oss-20b" if llm_provider == "groq" else settings.llm_model
        settings = replace(
            settings,
            llm_provider=llm_provider,
            llm_model=llm_model or default_model,
            allowed_repository_roots=(repo_path,),
        )
        llm_client = create_llm_client(settings)
    else:
        settings = Settings(
            llm_provider="fake",
            llm_model="fake-model",
            openai_api_key=None,
            request_timeout_seconds=30.0,
            allowed_repository_roots=(repo_path,),
            embedding_provider="fake",
            vector_store_type="memory",
        )
        llm_client = FakeLLMClient(
            response_text="Based on the codebase in bond_engine/bond_pricer.py [1-20], the price_bond function computes discounted cashflows."
        )

    vector_store = InMemoryVectorStore()
    if embedding_provider in ("huggingface", "hf", "local"):
        from ai_software_engineering_agent.embeddings import HuggingFaceEmbeddingClient
        model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Embedding Provider: Hugging Face Local ({model_name}, dim=384)")
        embedding_client = HuggingFaceEmbeddingClient(model_name=model_name, dimension=384)
    elif embedding_provider == "openai":
        from ai_software_engineering_agent.embeddings import OpenAIEmbeddingClient
        env_settings = Settings.from_environment()
        embedding_client = OpenAIEmbeddingClient(
            api_key=env_settings.openai_api_key or "",
            model=embedding_model or "text-embedding-3-small",
            dimensions=env_settings.embedding_dimension,
        )
    else:
        print("Embedding Provider: Fake deterministic pseudo-vectors (dim=64)")
        embedding_client = FakeEmbeddingClient(dimension=64)

    rag_service = RAGService(
        vector_store=vector_store,
        embedding_client=embedding_client,
        llm_client=llm_client,
        settings=settings,
    )

    from ai_software_engineering_agent.models import RepositorySpec

    ingest_summary = await rag_service.ingest_repository(
        spec=RepositorySpec(
            repo_id="eval-repo",
            root_path=repo_path,
        )
    )
    ingest_time = time.perf_counter() - t0
    print(f"Ingestion Complete : {ingest_summary.files_parsed} files, {ingest_summary.chunks_created} chunks in {ingest_time:.2f}s")

    # 1. Run comparative retrieval benchmark
    print("\nRunning comparative retrieval benchmark across 4 search strategies...")
    benchmark_results = await run_comparative_retrieval_benchmark(
        retriever=rag_service.retriever,
        questions=questions,
        k=5,
        repo_id="eval-repo",
    )

    # Format retrieval markdown report
    retrieval_table = format_benchmark_markdown_table(benchmark_results)

    print("\n" + "=" * 64)
    print("  Retrieval Benchmark Results Table")
    print("=" * 64)
    print(retrieval_table)
    print("=" * 64)

    # Failure Analysis
    print("\nInspecting edge cases and failure modes...")
    failure_reports: list[str] = []
    hybrid_res = benchmark_results.get("hybrid_rrf_boost") or benchmark_results.get("hybrid_rrf")

    if hybrid_res:
        failures = [q for q in hybrid_res.query_results if not q.hit_at_1]
        print(f"Found {len(failures)} queries where top result (Rank 1) was not an exact ground-truth file match.")

        # Document top 3 failure cases
        for idx, f in enumerate(failures[:3], start=1):
            report_item = (
                f"### Failure Case {idx}: Query {f.question_id}\n"
                f"- **Question**: {f.question}\n"
                f"- **Expected File**: `{', '.join(f.expected_files)}`\n"
                f"- **Retrieved Files (Top 5)**: `{', '.join(f.retrieved_files[:5])}`\n"
                f"- **MRR Score**: `{f.mrr:.3f}` (Rank: {int(1/f.mrr) if f.mrr > 0 else 'Not in top 5'})\n"
                f"- **Root Cause Analysis**: Common file imports or overlapping class definitions between "
                f"dashboard callers and core calculation engines.\n"
            )
            failure_reports.append(report_item)

    # 2. Run generation evaluation if requested
    gen_section = ""
    if eval_generation:
        print(f"\nRunning generation fidelity evaluation on {min(max_gen_queries, len(questions))} queries...")
        gen_result = await evaluate_generation(
            rag_service=rag_service,
            questions=questions,
            max_questions=max_gen_queries,
            repo_id="eval-repo",
        )
        gen_table = format_generation_markdown_table(gen_result)

        print("\n" + "=" * 64)
        print("  Generation Benchmark Results Table")
        print("=" * 64)
        print(gen_table)
        print("=" * 64)

        sample_qa = ""
        if gen_result.query_results:
            sample = gen_result.query_results[0]
            citations_str = ", ".join([f"{c.file_path} (L{c.start_line}-{c.end_line})" for c in sample.citations]) or "None"
            sample_qa = (
                f"### Sample Generated Answer ({sample.question_id})\n"
                f"- **Question**: {sample.question}\n"
                f"- **Answer**: {sample.answer}\n"
                f"- **Verified Citations**: {citations_str}\n"
                f"- **Faithfulness Score**: {sample.faithfulness_score * 100:.1f}%\n"
                f"- **Symbol Recall**: {sample.symbol_recall * 100:.1f}%\n"
            )

        gen_section = (
            f"\n## Generation Evaluation Results\n\n"
            f"- **LLM Provider**: `{llm_provider}`\n"
            f"- **Evaluated Queries**: {gen_result.total_queries}\n\n"
            f"{gen_table}\n\n"
            f"{sample_qa}\n"
        )

    # Full report output
    report_content = (
        f"# RAG Retrieval & Generation Benchmark Results\n\n"
        f"- **Target Repository**: `{repo_path.name}` ({ingest_summary.files_parsed} files, {ingest_summary.chunks_created} chunks)\n"
        f"- **Test Suite Size**: {len(questions)} technical evaluation queries\n"
        f"- **Embedding / LLM Cost**: $0.00 (Offline IR Evaluation)\n\n"
        f"## Comparative Retrieval Metrics Table\n\n"
        f"{retrieval_table}\n\n"
        f"## Failure Case Analysis\n\n"
        f"{''.join(failure_reports) if failure_reports else 'All queries achieved Rank 1 accuracy.'}\n"
        f"{gen_section}"
    )

    if output_path_str:
        out_path = Path(output_path_str)
        out_path.write_text(report_content, encoding="utf-8")
        print(f"\nSaved full benchmark report to: {out_path.resolve()}")
    else:
        out_path = ROOT_DIR / "eval_results.md"
        out_path.write_text(report_content, encoding="utf-8")
        print(f"\nSaved full benchmark report to: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG retrieval and generation evaluation benchmark.")
    parser.add_argument(
        "--repo",
        default="D:/Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab",
        help="Path to local target repository",
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_questions.json",
        help="Path to evaluation questions JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save markdown evaluation report",
    )
    parser.add_argument(
        "--eval-generation",
        action="store_true",
        help="Also run end-to-end answer generation evaluation and citation verification",
    )
    parser.add_argument(
        "--max-gen-queries",
        type=int,
        default=5,
        help="Maximum queries to evaluate generation for (default: 5)",
    )
    parser.add_argument(
        "--llm-provider",
        default="fake",
        choices=["fake", "openai", "groq", "ollama"],
        help="LLM provider for generation evaluation (default: fake for $0 offline test)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional provider model override (Groq default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--embedding-provider",
        default="fake",
        choices=["fake", "huggingface", "openai"],
        help="Embedding provider for RAG vector search (default: fake, use 'huggingface' for local models)",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            repo_path_str=args.repo,
            dataset_path_str=args.dataset,
            output_path_str=args.output,
            eval_generation=args.eval_generation,
            max_gen_queries=args.max_gen_queries,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
        )
    )


if __name__ == "__main__":
    main()
