"""Tests for the RAG service, prompt construction, citations, and end-to-end question answering."""

from pathlib import Path
import pytest

from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.llm import LLMRequest, LLMResponse
from ai_software_engineering_agent.models import CodeChunk, RepositorySpec, RetrievalResult
from ai_software_engineering_agent.rag import (
    RAGService,
    extract_citations,
    format_context_prompt,
)
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


class FakeLLMClient:
    def __init__(self, answer: str = "Grounded response citing [models/curve.py:10-25]."):
        self.answer = answer
        self.last_request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            text=self.answer,
            provider="fake",
            model="fake-model",
            provider_request_id="req-123",
        )


def test_format_context_prompt_includes_headers_and_guidelines():
    chunk = CodeChunk(
        chunk_id="chk1",
        repo_id="repo1",
        file_path="src/bootstrap.py",
        start_line=12,
        end_line=30,
        symbol_name="bootstrap_yield_curve",
        language="python",
        content="def bootstrap_yield_curve(): pass",
    )
    prompt = format_context_prompt([RetrievalResult(chunk=chunk, score=0.92)])

    assert "src/bootstrap.py" in prompt
    assert "Lines 12-30" in prompt
    assert "bootstrap_yield_curve" in prompt
    assert "RETRIEVED CONTEXT" in prompt


def test_extract_citations_finds_file_and_line_tags():
    text = "The curve is calibrated in [models/curve.py:10-25] and discounted in [utils/math.py:5-15 (discount)]."
    citations = extract_citations(text, retrieved_chunks=[])

    assert len(citations) == 2
    assert citations[0].file_path == "models/curve.py"
    assert citations[0].start_line == 10
    assert citations[0].end_line == 25
    assert citations[1].file_path == "utils/math.py"
    assert citations[1].symbol_name == "discount"


@pytest.mark.asyncio
async def test_rag_service_end_to_end_flow(tmp_path: Path):
    # Setup test repository files
    repo_dir = tmp_path / "bond_repo"
    repo_dir.mkdir()

    curve_file = repo_dir / "curve.py"
    curve_file.write_text(
        "class YieldCurve:\n"
        "    def __init__(self, tenors, rates):\n"
        "        self.tenors = tenors\n"
        "        self.rates = rates\n"
        "\n"
        "    def zero_rate(self, t: float) -> float:\n"
        "        \"\"\"Interpolates zero rate.\"\"\"\n"
        "        return 0.05\n",
        encoding="utf-8",
    )

    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-5.2",
        openai_api_key="mock-key",
        request_timeout_seconds=30,
        allowed_repository_roots=(repo_dir,),
    )

    vector_store = InMemoryVectorStore()
    embedding_client = FakeEmbeddingClient(dimension=64)
    llm_client = FakeLLMClient(
        answer="The zero rate is interpolated by `YieldCurve.zero_rate` [curve.py:6-8]."
    )

    rag_service = RAGService(
        vector_store=vector_store,
        embedding_client=embedding_client,
        llm_client=llm_client,
        settings=settings,
    )

    # 1. Ingest repository
    spec = RepositorySpec(repo_id="bond-repo", root_path=repo_dir)
    summary = await rag_service.ingest_repository(spec)

    assert summary.files_scanned == 1
    assert summary.chunks_created >= 1
    assert await vector_store.count_chunks("bond-repo") >= 1

    # 2. Answer query
    rag_response = await rag_service.answer_query(
        query="How is the zero rate calculated?",
        repo_id="bond-repo",
    )

    assert "zero rate" in rag_response.answer
    assert len(rag_response.citations) >= 1
    assert rag_response.citations[0].file_path == "curve.py"
    assert rag_response.citations[0].start_line == 6
    assert rag_response.citations[0].end_line == 8
    assert len(rag_response.retrieved_chunks) >= 1
