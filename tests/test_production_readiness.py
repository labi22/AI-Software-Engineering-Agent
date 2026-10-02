"""Comprehensive unit and integration tests for production readiness milestones."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from ai_software_engineering_agent.agent_persistence import (
    AgentRunRecord,
    InMemoryAgentRunStore,
)
from ai_software_engineering_agent.app import create_app
from ai_software_engineering_agent.auth import hash_api_key
from ai_software_engineering_agent.checkpointer import (
    PostgresCheckpointSaver,
    create_checkpointer,
)
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.coordination import (
    InMemoryCoordinationService,
    create_coordination_service,
)
from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.ingestion_jobs import (
    InMemoryIngestionJobStore,
)
from ai_software_engineering_agent.llm import (
    FakeLLMClient,
    LLMClient,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from ai_software_engineering_agent.llm_admission import (
    AdmissionControlledLLMClient,
    create_admission_controlled_client,
)
from ai_software_engineering_agent.metrics import GLOBAL_METRICS
from ai_software_engineering_agent.models import IngestionSummary, RepositorySpec
from ai_software_engineering_agent.vector_store import InMemoryVectorStore
from ai_software_engineering_agent.worker import process_next_job


# ---------------------------------------------------------------------------
# Test Helpers & Stubs
# ---------------------------------------------------------------------------


def make_test_settings(allowed_roots: tuple[Path, ...] = ()) -> Settings:
    return Settings(
        llm_provider="fake",
        llm_model="test-model",
        openai_api_key="mock-key",
        request_timeout_seconds=5.0,
        allowed_repository_roots=allowed_roots,
        embedding_provider="fake",
        vector_store_type="memory",
        rate_limit_per_minute=100,
        tenant_max_concurrent_runs=2,
    )


class FailingOrThrottledLLM:
    """Mock LLM simulating 429 rate limit or timeouts before succeeding."""

    def __init__(self, failures_before_success: int = 1, error_type: str = "rate_limit") -> None:
        self.attempts = 0
        self.failures_before_success = failures_before_success
        self.error_type = error_type

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            if self.error_type == "rate_limit":
                raise LLMProviderError("HTTP 429 Too Many Requests: Rate limit reached")
            elif self.error_type == "timeout":
                await asyncio.sleep(0.5)
                raise asyncio.TimeoutError()
            else:
                raise LLMProviderError("503 Service Unavailable")
        return LLMResponse(
            text="Recovered successfully",
            provider="mock",
            model="mock",
            provider_request_id="mock-1",
        )


class MockRAGService:
    """Mock RAGService for testing ingestion worker."""

    def __init__(self) -> None:
        self.ingested_specs: list[RepositorySpec] = []

    async def ingest_repository(self, spec: RepositorySpec, **kwargs) -> IngestionSummary:
        self.ingested_specs.append(spec)
        return IngestionSummary(
            repo_id=spec.repo_id,
            files_scanned=5,
            files_parsed=5,
            chunks_created=12,
            total_tokens=1500,
        )


# ---------------------------------------------------------------------------
# Phase 5A: Ingestion Queue & Worker Tests
# ---------------------------------------------------------------------------


def test_in_memory_ingestion_job_store_lifecycle():
    store = InMemoryIngestionJobStore()
    job = store.enqueue("org-1", {"repository_path": "tests/data", "repo_id": "test-repo"})

    assert job.status == "queued"
    assert job.organization_id == "org-1"
    assert job.attempts == 0

    # Claim next job
    claimed = store.claim_next()
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == "running"
    assert claimed.attempts == 1

    # Queue should be empty now
    assert store.claim_next() is None

    # Finish job
    store.finish(claimed.id, result={"chunks": 10})
    finished = store.get(job.id, "org-1")
    assert finished is not None
    assert finished.status == "succeeded"
    assert finished.result == {"chunks": 10}
    assert finished.completed_at is not None

    # Tenant isolation
    assert store.get(job.id, "other-org") is None


@pytest.mark.asyncio
async def test_worker_processes_job_successfully(tmp_path: Path):
    repo = tmp_path / "sample-repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def hello(): return 'world'\n", encoding="utf-8")

    settings = make_test_settings(allowed_roots=(repo,))
    store = InMemoryIngestionJobStore()
    job = store.enqueue(
        "org-test",
        {"repository_path": str(repo), "repo_id": "worker-repo"},
    )

    rag_mock = MockRAGService()
    claimed = await process_next_job(store, rag_mock, settings)

    assert claimed is not None
    assert len(rag_mock.ingested_specs) == 1
    assert rag_mock.ingested_specs[0].repo_id == "worker-repo"

    updated = store.get(job.id, "org-test")
    assert updated.status == "succeeded"
    assert updated.result["chunks_created"] == 12


def test_async_ingest_and_job_status_endpoints(tmp_path: Path):
    settings = make_test_settings(allowed_roots=(tmp_path,))
    app = create_app(
        settings=settings,
        client_factory=lambda _: FakeLLMClient(response_text="ok"),
        embedding_factory=lambda _: FakeEmbeddingClient(),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )
    client = TestClient(app)

    # Enqueue async ingestion
    res = client.post(
        "/v1/repositories/ingest/async",
        json={"repository_path": str(tmp_path), "repo_id": "async-repo"},
    )
    assert res.status_code == 202
    data = res.json()
    assert data["status"] == "queued"
    assert data["repo_id"] == "async-repo"
    job_id = data["job_id"]

    # Poll status
    status_res = client.get(f"/v1/ingestion-jobs/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "queued"
    assert status_data["attempts"] == 0


# ---------------------------------------------------------------------------
# Phase 5B: Agent Run Persistence & Checkpoints Tests
# ---------------------------------------------------------------------------


def test_in_memory_agent_run_store():
    store = InMemoryAgentRunStore()
    record = AgentRunRecord(
        id="run-1",
        organization_id="org-alpha",
        repo_id="repo-1",
        task="Test task",
        status="finished",
        total_steps=3,
        duration_ms=250.0,
        answer="Completed answer",
        citations=[{"file_path": "a.py", "start_line": 1, "end_line": 10}],
        steps=[{"step_number": 1, "status": "thinking"}],
    )
    store.save_run(record)

    fetched = store.get_run("run-1", "org-alpha")
    assert fetched is not None
    assert fetched.task == "Test task"
    assert fetched.answer == "Completed answer"

    # Tenant scoping
    assert store.get_run("run-1", "org-beta") is None

    # List runs
    runs = store.list_runs("org-alpha")
    assert len(runs) == 1
    assert runs[0].id == "run-1"


def test_agent_run_endpoint_persists_run_and_retrieves_audit():
    settings = make_test_settings()
    store = InMemoryAgentRunStore()
    app = create_app(
        settings=settings,
        client_factory=lambda _: FakeLLMClient(response_text="LangGraph final answer."),
        embedding_factory=lambda _: FakeEmbeddingClient(),
        vector_store_factory=lambda _: InMemoryVectorStore(),
        agent_run_store=store,
    )
    client = TestClient(app)

    run_res = client.post("/v1/agent/run", json={"task": "Persist me"})
    assert run_res.status_code == 200
    run_id = run_res.json()["request_id"]

    # Retrieve via GET /v1/agent/runs/{run_id}
    detail_res = client.get(f"/v1/agent/runs/{run_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["run_id"] == run_id
    assert detail_data["answer"] == "LangGraph final answer."
    assert detail_data["status"] == "finished"

    # List runs via GET /v1/agent/runs
    list_res = client.get("/v1/agent/runs")
    assert list_res.status_code == 200
    assert any(r["run_id"] == run_id for r in list_res.json())


# ---------------------------------------------------------------------------
# Phase 5C: Redis Coordination (Idempotency, Rate Limits, Concurrency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_coordination_idempotency():
    coord = InMemoryCoordinationService()

    # First reservation succeeds
    allowed, cached = await coord.check_or_reserve_idempotency("key-1", "org-1", ttl_seconds=60)
    assert allowed is True
    assert cached is None

    # Second check while in-progress reports conflict (False, None)
    allowed2, cached2 = await coord.check_or_reserve_idempotency("key-1", "org-1")
    assert allowed2 is False
    assert cached2 is None

    # Save completed response
    await coord.save_idempotent_response("key-1", "org-1", {"answer": "cached answer"}, status_code=200)

    # Third check returns cached response (True, cached)
    allowed3, cached3 = await coord.check_or_reserve_idempotency("key-1", "org-1")
    assert allowed3 is True
    assert cached3 is not None
    assert cached3["body"] == {"answer": "cached answer"}


@pytest.mark.asyncio
async def test_in_memory_coordination_rate_limiting():
    coord = InMemoryCoordinationService()

    # Allow up to 3 requests per 60 seconds
    for _ in range(3):
        allowed, rem, retry_after = await coord.check_rate_limit("user-1", max_requests=3, window_seconds=60)
        assert allowed is True

    # 4th request must be rejected
    blocked, rem, retry_after = await coord.check_rate_limit("user-1", max_requests=3, window_seconds=60)
    assert blocked is False
    assert rem == 0
    assert retry_after > 0


@pytest.mark.asyncio
async def test_in_memory_coordination_concurrency_lease():
    coord = InMemoryCoordinationService()

    lease1 = await coord.acquire_concurrency_lease("tenant-1", max_concurrent=2)
    lease2 = await coord.acquire_concurrency_lease("tenant-1", max_concurrent=2)
    assert lease1 is not None
    assert lease2 is not None

    # Exceeding limit
    lease3 = await coord.acquire_concurrency_lease("tenant-1", max_concurrent=2)
    assert lease3 is None

    # Release one lease
    await coord.release_concurrency_lease("tenant-1", lease1)

    # Now acquisition succeeds
    lease4 = await coord.acquire_concurrency_lease("tenant-1", max_concurrent=2)
    assert lease4 is not None


def test_agent_run_idempotency_key_header():
    settings = make_test_settings()
    app = create_app(
        settings=settings,
        client_factory=lambda _: FakeLLMClient(response_text="First execution answer"),
        embedding_factory=lambda _: FakeEmbeddingClient(),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )
    client = TestClient(app)

    headers = {"Idempotency-Key": "req-unique-123"}
    res1 = client.post("/v1/agent/run", json={"task": "Repeatable task"}, headers=headers)
    assert res1.status_code == 200
    first_data = res1.json()

    # Second request with identical key returns cached response
    res2 = client.post("/v1/agent/run", json={"task": "Repeatable task"}, headers=headers)
    assert res2.status_code == 200
    second_data = res2.json()

    assert first_data["request_id"] == second_data["request_id"]
    assert first_data["answer"] == second_data["answer"]


# ---------------------------------------------------------------------------
# Phase 5D: LLM Admission Control & Quota Resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admission_controlled_llm_client_retries_on_rate_limit():
    failing_client = FailingOrThrottledLLM(failures_before_success=2, error_type="rate_limit")
    controlled_client = AdmissionControlledLLMClient(
        inner_client=failing_client,
        rpm_limit=100,
        max_retries=3,
        base_delay=0.01,
        max_delay=0.1,
    )

    response = await controlled_client.generate(LLMRequest(prompt="test"))
    assert response.text == "Recovered successfully"
    assert failing_client.attempts == 3


@pytest.mark.asyncio
async def test_admission_controlled_llm_client_exhausts_retries():
    always_failing = FailingOrThrottledLLM(failures_before_success=10, error_type="rate_limit")
    controlled_client = AdmissionControlledLLMClient(
        inner_client=always_failing,
        rpm_limit=100,
        max_retries=2,
        base_delay=0.01,
        max_delay=0.05,
    )

    with pytest.raises(LLMQuotaExhaustedError):
        await controlled_client.generate(LLMRequest(prompt="exhaust test"))


# ---------------------------------------------------------------------------
# Phase 5E: Observability, Metrics & Health
# ---------------------------------------------------------------------------


def test_metrics_and_health_endpoints():
    settings = make_test_settings()
    app = create_app(
        settings=settings,
        client_factory=lambda _: FakeLLMClient(response_text="Metrics test"),
        embedding_factory=lambda _: FakeEmbeddingClient(),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )
    client = TestClient(app)

    # Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["status"] == "ok"
    assert "components" in h_data
    assert h_data["components"]["api"] == "healthy"

    # Hit an endpoint to generate request metrics
    client.post("/v1/generate", json={"prompt": "Generate telemetry"})

    # Check Prometheus metrics
    m_res = client.get("/metrics")
    assert m_res.status_code == 200
    assert "http_requests_total" in m_res.text
    assert "http_request_duration_seconds" in m_res.text
