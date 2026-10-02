"""Tests for API endpoints including generation, repository ingestion, and grounded RAG."""

import asyncio
from dataclasses import dataclass, replace
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from ai_software_engineering_agent.app import create_app
from ai_software_engineering_agent.auth import hash_api_key
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.llm import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    OpenAIResponsesClient,
    create_llm_client,
)
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


@dataclass
class FakeLLMClient:
    response_text: str = "A generated answer citing [models/pricing.py:5-20]."

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=self.response_text,
            provider="fake",
            model="fake-model",
            provider_request_id="fake-request-1",
        )


def fake_settings(allowed_roots: tuple[Path, ...] = ()) -> Settings:
    return Settings(
        llm_provider="openai",
        llm_model="test-model",
        openai_api_key="mock-key",
        request_timeout_seconds=30,
        allowed_repository_roots=allowed_roots,
        embedding_provider="fake",
        vector_store_type="memory",
    )


def test_settings_rejects_invalid_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings.from_environment({"LLM_PROVIDER": "copilot"})


def test_settings_supports_groq_and_reads_its_key():
    settings = Settings.from_environment(
        {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "test-groq-key"}
    )

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key == "test-groq-key"


def test_settings_requires_postgres_for_enabled_authentication():
    digest = hash_api_key("test-key")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Settings.from_environment({
            "AUTH_ENABLED": "true",
            "AUTH_API_KEY_HASHES": f"key-1:{digest}:org-1",
        })


def test_groq_factory_requires_a_groq_key():
    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-20b",
        openai_api_key=None,
        request_timeout_seconds=30,
        allowed_repository_roots=(),
    )

    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        create_llm_client(settings)


def test_openai_adapter_normalizes_a_response_without_a_network_call():
    class FakeResponses:
        def __init__(self) -> None:
            self.request_arguments: dict[str, str] | None = None

        def create(self, **kwargs: str) -> object:
            self.request_arguments = kwargs
            return type(
                "Response",
                (),
                {"output_text": "Normalized output.", "model": "returned-model", "id": "resp_123"},
            )()

    fake_responses = FakeResponses()
    fake_client = type("Client", (), {"responses": fake_responses})()
    adapter = OpenAIResponsesClient(
        api_key="not-used-by-the-fake",
        model="configured-model",
        timeout_seconds=30,
        client=fake_client,
    )

    result = asyncio.run(
        adapter.generate(LLMRequest(prompt="Hello", system_instruction="Be concise."))
    )

    assert fake_responses.request_arguments == {
        "model": "configured-model",
        "input": "Hello",
        "instructions": "Be concise.",
    }
    assert result == LLMResponse(
        text="Normalized output.",
        provider="openai",
        model="returned-model",
        provider_request_id="resp_123",
    )


def test_openai_compatible_adapter_preserves_the_configured_provider():
    fake_response = type(
        "Response",
        (),
        {"output_text": "Groq output.", "model": "openai/gpt-oss-20b", "id": "resp_456"},
    )()
    fake_client = type(
        "Client",
        (),
        {"responses": type("Responses", (), {"create": lambda self, **_: fake_response})()},
    )()
    adapter = OpenAIResponsesClient(
        api_key="not-used-by-the-fake",
        model="openai/gpt-oss-20b",
        timeout_seconds=30,
        provider="groq",
        client=fake_client,
    )

    result = asyncio.run(adapter.generate(LLMRequest(prompt="Hello")))

    assert result.provider == "groq"


def test_health_does_not_require_an_api_key():
    client = TestClient(create_app(settings=fake_settings()))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "components" in body
    assert body["components"]["api"] == "healthy"


def test_generate_uses_injected_fake_client():
    app = create_app(
        settings=fake_settings(),
        client_factory=lambda _: FakeLLMClient(),
    )
    client = TestClient(app)

    response = client.post("/v1/generate", json={"prompt": "Explain a yield curve."})

    assert response.status_code == 200
    assert response.json()["text"].startswith("A generated answer")
    assert response.json()["provider"] == "fake"
    assert response.json()["provider_request_id"] == "fake-request-1"
    assert response.json()["request_id"]


def test_generate_rejects_empty_prompt():
    client = TestClient(
        create_app(settings=fake_settings(), client_factory=lambda _: FakeLLMClient())
    )

    response = client.post("/v1/generate", json={"prompt": ""})

    assert response.status_code == 422


def test_generate_returns_actionable_configuration_error():
    no_key_settings = Settings(
        llm_provider="openai",
        llm_model="test-model",
        openai_api_key=None,
        request_timeout_seconds=30,
        allowed_repository_roots=(),
    )
    client = TestClient(create_app(settings=no_key_settings))

    response = client.post("/v1/generate", json={"prompt": "Hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "OPENAI_API_KEY must be set when LLM_PROVIDER=openai."


def test_generate_hides_provider_error_details():
    def failing_factory(_: Settings) -> FakeLLMClient:
        raise LLMProviderError("Sensitive provider detail")

    client = TestClient(
        create_app(settings=fake_settings(), client_factory=failing_factory)
    )

    response = client.post("/v1/generate", json={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "The configured LLM provider could not complete the request."


def test_ingest_and_rag_query_flow(tmp_path: Path):
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    code_file = repo_dir / "calc.py"
    code_file.write_text(
        "def compute_npv(rate: float, flows: list[float]) -> float:\n"
        "    \"\"\"Calculate NPV.\"\"\"\n"
        "    return sum(f / ((1 + rate) ** i) for i, f in enumerate(flows))\n",
        encoding="utf-8",
    )

    app = create_app(
        settings=fake_settings(allowed_roots=(repo_dir,)),
        client_factory=lambda _: FakeLLMClient(
            response_text="NPV is computed discounted in [calc.py:1-3]."
        ),
        embedding_factory=lambda _: FakeEmbeddingClient(dimension=32),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )
    client = TestClient(app)

    # 1. Ingest repository
    ingest_resp = client.post(
        "/v1/repositories/ingest",
        json={"repository_path": str(repo_dir), "repo_id": "test-repo"},
    )
    assert ingest_resp.status_code == 200
    data = ingest_resp.json()
    assert data["repo_id"] == "test-repo"
    assert data["files_scanned"] == 1
    assert data["chunks_created"] >= 1
    assert data["status"] == "success"

    # 2. Query RAG endpoint with hybrid retrieval and debug info
    query_resp = client.post(
        "/v1/rag/query",
        json={
            "query": "How is NPV computed?",
            "repo_id": "test-repo",
            "top_k": 3,
            "retrieval_strategy": "hybrid",
            "include_debug_info": True,
            "path_patterns": ["*.py"],
        },
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert "NPV is computed" in q_data["answer"]
    assert len(q_data["citations"]) >= 1
    assert q_data["citations"][0]["file_path"] == "calc.py"
    assert q_data["citations"][0]["is_verified"] is True
    assert len(q_data["retrieved_chunks"]) >= 1
    assert q_data["retrieved_chunks"][0]["debug_info"] is not None
    assert q_data["strategy_used"] == "hybrid"


def test_rag_query_rejects_invalid_strategy():
    app = create_app(
        settings=fake_settings(),
        client_factory=lambda _: FakeLLMClient(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/rag/query",
        json={"query": "test query", "retrieval_strategy": "invalid-strategy"},
    )
    assert response.status_code == 422


def test_ingest_github_repository_uses_isolated_clone_source(tmp_path: Path, monkeypatch):
    clone_path = tmp_path / "cloned-repository"
    clone_path.mkdir()
    (clone_path / "bond.py").write_text("def price():\n    return 100\n", encoding="utf-8")
    observed: dict[str, object] = {}

    @asynccontextmanager
    async def fake_clone(url: str, ref: str, settings):
        observed["url"] = url
        observed["ref"] = ref
        yield clone_path

    monkeypatch.setattr("ai_software_engineering_agent.app.cloned_github_repository", fake_clone)
    app = create_app(
        settings=fake_settings(),
        client_factory=lambda _: FakeLLMClient(),
        embedding_factory=lambda _: FakeEmbeddingClient(dimension=32),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )

    response = TestClient(app).post(
        "/v1/repositories/ingest",
        json={
            "github_url": "https://github.com/labi22/Fixed-Income-Analytics-Bond-Valuation-Platform",
            "ref": "main",
            "repo_id": "bond-platform",
        },
    )

    assert response.status_code == 200
    assert response.json()["repo_id"] == "bond-platform"
    assert observed == {
        "url": "https://github.com/labi22/Fixed-Income-Analytics-Bond-Valuation-Platform",
        "ref": "main",
    }


def test_ingest_rejects_ambiguous_or_missing_repository_source():
    client = TestClient(create_app(settings=fake_settings(), client_factory=lambda _: FakeLLMClient()))

    missing = client.post("/v1/repositories/ingest", json={})
    ambiguous = client.post(
        "/v1/repositories/ingest",
        json={"repository_path": "C:/repo", "github_url": "https://github.com/labi22/example"},
    )

    assert missing.status_code == 422
    assert ambiguous.status_code == 422


def test_authenticated_endpoints_require_a_valid_api_key():
    api_key = "test-secret-key"
    settings = replace(
        fake_settings(),
        auth_enabled=True,
        auth_api_key_hashes=(("test-key", hash_api_key(api_key), "org-a"),),
    )
    client = TestClient(create_app(settings=settings, client_factory=lambda _: FakeLLMClient()))

    missing = client.post("/v1/generate", json={"prompt": "hello"})
    invalid = client.post("/v1/generate", json={"prompt": "hello"}, headers={"X-API-Key": "wrong"})
    valid = client.post("/v1/generate", json={"prompt": "hello"}, headers={"X-API-Key": api_key})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert valid.status_code == 200


def test_tenant_cannot_claim_another_organizations_repository(tmp_path: Path):
    repo = tmp_path / "tenant-repo"
    repo.mkdir()
    (repo / "example.py").write_text("def value(): return 1\n", encoding="utf-8")
    first_key, second_key = "org-a-key", "org-b-key"
    settings = replace(
        fake_settings(allowed_roots=(repo,)),
        auth_enabled=True,
        auth_api_key_hashes=(
            ("key-a", hash_api_key(first_key), "org-a"),
            ("key-b", hash_api_key(second_key), "org-b"),
        ),
    )
    client = TestClient(create_app(
        settings=settings,
        client_factory=lambda _: FakeLLMClient(),
        embedding_factory=lambda _: FakeEmbeddingClient(dimension=32),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    ))

    created = client.post(
        "/v1/repositories/ingest",
        json={"repository_path": str(repo), "repo_id": "shared-id"},
        headers={"X-API-Key": first_key},
    )
    forbidden = client.post(
        "/v1/repositories/ingest",
        json={"repository_path": str(repo), "repo_id": "shared-id"},
        headers={"X-API-Key": second_key},
    )
    cross_tenant_query = client.post(
        "/v1/rag/query",
        json={"query": "value", "repo_id": "shared-id"},
        headers={"X-API-Key": second_key},
    )

    assert created.status_code == 200
    assert forbidden.status_code == 403
    assert cross_tenant_query.status_code == 403
