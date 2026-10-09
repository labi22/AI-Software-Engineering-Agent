"""FastAPI application for the AI software engineering agent with production readiness."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent import AgentError
from .agent_persistence import (
    AgentRunRecord,
    AgentRunStore,
    InMemoryAgentRunStore,
    PostgresAgentRunStore,
)
from .agent_state import AgentStatus
from .auth import (
    APIKeyAuthenticator,
    AuthenticationError,
    AuthorizationError,
    InMemoryRepositoryOwnership,
    PostgresRepositoryOwnership,
    RepositoryOwnership,
    TenantPrincipal,
)
from .checkpointer import create_checkpointer
from .config import Settings
from .coordination import (
    CoordinationService,
    InMemoryCoordinationService,
    RedisCoordinationService,
    create_coordination_service,
)
from .embeddings import (
    EmbeddingClient,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    create_embedding_client,
)
from .engineering_tools import EngineeringToolContext, create_engineering_tool_registry
from .ingestion import (
    IngestionError,
    RemoteRepositoryError,
    RepositoryAccessError,
    cloned_github_repository,
    validate_repository_path,
)
from .ingestion_jobs import (
    IngestionJob,
    IngestionJobStore,
    InMemoryIngestionJobStore,
    PostgresIngestionJobStore,
)
from .langgraph_agent import LangGraphAgent
from .llm import (
    LLMClient,
    LLMConfigurationError,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRequest,
    LLMTimeoutError,
    create_llm_client,
)
from .llm_admission import create_admission_controlled_client
from .mcp import INVALID_PARAMS, PARSE_ERROR, JSONRPCError, JSONRPCResponse, MCPServer
from .metrics import GLOBAL_METRICS
from .models import (
    MetadataFilter,
    RepositorySpec,
    RetrievalStrategy,
)
from .rag import RAGError, RAGService
from .safety import PathSecurityError
from .tools import create_default_tool_registry
from .vector_store import (
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
    create_vector_store,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Validated API request for a single LLM generation."""

    prompt: str = Field(min_length=1, max_length=20_000)
    system_instruction: str | None = Field(default=None, max_length=10_000)


class GenerateResponse(BaseModel):
    """Normalized API response independent of the LLM provider SDK."""

    request_id: str
    text: str
    provider: str
    model: str
    provider_request_id: str | None


class IngestRequest(BaseModel):
    """Request payload to ingest an external code repository."""

    model_config = ConfigDict(extra="forbid")

    repository_path: str | None = Field(
        default=None,
        min_length=1,
        description="Absolute or relative local directory path",
    )
    github_url: str | None = Field(
        default=None,
        min_length=1,
        description="Public HTTPS GitHub repository URL",
    )
    ref: str = Field(
        default="HEAD",
        min_length=1,
        max_length=255,
        description="Branch or tag to shallow-clone",
    )
    repo_id: str | None = Field(default=None, description="Optional custom repository ID")
    name: str | None = Field(
        default=None, description="Optional display name for the repository"
    )

    @model_validator(mode="after")
    def exactly_one_repository_source(self) -> "IngestRequest":
        if bool(self.repository_path) == bool(self.github_url):
            raise ValueError("Provide exactly one of repository_path or github_url.")
        return self


class IngestResponse(BaseModel):
    """Metrics and status resulting from a synchronous repository ingestion run."""

    repo_id: str
    files_scanned: int
    files_parsed: int
    chunks_created: int
    total_tokens: int
    status: str = "success"


class AsyncIngestResponse(BaseModel):
    """Result of enqueuing an asynchronous repository ingestion job."""

    job_id: str
    repo_id: str
    status: str = "queued"
    created_at: str | None = None


class IngestionJobStatusResponse(BaseModel):
    """Status, attempts, and result metrics for a queued or executed ingestion job."""

    job_id: str
    organization_id: str
    status: str
    attempts: int
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class CitationModel(BaseModel):
    """Source citation identifying file and line numbers with verification status."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    snippet: str | None = None
    is_verified: bool = True


class RetrievalDebugInfoModel(BaseModel):
    """Detailed ranking metrics from dense and lexical retrieval systems."""

    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_boost: float = 0.0
    fusion_score: float = 0.0


class RetrievedChunkModel(BaseModel):
    """Retrieved context snippet with score and optional debug info."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    language: str
    content: str
    score: float
    debug_info: RetrievalDebugInfoModel | None = None


class RAGQueryRequest(BaseModel):
    """Request payload to ask a grounded question against an ingested repository."""

    query: str = Field(min_length=1, max_length=10_000)
    repo_id: str | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_strategy: str = Field(default="hybrid", description="dense, bm25, or hybrid")
    languages: list[str] | None = Field(
        default=None, description="Filter chunks by programming languages"
    )
    path_patterns: list[str] | None = Field(
        default=None, description="Filter chunks by glob file path patterns"
    )
    symbol_only: bool = Field(
        default=False, description="Filter chunks that contain named code symbols only"
    )
    include_debug_info: bool = Field(
        default=False, description="Include rank breakdown in response"
    )


class RAGQueryResponse(BaseModel):
    """Grounded answer with verified citations, retrieval context, and strategy metrics."""

    query: str
    answer: str
    citations: list[CitationModel]
    retrieved_chunks: list[RetrievedChunkModel]
    model: str
    provider: str
    strategy_used: str = "hybrid"
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    cache_hit: bool | None = None


class AgentStepModel(BaseModel):
    """A single step in the agent execution trace."""

    step_number: int
    status: str
    reasoning: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    observation: str | None = None
    is_error: bool = False


class AgentRunRequest(BaseModel):
    """Request payload to run the LangGraph agent on a task."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        max_length=10_000,
        description="The task or question for the agent to solve",
    )
    repository_path: str | None = Field(
        default=None,
        description="Optional local repository path to bind safe engineering tools to",
    )
    repo_id: str | None = Field(default=None, description="Optional repository ID")
    max_steps: int = Field(
        default=10, ge=1, le=30, description="Maximum number of reasoning/action steps"
    )


class AgentRunResponse(BaseModel):
    """Complete result of an agent run including the final answer and full trace."""

    request_id: str
    task: str
    answer: str
    status: str
    total_steps: int
    duration_ms: float
    steps: list[AgentStepModel]
    citations: list[CitationModel]
    orchestration_engine: str = "langgraph"


class AgentRunDetailResponse(BaseModel):
    """Persisted agent run record retrieved from durable audit storage."""

    run_id: str
    organization_id: str
    repo_id: str | None = None
    task: str
    status: str
    total_steps: int
    duration_ms: float
    answer: str
    citations: list[CitationModel]
    steps: list[AgentStepModel]
    created_at: str | None = None
    completed_at: str | None = None


ClientFactory = Callable[[Settings], LLMClient]
EmbeddingFactory = Callable[[Settings], EmbeddingClient]
VectorStoreFactory = Callable[[Settings], VectorStore]


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    settings: Settings | None = None,
    client_factory: ClientFactory = create_llm_client,
    embedding_factory: EmbeddingFactory = create_embedding_client,
    vector_store_factory: VectorStoreFactory = create_vector_store,
    authenticator: APIKeyAuthenticator | None = None,
    repository_ownership: RepositoryOwnership | None = None,
    ingestion_job_store: IngestionJobStore | None = None,
    agent_run_store: AgentRunStore | None = None,
    coordination_service: CoordinationService | None = None,
    checkpointer: Any | None = None,
) -> FastAPI:
    """Create the API application with injectable settings and production services."""
    app = FastAPI(
        title="AI Software Engineering Agent",
        version="0.3.0",
        description="Production-grade multi-tenant Agentic AI platform with LangGraph orchestration, hybrid RAG, and durable job queues.",
    )
    app.state.settings = settings or Settings.from_environment()
    app.state.client_factory = client_factory
    app.state.embedding_factory = embedding_factory
    app.state.vector_store_factory = vector_store_factory

    configured_keys = {
        digest: TenantPrincipal(organization_id=organization_id, key_id=key_id)
        for key_id, digest, organization_id in app.state.settings.auth_api_key_hashes
    }
    app.state.authenticator = authenticator or APIKeyAuthenticator(configured_keys)
    app.state.repository_ownership = repository_ownership or (
        PostgresRepositoryOwnership(
            app.state.settings.database_url, app.state.settings.auth_api_key_hashes
        )
        if app.state.settings.auth_enabled and app.state.settings.database_url
        else InMemoryRepositoryOwnership()
    )

    app.state.ingestion_job_store = ingestion_job_store or (
        PostgresIngestionJobStore(app.state.settings.database_url)
        if app.state.settings.database_url
        else InMemoryIngestionJobStore()
    )
    app.state.agent_run_store = agent_run_store or (
        PostgresAgentRunStore(app.state.settings.database_url)
        if app.state.settings.database_url
        else InMemoryAgentRunStore()
    )
    app.state.coordination_service = coordination_service or create_coordination_service(
        app.state.settings.redis_url
    )
    app.state.checkpointer = checkpointer or create_checkpointer(
        app.state.settings.database_url
    )
    app.state.metrics = GLOBAL_METRICS

    app.state.llm_client = None
    app.state.embedding_client = None
    app.state.vector_store = None
    app.state.rag_service = None

    # Middleware: Correlation ID, request timing, and metrics collection
    @app.middleware("http")
    async def correlation_id_and_metrics_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = req_id
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time
        response.headers["X-Request-ID"] = req_id

        # Update metrics counters & histograms
        request.app.state.metrics.inc_counter(
            "http_requests_total",
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        )
        request.app.state.metrics.observe_histogram(
            "http_request_duration_seconds",
            round(duration, 4),
        )
        return response

    def require_principal(request: Request) -> TenantPrincipal | None:
        if not request.app.state.settings.auth_enabled:
            return None
        try:
            return request.app.state.authenticator.authenticate(
                request.headers.get("X-API-Key")
            )
        except AuthenticationError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=str(error)
            ) from error

    def require_repository_owner(
        request: Request, repo_id: str, principal: TenantPrincipal | None
    ) -> None:
        if principal is None:
            return
        try:
            request.app.state.repository_ownership.require_owner(
                repo_id, principal.organization_id
            )
        except AuthorizationError as error:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=str(error)
            ) from error

    async def apply_rate_limit(
        request: Request, principal: TenantPrincipal | None = None
    ) -> None:
        ident = (
            principal.organization_id
            if principal
            else (request.client.host if request.client else "anonymous")
        )
        allowed, remaining, retry_after = (
            await request.app.state.coordination_service.check_rate_limit(
                identifier=ident,
                max_requests=request.app.state.settings.rate_limit_per_minute,
                window_seconds=60,
            )
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after}s.",
                headers={"Retry-After": str(int(retry_after))},
            )

    # -----------------------------------------------------------------------
    # Observability & Health Endpoints
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness and multi-tier readiness health probe."""
        components = {
            "api": "healthy",
            "database": "configured" if app.state.settings.database_url else "ephemeral",
            "redis": "configured" if app.state.settings.redis_url else "in_memory",
            "vector_store": app.state.settings.vector_store_type,
            "llm_provider": app.state.settings.llm_provider,
        }
        return {"status": "ok", "version": "0.3.0", "components": components}

    @app.get("/metrics")
    async def metrics():
        """Prometheus-compatible metrics exposition endpoint."""
        return PlainTextResponse(
            app.state.metrics.render_prometheus_text(),
            media_type="text/plain; version=0.0.4",
        )

    # -----------------------------------------------------------------------
    # Core Endpoints
    # -----------------------------------------------------------------------

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        try:
            client = _get_llm_client(request)
            result = await client.generate(
                LLMRequest(
                    prompt=payload.prompt,
                    system_instruction=payload.system_instruction,
                )
            )
        except LLMQuotaExhaustedError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": str(int(error.retry_after or 5))},
            ) from error
        except LLMTimeoutError as error:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(error),
            ) from error
        except LLMConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except LLMProviderError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The configured LLM provider could not complete the request.",
            ) from error

        return GenerateResponse(request_id=str(uuid4()), **result.__dict__)

    @app.post("/v1/repositories/ingest", response_model=IngestResponse)
    async def ingest_repository(
        payload: IngestRequest, request: Request
    ) -> IngestResponse:
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        repo_id = (
            payload.repo_id
            or (
                payload.github_url.rstrip("/").removesuffix(".git").split("/")[-1]
                if payload.github_url
                else Path(payload.repository_path or "").name
            )
            or "repo-1"
        )
        newly_claimed = False
        if principal is not None:
            try:
                newly_claimed = request.app.state.repository_ownership.claim_for_ingestion(
                    repo_id, principal.organization_id
                )
            except AuthorizationError as error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=str(error)
                ) from error
        try:
            rag_svc = _get_rag_service(request)
            if payload.github_url:
                derived_name = (
                    payload.github_url.rstrip("/").removesuffix(".git").split("/")[-1]
                )
                repo_id = payload.repo_id or derived_name or "repo-1"
                async with cloned_github_repository(
                    payload.github_url,
                    payload.ref,
                    request.app.state.settings,
                ) as clone_path:
                    summary = await rag_svc.ingest_repository(
                        RepositorySpec(
                            repo_id=repo_id,
                            root_path=clone_path,
                            name=payload.name or repo_id,
                        ),
                        allowed_roots=(clone_path,),
                        max_files=request.app.state.settings.github_max_files,
                        max_total_bytes=request.app.state.settings.github_max_repository_bytes,
                    )
            else:
                repo_id = (
                    payload.repo_id
                    or Path(payload.repository_path or "").name
                    or "repo-1"
                )
                summary = await rag_svc.ingest_repository(
                    RepositorySpec(
                        repo_id=repo_id,
                        root_path=Path(payload.repository_path or ""),
                        name=payload.name or repo_id,
                    )
                )
        except (RepositoryAccessError, RemoteRepositoryError, IngestionError) as error:
            if principal is not None and newly_claimed:
                request.app.state.repository_ownership.release_if_new(
                    repo_id, principal.organization_id
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except (
            EmbeddingConfigurationError,
            VectorStoreConfigurationError,
            LLMConfigurationError,
        ) as error:
            if principal is not None and newly_claimed:
                request.app.state.repository_ownership.release_if_new(
                    repo_id, principal.organization_id
                )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (
            EmbeddingProviderError,
            VectorStoreError,
            IngestionError,
            LLMProviderError,
        ) as error:
            if principal is not None and newly_claimed:
                request.app.state.repository_ownership.release_if_new(
                    repo_id, principal.organization_id
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Repository ingestion failed: {error}",
            ) from error

        return IngestResponse(
            repo_id=summary.repo_id,
            files_scanned=summary.files_scanned,
            files_parsed=summary.files_parsed,
            chunks_created=summary.chunks_created,
            total_tokens=summary.total_tokens,
            status="success",
        )

    @app.post(
        "/v1/repositories/ingest/async",
        response_model=AsyncIngestResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def ingest_repository_async(
        payload: IngestRequest, request: Request
    ) -> AsyncIngestResponse:
        """Enqueue an asynchronous repository ingestion job to be processed by background workers."""
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        repo_id = (
            payload.repo_id
            or (
                payload.github_url.rstrip("/").removesuffix(".git").split("/")[-1]
                if payload.github_url
                else Path(payload.repository_path or "").name
            )
            or "repo-1"
        )
        if principal is not None:
            try:
                request.app.state.repository_ownership.claim_for_ingestion(
                    repo_id, principal.organization_id
                )
            except AuthorizationError as error:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail=str(error)
                ) from error

        job_payload = {
            "github_url": payload.github_url,
            "repository_path": payload.repository_path,
            "ref": payload.ref,
            "repo_id": repo_id,
            "name": payload.name or repo_id,
        }
        org_id = principal.organization_id if principal else "anonymous"
        job = request.app.state.ingestion_job_store.enqueue(org_id, job_payload)

        return AsyncIngestResponse(
            job_id=job.id,
            repo_id=repo_id,
            status=job.status,
            created_at=job.created_at,
        )

    @app.get(
        "/v1/ingestion-jobs/{job_id}",
        response_model=IngestionJobStatusResponse,
    )
    async def get_ingestion_job(
        job_id: str, request: Request
    ) -> IngestionJobStatusResponse:
        """Poll the execution progress, attempts, and result metrics of an asynchronous ingestion job."""
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        org_id = principal.organization_id if principal else "anonymous"
        job = request.app.state.ingestion_job_store.get(job_id, org_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ingestion job '{job_id}' not found.",
            )

        return IngestionJobStatusResponse(
            job_id=job.id,
            organization_id=job.organization_id,
            status=job.status,
            attempts=job.attempts,
            error=job.error,
            result=job.result,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )

    @app.post("/v1/rag/query", response_model=RAGQueryResponse)
    async def rag_query(
        payload: RAGQueryRequest, request: Request
    ) -> RAGQueryResponse:
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        if principal is not None and not payload.repo_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="repo_id is required for tenant-scoped queries.",
            )
        if payload.repo_id:
            require_repository_owner(request, payload.repo_id, principal)
        strat_str = payload.retrieval_strategy.lower()
        if strat_str not in ("dense", "bm25", "hybrid"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="retrieval_strategy must be 'dense', 'bm25', or 'hybrid'.",
            )
        strategy = RetrievalStrategy(strat_str)

        metadata_filter = MetadataFilter(
            repo_id=payload.repo_id,
            languages=tuple(payload.languages) if payload.languages else (),
            path_patterns=tuple(payload.path_patterns)
            if payload.path_patterns
            else (),
            symbol_only=payload.symbol_only,
        )

        try:
            rag_svc = _get_rag_service(request)
            response = await rag_svc.answer_query(
                query=payload.query,
                strategy=strategy,
                filter=metadata_filter,
                top_k=payload.top_k,
            )
        except LLMQuotaExhaustedError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": str(int(error.retry_after or 5))},
            ) from error
        except LLMTimeoutError as error:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(error),
            ) from error
        except (
            LLMConfigurationError,
            EmbeddingConfigurationError,
            VectorStoreConfigurationError,
        ) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (
            LLMProviderError,
            EmbeddingProviderError,
            VectorStoreError,
            RAGError,
        ) as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not generate grounded answer from retrieved context: {error}",
            ) from error

        citations_data = [
            CitationModel(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                symbol_name=c.symbol_name,
                snippet=c.snippet,
                is_verified=c.is_verified,
            )
            for c in response.citations
        ]

        chunks_data = []
        for r in response.retrieved_chunks:
            dbg = None
            if payload.include_debug_info and r.debug_info:
                dbg = RetrievalDebugInfoModel(
                    dense_rank=r.debug_info.dense_rank,
                    dense_score=r.debug_info.dense_score,
                    bm25_rank=r.debug_info.bm25_rank,
                    bm25_score=r.debug_info.bm25_score,
                    rerank_boost=r.debug_info.rerank_boost,
                    fusion_score=r.debug_info.fusion_score,
                )
            chunks_data.append(
                RetrievedChunkModel(
                    chunk_id=r.chunk.chunk_id,
                    file_path=r.chunk.file_path,
                    start_line=r.chunk.start_line,
                    end_line=r.chunk.end_line,
                    symbol_name=r.chunk.symbol_name,
                    language=r.chunk.language,
                    content=r.chunk.content,
                    score=round(r.score, 4),
                    debug_info=dbg,
                )
            )

        return RAGQueryResponse(
            query=response.query,
            answer=response.answer,
            citations=citations_data,
            retrieved_chunks=chunks_data,
            model=response.model,
            provider=response.provider,
            strategy_used=response.strategy_used,
            retrieval_ms=response.retrieval_ms,
            generation_ms=response.generation_ms,
            cache_hit=response.cache_hit,
        )

    @app.post("/v1/agent/run", response_model=AgentRunResponse)
    async def agent_run(
        payload: AgentRunRequest, request: Request
    ) -> AgentRunResponse:
        """Run the LangGraph software engineering agent with durable checkpointing and run persistence."""
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        if principal is not None and not payload.repo_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="repo_id is required for tenant-scoped agent runs.",
            )
        if payload.repo_id:
            require_repository_owner(request, payload.repo_id, principal)

        org_id = principal.organization_id if principal else "anonymous"

        # Check distributed idempotency if key header provided
        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            allowed, cached = (
                await request.app.state.coordination_service.check_or_reserve_idempotency(
                    idempotency_key=idempotency_key,
                    organization_id=org_id,
                    ttl_seconds=request.app.state.settings.idempotency_ttl_seconds,
                )
            )
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A request with this Idempotency-Key is currently being processed.",
                )
            if cached is not None:
                return AgentRunResponse(**cached["body"])

        # Acquire tenant-bounded concurrency lease
        lease_id = (
            await request.app.state.coordination_service.acquire_concurrency_lease(
                identifier=org_id,
                max_concurrent=request.app.state.settings.tenant_max_concurrent_runs,
            )
        )
        if lease_id is None:
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Tenant concurrency limit exceeded for active agent runs.",
            )

        try:
            llm_client = _get_llm_client(request)
        except LLMConfigurationError as error:
            await request.app.state.coordination_service.release_concurrency_lease(
                org_id, lease_id
            )
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error

        rag_svc = _get_rag_service(request)
        if payload.repository_path:
            try:
                repo_root = validate_repository_path(
                    payload.repository_path,
                    allowed_roots=request.app.state.settings.allowed_repository_roots,
                )
            except (RepositoryAccessError, PathSecurityError) as error:
                await request.app.state.coordination_service.release_concurrency_lease(
                    org_id, lease_id
                )
                if idempotency_key:
                    await request.app.state.coordination_service.release_idempotency(
                        idempotency_key, org_id
                    )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid repository path: {error}",
                ) from error

            eng_context = EngineeringToolContext(
                repo_root=repo_root,
                allowed_roots=request.app.state.settings.allowed_repository_roots,
                default_timeout_seconds=request.app.state.settings.tool_timeout_seconds,
                max_output_chars=request.app.state.settings.tool_max_output_chars,
                rag_service=rag_svc,
            )
            tool_registry = create_engineering_tool_registry(eng_context)
        else:
            tool_registry = create_default_tool_registry(
                rag_svc, repo_id=payload.repo_id
            )

        agent = LangGraphAgent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            settings=request.app.state.settings,
            max_steps=payload.max_steps,
            checkpointer=request.app.state.checkpointer,
        )

        run_id = str(uuid4())
        try:
            result = await agent.run(payload.task, thread_id=run_id)
        except LLMQuotaExhaustedError as error:
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": str(int(error.retry_after or 5))},
            ) from error
        except LLMTimeoutError as error:
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(error),
            ) from error
        except LLMConfigurationError as error:
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except LLMProviderError as error:
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The configured LLM provider could not complete the agent run.",
            ) from error
        except AgentError as error:
            if idempotency_key:
                await request.app.state.coordination_service.release_idempotency(
                    idempotency_key, org_id
                )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        finally:
            await request.app.state.coordination_service.release_concurrency_lease(
                org_id, lease_id
            )

        # Build trace from AgentStep records
        steps_data: list[AgentStepModel] = []
        for step in result.steps:
            tc = step.tool_call
            obs = step.observation
            status_val = (
                step.status.value
                if hasattr(step.status, "value")
                else str(step.status)
            )
            steps_data.append(
                AgentStepModel(
                    step_number=step.step_number,
                    status=status_val,
                    reasoning=step.reasoning,
                    tool_name=tc.tool_name if tc else None,
                    tool_arguments=dict(tc.arguments) if tc else None,
                    observation=obs.content if obs else None,
                    is_error=obs.is_error if obs else False,
                )
            )

        citations_data = [
            CitationModel(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                symbol_name=c.symbol_name,
                snippet=c.snippet,
                is_verified=c.is_verified,
            )
            for c in result.citations
        ]

        response_payload = AgentRunResponse(
            request_id=run_id,
            task=result.task,
            answer=result.answer,
            status=result.status,
            total_steps=result.total_steps,
            duration_ms=round(result.duration_ms, 2),
            steps=steps_data,
            citations=citations_data,
            orchestration_engine="langgraph",
        )

        # Durable run persistence
        run_record = AgentRunRecord(
            id=run_id,
            organization_id=org_id,
            repo_id=payload.repo_id,
            task=result.task,
            status=result.status,
            total_steps=result.total_steps,
            duration_ms=round(result.duration_ms, 2),
            answer=result.answer,
            citations=[c.model_dump() for c in citations_data],
            steps=[s.model_dump() for s in steps_data],
        )
        request.app.state.agent_run_store.save_run(run_record)

        # Cache idempotent response if requested
        if idempotency_key:
            await request.app.state.coordination_service.save_idempotent_response(
                idempotency_key=idempotency_key,
                organization_id=org_id,
                response=response_payload.model_dump(),
                status_code=200,
                ttl_seconds=request.app.state.settings.idempotency_ttl_seconds,
            )

        return response_payload

    @app.get("/v1/agent/runs/{run_id}", response_model=AgentRunDetailResponse)
    async def get_agent_run(
        run_id: str, request: Request
    ) -> AgentRunDetailResponse:
        """Retrieve audit history and tool trace for a past agent run."""
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        org_id = principal.organization_id if principal else "anonymous"
        record = request.app.state.agent_run_store.get_run(run_id, org_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent run '{run_id}' not found.",
            )

        citations_data = [
            CitationModel(**c) if isinstance(c, dict) else c
            for c in (record.citations or [])
        ]
        steps_data = [
            AgentStepModel(**s) if isinstance(s, dict) else s
            for s in (record.steps or [])
        ]

        return AgentRunDetailResponse(
            run_id=record.id,
            organization_id=record.organization_id,
            repo_id=record.repo_id,
            task=record.task,
            status=record.status,
            total_steps=record.total_steps,
            duration_ms=record.duration_ms,
            answer=record.answer,
            citations=citations_data,
            steps=steps_data,
            created_at=record.created_at,
            completed_at=record.completed_at,
        )

    @app.get("/v1/agent/runs", response_model=list[AgentRunDetailResponse])
    async def list_agent_runs(
        request: Request, limit: int = 20
    ) -> list[AgentRunDetailResponse]:
        """List historical agent runs for the authenticated organization."""
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        org_id = principal.organization_id if principal else "anonymous"
        records = request.app.state.agent_run_store.list_runs(org_id, limit=limit)
        results = []
        for r in records:
            citations_data = [
                CitationModel(**c) if isinstance(c, dict) else c
                for c in (r.citations or [])
            ]
            steps_data = [
                AgentStepModel(**s) if isinstance(s, dict) else s
                for s in (r.steps or [])
            ]
            results.append(
                AgentRunDetailResponse(
                    run_id=r.id,
                    organization_id=r.organization_id,
                    repo_id=r.repo_id,
                    task=r.task,
                    status=r.status,
                    total_steps=r.total_steps,
                    duration_ms=r.duration_ms,
                    answer=r.answer,
                    citations=citations_data,
                    steps=steps_data,
                    created_at=r.created_at,
                    completed_at=r.completed_at,
                )
            )
        return results

    @app.post("/v1/mcp")
    async def handle_mcp_request(
        request: Request,
        repository_path: str | None = None,
        repo_id: str | None = None,
        read_only: bool = False,
    ):
        """Standard JSON-RPC 2.0 endpoint for Model Context Protocol (MCP) clients."""
        principal = require_principal(request)
        await apply_rate_limit(request, principal)

        try:
            raw_body = await request.json()
        except Exception as exc:
            return JSONRPCResponse(
                id=None,
                error=JSONRPCError(code=PARSE_ERROR, message=f"Parse error: {exc}"),
            ).to_dict()

        settings: Settings = request.app.state.settings
        repo_path_str = repository_path or request.headers.get("X-Repository-Path")
        repo_id_str = repo_id or request.headers.get("X-Repo-ID") or "default"
        if principal is not None:
            require_repository_owner(request, repo_id_str, principal)
        is_read_only = read_only or (
            request.headers.get("X-Read-Only", "").lower() == "true"
        )

        tool_context = None
        if repo_path_str:
            try:
                safe_repo_root = validate_repository_path(
                    repo_path_str, settings.allowed_repository_roots
                )
                rag_service = _get_rag_service(request)
                tool_context = EngineeringToolContext(
                    repo_root=safe_repo_root,
                    allowed_roots=settings.allowed_repository_roots,
                    default_timeout_seconds=float(settings.agent_step_timeout_seconds),
                    rag_service=rag_service,
                )
            except (RepositoryAccessError, IngestionError, PathSecurityError) as err:
                req_id = raw_body.get("id") if isinstance(raw_body, dict) else None
                return JSONRPCResponse(
                    id=req_id,
                    error=JSONRPCError(
                        code=INVALID_PARAMS,
                        message=f"Invalid repository context: {err}",
                    ),
                ).to_dict()

        server = MCPServer(
            tool_context=tool_context,
            repo_id=repo_id_str,
            read_only=is_read_only,
        )

        response = await server.handle_request(raw_body)
        if response is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        return response

    return app


# ---------------------------------------------------------------------------
# Internal Dependency Resolvers
# ---------------------------------------------------------------------------


def _get_llm_client(request: Request) -> LLMClient:
    client = request.app.state.llm_client
    if client is None:
        raw_client = request.app.state.client_factory(request.app.state.settings)
        client = create_admission_controlled_client(
            raw_client, request.app.state.settings
        )
        request.app.state.llm_client = client
    return client


def _get_embedding_client(request: Request) -> EmbeddingClient:
    client = request.app.state.embedding_client
    if client is None:
        client = request.app.state.embedding_factory(request.app.state.settings)
        request.app.state.embedding_client = client
    return client


def _get_vector_store(request: Request) -> VectorStore:
    store = request.app.state.vector_store
    if store is None:
        store = request.app.state.vector_store_factory(request.app.state.settings)
        request.app.state.vector_store = store
    return store


def _get_rag_service(request: Request) -> RAGService:
    service = request.app.state.rag_service
    if service is None:
        try:
            llm_client = _get_llm_client(request)
        except LLMConfigurationError:
            llm_client = None
        embedding_client = _get_embedding_client(request)
        vector_store = _get_vector_store(request)
        service = RAGService(
            vector_store=vector_store,
            embedding_client=embedding_client,
            llm_client=llm_client,
            coordination=request.app.state.coordination_service,
            settings=request.app.state.settings,
        )
        request.app.state.rag_service = service
    return service


app = create_app()