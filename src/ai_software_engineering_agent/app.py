"""FastAPI application for the AI software engineering agent."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .agent import Agent, AgentError
from .agent_state import AgentStatus
from .config import Settings
from .embeddings import (
    EmbeddingClient,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    create_embedding_client,
)
from .engineering_tools import EngineeringToolContext, create_engineering_tool_registry
from .ingestion import IngestionError, RepositoryAccessError, validate_repository_path
from .llm import LLMClient, LLMConfigurationError, LLMProviderError, LLMRequest, create_llm_client
from .models import (
    MetadataFilter,
    RepositorySpec,
    RetrievalStrategy,
)
from .mcp import INVALID_PARAMS, PARSE_ERROR, JSONRPCError, JSONRPCResponse, MCPServer
from .rag import RAGError, RAGService
from .safety import PathSecurityError
from .tools import create_default_tool_registry
from .vector_store import (
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
    create_vector_store,
)


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

    repository_path: str = Field(min_length=1, description="Absolute or relative local directory path")
    repo_id: str | None = Field(default=None, description="Optional custom repository ID")
    name: str | None = Field(default=None, description="Optional display name for the repository")


class IngestResponse(BaseModel):
    """Metrics and status resulting from a repository ingestion run."""

    repo_id: str
    files_scanned: int
    files_parsed: int
    chunks_created: int
    total_tokens: int
    status: str = "success"


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
    languages: list[str] | None = Field(default=None, description="Filter chunks by programming languages")
    path_patterns: list[str] | None = Field(default=None, description="Filter chunks by glob file path patterns")
    symbol_only: bool = Field(default=False, description="Filter chunks that contain named code symbols only")
    include_debug_info: bool = Field(default=False, description="Include rank breakdown in response")


class RAGQueryResponse(BaseModel):
    """Grounded answer with verified citations, retrieval context, and strategy metrics."""

    query: str
    answer: str
    citations: list[CitationModel]
    retrieved_chunks: list[RetrievedChunkModel]
    model: str
    provider: str
    strategy_used: str = "hybrid"


class AgentStepModel(BaseModel):
    """A single step in the agent execution trace."""

    step_number: int
    status: str
    reasoning: str | None = None
    tool_name: str | None = None
    tool_arguments: dict | None = None
    observation: str | None = None
    is_error: bool = False


class AgentRunRequest(BaseModel):
    """Request payload to run the ReAct agent on a task."""

    task: str = Field(min_length=1, max_length=10_000, description="The task or question for the agent to solve")
    repository_path: str | None = Field(default=None, description="Optional local repository path to bind safe engineering tools to")
    repo_id: str | None = Field(default=None, description="Optional repository ID")
    max_steps: int = Field(default=10, ge=1, le=30, description="Maximum number of reasoning/action steps")


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


ClientFactory = Callable[[Settings], LLMClient]
EmbeddingFactory = Callable[[Settings], EmbeddingClient]
VectorStoreFactory = Callable[[Settings], VectorStore]


def create_app(
    *,
    settings: Settings | None = None,
    client_factory: ClientFactory = create_llm_client,
    embedding_factory: EmbeddingFactory = create_embedding_client,
    vector_store_factory: VectorStoreFactory = create_vector_store,
) -> FastAPI:
    """Create the API application with injectable settings and component factories."""
    app = FastAPI(title="AI Software Engineering Agent", version="0.3.0")
    app.state.settings = settings or Settings.from_environment()
    app.state.client_factory = client_factory
    app.state.embedding_factory = embedding_factory
    app.state.vector_store_factory = vector_store_factory

    app.state.llm_client = None
    app.state.embedding_client = None
    app.state.vector_store = None
    app.state.rag_service = None

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        try:
            client = _get_llm_client(request)
            result = await client.generate(
                LLMRequest(
                    prompt=payload.prompt,
                    system_instruction=payload.system_instruction,
                )
            )
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
    async def ingest_repository(payload: IngestRequest, request: Request) -> IngestResponse:
        repo_id = payload.repo_id or Path(payload.repository_path).name or "repo-1"
        spec = RepositorySpec(
            repo_id=repo_id,
            root_path=Path(payload.repository_path),
            name=payload.name or repo_id,
        )

        try:
            rag_svc = _get_rag_service(request)
            summary = await rag_svc.ingest_repository(spec)
        except RepositoryAccessError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except (EmbeddingConfigurationError, VectorStoreConfigurationError, LLMConfigurationError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (EmbeddingProviderError, VectorStoreError, IngestionError, LLMProviderError) as error:
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

    @app.post("/v1/rag/query", response_model=RAGQueryResponse)
    async def rag_query(payload: RAGQueryRequest, request: Request) -> RAGQueryResponse:
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
            path_patterns=tuple(payload.path_patterns) if payload.path_patterns else (),
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
        except (LLMConfigurationError, EmbeddingConfigurationError, VectorStoreConfigurationError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (LLMProviderError, EmbeddingProviderError, VectorStoreError, RAGError) as error:
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
        )

    @app.post("/v1/agent/run", response_model=AgentRunResponse)
    async def agent_run(payload: AgentRunRequest, request: Request) -> AgentRunResponse:
        try:
            llm_client = _get_llm_client(request)
        except LLMConfigurationError as error:
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
            tool_registry = create_default_tool_registry(rag_svc)

        agent = Agent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            settings=request.app.state.settings,
            max_steps=payload.max_steps,
        )

        try:
            result = await agent.run(payload.task)
        except LLMConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except LLMProviderError as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The configured LLM provider could not complete the agent run.",
            ) from error
        except AgentError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

        # Build trace from AgentStep records
        steps_data: list[AgentStepModel] = []
        for step in result.steps:
            tc = step.tool_call
            obs = step.observation
            steps_data.append(
                AgentStepModel(
                    step_number=step.step_number,
                    status=step.status.value,
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

        return AgentRunResponse(
            request_id=str(uuid4()),
            task=result.task,
            answer=result.answer,
            status=result.status,
            total_steps=result.total_steps,
            duration_ms=round(result.duration_ms, 2),
            steps=steps_data,
            citations=citations_data,
        )

    @app.post("/v1/mcp")
    async def handle_mcp_request(
        request: Request,
        repository_path: str | None = None,
        repo_id: str | None = None,
        read_only: bool = False,
    ):
        """Standard JSON-RPC 2.0 endpoint for Model Context Protocol (MCP) clients."""
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
        is_read_only = read_only or (request.headers.get("X-Read-Only", "").lower() == "true")

        tool_context = None
        if repo_path_str:
            try:
                safe_repo_root = validate_repository_path(repo_path_str, settings.allowed_repository_roots)
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
                    error=JSONRPCError(code=INVALID_PARAMS, message=f"Invalid repository context: {err}"),
                ).to_dict()

        server = MCPServer(
            tool_context=tool_context,
            repo_id=repo_id_str,
            read_only=is_read_only,
        )

        response = await server.handle_request(raw_body)
        if response is None:
            from fastapi.responses import Response
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        return response

    return app


def _get_llm_client(request: Request) -> LLMClient:
    client = request.app.state.llm_client
    if client is None:
        client = request.app.state.client_factory(request.app.state.settings)
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
            settings=request.app.state.settings,
        )
        request.app.state.rag_service = service
    return service


app = create_app()
