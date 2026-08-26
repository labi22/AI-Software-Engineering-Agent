"""FastAPI application for the AI software engineering agent."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import Settings
from .embeddings import (
    EmbeddingClient,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    create_embedding_client,
)
from .ingestion import IngestionError, RepositoryAccessError
from .llm import LLMClient, LLMConfigurationError, LLMProviderError, LLMRequest, create_llm_client
from .models import RepositorySpec
from .rag import RAGError, RAGService
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
    """Source citation identifying file and line numbers."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    snippet: str | None = None


class RetrievedChunkModel(BaseModel):
    """Retrieved context snippet with score."""

    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    language: str
    content: str
    score: float


class RAGQueryRequest(BaseModel):
    """Request payload to ask a grounded question against an ingested repository."""

    query: str = Field(min_length=1, max_length=10_000)
    repo_id: str | None = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGQueryResponse(BaseModel):
    """Grounded answer with citations and retrieved context."""

    query: str
    answer: str
    citations: list[CitationModel]
    retrieved_chunks: list[RetrievedChunkModel]
    model: str
    provider: str


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
    app = FastAPI(title="AI Software Engineering Agent", version="0.2.0")
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
        rag_svc = _get_rag_service(request)
        repo_id = payload.repo_id or Path(payload.repository_path).name or "repo-1"
        spec = RepositorySpec(
            repo_id=repo_id,
            root_path=Path(payload.repository_path),
            name=payload.name or repo_id,
        )

        try:
            summary = await rag_svc.ingest_repository(spec)
        except RepositoryAccessError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        except (EmbeddingConfigurationError, VectorStoreConfigurationError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except (EmbeddingProviderError, VectorStoreError, IngestionError) as error:
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
        rag_svc = _get_rag_service(request)

        try:
            response = await rag_svc.answer_query(
                query=payload.query,
                repo_id=payload.repo_id,
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
                detail="Could not generate grounded answer from retrieved context.",
            ) from error

        citations_data = [
            CitationModel(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                symbol_name=c.symbol_name,
                snippet=c.snippet,
            )
            for c in response.citations
        ]

        chunks_data = [
            RetrievedChunkModel(
                chunk_id=r.chunk.chunk_id,
                file_path=r.chunk.file_path,
                start_line=r.chunk.start_line,
                end_line=r.chunk.end_line,
                symbol_name=r.chunk.symbol_name,
                language=r.chunk.language,
                content=r.chunk.content,
                score=round(r.score, 4),
            )
            for r in response.retrieved_chunks
        ]

        return RAGQueryResponse(
            query=response.query,
            answer=response.answer,
            citations=citations_data,
            retrieved_chunks=chunks_data,
            model=response.model,
            provider=response.provider,
        )

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
        llm_client = _get_llm_client(request)
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
