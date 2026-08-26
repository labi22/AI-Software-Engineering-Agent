"""AI Software Engineering Agent package."""

from .agent import AgentPlan, build_plan, extract_keywords
from .app import create_app
from .config import Settings
from .embeddings import EmbeddingClient, FakeEmbeddingClient, OpenAIEmbeddingClient, create_embedding_client
from .ingestion import chunk_document, discover_files, parse_file, validate_repository_path
from .llm import LLMClient, LLMRequest, LLMResponse, OpenAIResponsesClient, create_llm_client
from .models import (
    Citation,
    CodeChunk,
    FileDocument,
    IngestionSummary,
    RAGResponse,
    RepositorySpec,
    RetrievalResult,
)
from .rag import RAGService, extract_citations, format_context_prompt
from .vector_store import InMemoryVectorStore, PgVectorStore, VectorStore, create_vector_store

__all__ = [
    "AgentPlan",
    "Citation",
    "CodeChunk",
    "EmbeddingClient",
    "FakeEmbeddingClient",
    "FileDocument",
    "InMemoryVectorStore",
    "IngestionSummary",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "OpenAIEmbeddingClient",
    "OpenAIResponsesClient",
    "PgVectorStore",
    "RAGResponse",
    "RAGService",
    "RepositorySpec",
    "RetrievalResult",
    "Settings",
    "VectorStore",
    "build_plan",
    "chunk_document",
    "create_app",
    "create_embedding_client",
    "create_llm_client",
    "create_vector_store",
    "discover_files",
    "extract_citations",
    "extract_keywords",
    "format_context_prompt",
    "parse_file",
    "validate_repository_path",
]
