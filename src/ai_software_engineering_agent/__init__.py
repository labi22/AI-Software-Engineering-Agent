"""AI Software Engineering Agent package."""

from .agent import AgentPlan, build_plan, extract_keywords
from .app import create_app
from .config import Settings
from .embeddings import EmbeddingClient, FakeEmbeddingClient, OpenAIEmbeddingClient, create_embedding_client
from .ingestion import chunk_document, discover_files, parse_file, validate_repository_path
from .lexical import BM25Index, CodeTokenizer
from .llm import FakeLLMClient, LLMClient, LLMRequest, LLMResponse, OpenAIResponsesClient, create_llm_client
from .models import (
    Citation,
    CodeChunk,
    FileDocument,
    IngestionSummary,
    MetadataFilter,
    RAGResponse,
    RepositorySpec,
    RetrievalDebugInfo,
    RetrievalResult,
    RetrievalStrategy,
)
from .rag import RAGService, extract_citations, format_context_prompt, validate_citations
from .retrieval import HybridRetriever, QueryExpander, SymbolBoostReranker, reciprocal_rank_fusion
from .vector_store import InMemoryVectorStore, PgVectorStore, VectorStore, create_vector_store

__all__ = [
    "AgentPlan",
    "BM25Index",
    "Citation",
    "CodeChunk",
    "CodeTokenizer",
    "EmbeddingClient",
    "FakeEmbeddingClient",
    "FakeLLMClient",
    "FileDocument",
    "HybridRetriever",
    "InMemoryVectorStore",
    "IngestionSummary",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "MetadataFilter",
    "OpenAIEmbeddingClient",
    "OpenAIResponsesClient",
    "PgVectorStore",
    "QueryExpander",
    "RAGResponse",
    "RAGService",
    "RepositorySpec",
    "RetrievalDebugInfo",
    "RetrievalResult",
    "RetrievalStrategy",
    "Settings",
    "SymbolBoostReranker",
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
    "reciprocal_rank_fusion",
    "validate_citations",
    "validate_repository_path",
]
