"""Domain models for repository ingestion, code chunking, hybrid retrieval, and semantic RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RetrievalStrategy(str, Enum):
    """Retrieval strategies supported by the search pipeline."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class MetadataFilter:
    """Filter criteria applied during dense, lexical, or hybrid search."""

    repo_id: str | None = None
    languages: tuple[str, ...] = ()
    path_patterns: tuple[str, ...] = ()
    symbol_only: bool = False


@dataclass(frozen=True)
class RepositorySpec:
    """Specification for an external or local repository to ingest."""

    repo_id: str
    root_path: Path
    name: str | None = None


@dataclass(frozen=True)
class FileDocument:
    """A scanned and parsed file from a repository."""

    file_path: Path
    relative_path: str
    language: str
    content: str
    line_count: int
    size_bytes: int


@dataclass(frozen=True)
class CodeChunk:
    """A granular chunk of source code or documentation with location metadata."""

    chunk_id: str
    repo_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    language: str
    content: str
    token_count: int = 0


@dataclass(frozen=True)
class RetrievalDebugInfo:
    """Debug metrics showing individual strategy ranks and fusion scores."""

    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rerank_boost: float = 0.0
    fusion_score: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved code chunk paired with its relevance score and optional debug info."""

    chunk: CodeChunk
    score: float
    debug_info: RetrievalDebugInfo | None = None


@dataclass(frozen=True)
class Citation:
    """Citation identifying exact file and line range used to ground an answer."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    snippet: str | None = None
    is_verified: bool = True

    def formatted(self) -> str:
        symbol_info = f" ({self.symbol_name})" if self.symbol_name else ""
        return f"[{self.file_path}:{self.start_line}-{self.end_line}{symbol_info}]"


@dataclass(frozen=True)
class IngestionSummary:
    """Summary metrics of a repository ingestion run."""

    repo_id: str
    files_scanned: int
    files_parsed: int
    chunks_created: int
    total_tokens: int


@dataclass(frozen=True)
class RAGResponse:
    """Complete grounded RAG answer with source citations and retrieval details."""

    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    strategy_used: str = "hybrid"
