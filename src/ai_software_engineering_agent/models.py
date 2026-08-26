"""Domain models for repository ingestion, code chunking, and semantic RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
class RetrievalResult:
    """A retrieved code chunk paired with its semantic relevance score."""

    chunk: CodeChunk
    score: float


@dataclass(frozen=True)
class Citation:
    """Citation identifying exact file and line range used to ground an answer."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    snippet: str | None = None

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
    """Complete grounded RAG answer with source citations."""

    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)
    model: str = ""
    provider: str = ""
