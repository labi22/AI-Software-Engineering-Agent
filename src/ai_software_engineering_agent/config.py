"""Runtime configuration for the API service and RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    """Settings loaded from environment variables without reading secrets from files."""

    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    request_timeout_seconds: float
    allowed_repository_roots: tuple[Path, ...]
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    vector_store_type: str = "memory"
    database_url: str | None = None
    chunk_size_lines: int = 60
    chunk_overlap_lines: int = 15

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        environment = os.environ if environ is None else environ
        provider = environment.get("LLM_PROVIDER", "openai").strip().lower()
        if provider != "openai":
            raise ValueError("LLM_PROVIDER must be 'openai'.")

        try:
            timeout = float(environment.get("LLM_REQUEST_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be a number.") from error
        if timeout <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be greater than zero.")

        roots = tuple(
            Path(value.strip())
            for value in environment.get("ALLOWED_REPOSITORY_ROOTS", "").split(",")
            if value.strip()
        )

        embedding_provider = environment.get("EMBEDDING_PROVIDER", "openai").strip().lower()
        embedding_model = environment.get("EMBEDDING_MODEL", "text-embedding-3-small").strip()
        
        try:
            embedding_dimension = int(environment.get("EMBEDDING_DIMENSION", "1536"))
        except ValueError as error:
            raise ValueError("EMBEDDING_DIMENSION must be an integer.") from error

        db_url = environment.get("DATABASE_URL") or None
        vector_store_type = environment.get("VECTOR_STORE_TYPE", "pgvector" if db_url else "memory").strip().lower()
        if vector_store_type not in ("memory", "pgvector"):
            raise ValueError("VECTOR_STORE_TYPE must be 'memory' or 'pgvector'.")

        try:
            chunk_size = int(environment.get("CHUNK_SIZE_LINES", "60"))
            chunk_overlap = int(environment.get("CHUNK_OVERLAP_LINES", "15"))
        except ValueError as error:
            raise ValueError("CHUNK_SIZE_LINES and CHUNK_OVERLAP_LINES must be integers.") from error

        return cls(
            llm_provider=provider,
            llm_model=environment.get("LLM_MODEL", "gpt-5.2").strip(),
            openai_api_key=environment.get("OPENAI_API_KEY") or None,
            request_timeout_seconds=timeout,
            allowed_repository_roots=roots,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            vector_store_type=vector_store_type,
            database_url=db_url,
            chunk_size_lines=chunk_size,
            chunk_overlap_lines=chunk_overlap,
        )
