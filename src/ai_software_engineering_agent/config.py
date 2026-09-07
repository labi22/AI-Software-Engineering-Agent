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
    agent_max_steps: int = 10
    agent_step_timeout_seconds: float = 30.0
    tool_timeout_seconds: float = 30.0
    tool_max_output_chars: int = 8000

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        environment = os.environ if environ is None else environ
        provider = environment.get("LLM_PROVIDER", "openai").strip().lower()
        if provider not in ("openai", "fake"):
            raise ValueError("LLM_PROVIDER must be 'openai' or 'fake'.")

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
        if embedding_provider not in ("openai", "fake"):
            raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'fake'.")
            
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

        try:
            agent_max_steps = int(environment.get("AGENT_MAX_STEPS", "10"))
        except ValueError as error:
            raise ValueError("AGENT_MAX_STEPS must be an integer.") from error

        try:
            agent_step_timeout = float(environment.get("AGENT_STEP_TIMEOUT_SECONDS", "30.0"))
        except ValueError as error:
            raise ValueError("AGENT_STEP_TIMEOUT_SECONDS must be a number.") from error

        try:
            tool_timeout = float(environment.get("TOOL_TIMEOUT_SECONDS", "30.0"))
        except ValueError as error:
            raise ValueError("TOOL_TIMEOUT_SECONDS must be a number.") from error

        try:
            tool_max_output = int(environment.get("TOOL_MAX_OUTPUT_CHARS", "8000"))
        except ValueError as error:
            raise ValueError("TOOL_MAX_OUTPUT_CHARS must be an integer.") from error

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
            agent_max_steps=agent_max_steps,
            agent_step_timeout_seconds=agent_step_timeout,
            tool_timeout_seconds=tool_timeout,
            tool_max_output_chars=tool_max_output,
        )
