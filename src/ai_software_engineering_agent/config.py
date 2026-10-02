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
    llm_base_url: str | None = None
    groq_api_key: str | None = None
    github_clone_timeout_seconds: float = 60.0
    github_max_repository_bytes: int = 100 * 1024 * 1024
    github_max_files: int = 10_000
    auth_enabled: bool = False
    auth_api_key_hashes: tuple[tuple[str, str, str], ...] = ()
    redis_url: str | None = None
    rate_limit_per_minute: int = 60
    tenant_max_concurrent_runs: int = 2
    llm_rpm_limit: int = 30
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 10.0
    idempotency_ttl_seconds: int = 86400

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        environment = os.environ if environ is None else environ
        provider = environment.get("LLM_PROVIDER", "openai").strip().lower()
        if provider not in ("openai", "groq", "ollama", "fake"):
            raise ValueError("LLM_PROVIDER must be 'openai', 'groq', 'ollama', or 'fake'.")

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
        if embedding_provider not in ("openai", "fake", "huggingface", "hf", "local"):
            raise ValueError("EMBEDDING_PROVIDER must be 'openai', 'fake', or 'huggingface'.")

        default_model = (
            "sentence-transformers/all-MiniLM-L6-v2"
            if embedding_provider in ("huggingface", "hf", "local")
            else "text-embedding-3-small"
        )
        embedding_model = environment.get("EMBEDDING_MODEL", default_model).strip()

        default_dim = "384" if embedding_provider in ("huggingface", "hf", "local") else "1536"
        try:
            embedding_dimension = int(environment.get("EMBEDDING_DIMENSION", default_dim))
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

        try:
            github_clone_timeout = float(environment.get("GITHUB_CLONE_TIMEOUT_SECONDS", "60"))
            github_max_repository_bytes = int(
                environment.get("GITHUB_MAX_REPOSITORY_BYTES", str(100 * 1024 * 1024))
            )
            github_max_files = int(environment.get("GITHUB_MAX_FILES", "10000"))
        except ValueError as error:
            raise ValueError(
                "GITHUB_CLONE_TIMEOUT_SECONDS, GITHUB_MAX_REPOSITORY_BYTES, and GITHUB_MAX_FILES must be numbers."
            ) from error
        if github_clone_timeout <= 0 or github_max_repository_bytes <= 0 or github_max_files <= 0:
            raise ValueError("GitHub clone limits must be greater than zero.")

        auth_enabled = environment.get("AUTH_ENABLED", "false").strip().lower() == "true"
        raw_key_config = environment.get("AUTH_API_KEY_HASHES", "").strip()
        key_hashes: list[tuple[str, str, str]] = []
        if raw_key_config:
            for item in raw_key_config.split(","):
                parts = item.strip().split(":")
                if len(parts) != 3 or not all(parts):
                    raise ValueError("AUTH_API_KEY_HASHES entries must be '<key_id>:<sha256>:<organization_id>'.")
                key_id, key_hash, organization_id = parts
                if len(key_hash) != 64 or any(char not in "0123456789abcdefABCDEF" for char in key_hash):
                    raise ValueError("AUTH_API_KEY_HASHES must contain SHA-256 hexadecimal digests.")
                key_hashes.append((key_id, key_hash.lower(), organization_id))
        if auth_enabled and not key_hashes:
            raise ValueError("AUTH_API_KEY_HASHES must be configured when AUTH_ENABLED=true.")
        if auth_enabled and not db_url:
            raise ValueError("DATABASE_URL must be configured when AUTH_ENABLED=true for durable tenant ownership.")

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
            llm_base_url=environment.get("LLM_BASE_URL", "").strip() or None,
            groq_api_key=environment.get("GROQ_API_KEY") or None,
            github_clone_timeout_seconds=github_clone_timeout,
            github_max_repository_bytes=github_max_repository_bytes,
            github_max_files=github_max_files,
            auth_enabled=auth_enabled,
            auth_api_key_hashes=tuple(key_hashes),
            redis_url=environment.get("REDIS_URL") or None,
            rate_limit_per_minute=int(environment.get("RATE_LIMIT_PER_MINUTE", "60")),
            tenant_max_concurrent_runs=int(environment.get("TENANT_MAX_CONCURRENT_RUNS", "2")),
            llm_rpm_limit=int(environment.get("LLM_RPM_LIMIT", "30")),
            llm_max_retries=int(environment.get("LLM_MAX_RETRIES", "3")),
            llm_retry_base_delay=float(environment.get("LLM_RETRY_BASE_DELAY", "1.0")),
            llm_retry_max_delay=float(environment.get("LLM_RETRY_MAX_DELAY", "10.0")),
            idempotency_ttl_seconds=int(environment.get("IDEMPOTENCY_TTL_SECONDS", "86400")),
        )
