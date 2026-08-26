"""Vector store abstraction, in-memory implementation, and PostgreSQL + pgvector storage."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Protocol, Sequence

from .config import Settings
from .models import CodeChunk, RetrievalResult


class VectorStoreError(RuntimeError):
    """Base error for vector store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when the vector store is misconfigured or cannot connect."""


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class VectorStore(Protocol):
    """Protocol for persisting code chunks and performing vector similarity search."""

    async def initialize(self) -> None:
        """Initialize required schemas, tables, or indexes."""

    async def store_chunks(
        self,
        chunks: Sequence[CodeChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Store code chunks paired with their embedding vectors."""

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-K most similar code chunks."""

    async def delete_repository(self, repo_id: str) -> None:
        """Delete all stored chunks for a repository."""

    async def count_chunks(self, repo_id: str | None = None) -> int:
        """Return the number of stored chunks."""


class InMemoryVectorStore:
    """Thread-safe in-memory vector store for unit tests and local runs without database setup."""

    def __init__(self) -> None:
        self._chunks: dict[str, CodeChunk] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def store_chunks(
        self,
        chunks: Sequence[CodeChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings length mismatch.")

        async with self._lock:
            for chunk, emb in zip(chunks, embeddings):
                self._chunks[chunk.chunk_id] = chunk
                self._embeddings[chunk.chunk_id] = emb

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        async with self._lock:
            candidates: list[tuple[CodeChunk, float]] = []
            for chunk_id, chunk in self._chunks.items():
                if repo_id and chunk.repo_id != repo_id:
                    continue
                emb = self._embeddings.get(chunk_id)
                if emb is None:
                    continue
                score = _cosine_similarity(query_embedding, emb)
                candidates.append((chunk, score))

            # Sort descending by cosine similarity
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [
                RetrievalResult(chunk=chunk, score=score)
                for chunk, score in candidates[:limit]
            ]

    async def delete_repository(self, repo_id: str) -> None:
        async with self._lock:
            to_delete = [cid for cid, c in self._chunks.items() if c.repo_id == repo_id]
            for cid in to_delete:
                self._chunks.pop(cid, None)
                self._embeddings.pop(cid, None)

    async def count_chunks(self, repo_id: str | None = None) -> int:
        async with self._lock:
            if repo_id is None:
                return len(self._chunks)
            return sum(1 for c in self._chunks.values() if c.repo_id == repo_id)


class PgVectorStore:
    """PostgreSQL storage adapter using pgvector extension."""

    def __init__(
        self,
        database_url: str,
        dimension: int = 1536,
    ) -> None:
        self._database_url = database_url
        self._dimension = dimension
        self._pool: Any | None = None

    def _get_connection(self) -> Any:
        try:
            import psycopg
            return psycopg.connect(self._database_url, autocommit=True)
        except ImportError as error:
            raise VectorStoreConfigurationError(
                "psycopg is required for pgvector. Install with `pip install psycopg[binary] pgvector`."
            ) from error
        except Exception as error:
            raise VectorStoreConfigurationError(
                f"Could not connect to PostgreSQL database: {error}"
            ) from error

    async def initialize(self) -> None:
        def _init_db() -> None:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS repositories (
                            id TEXT PRIMARY KEY,
                            name TEXT,
                            root_path TEXT,
                            ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            chunk_count INTEGER DEFAULT 0
                        );
                        """
                    )
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS code_chunks (
                            id TEXT PRIMARY KEY,
                            repo_id TEXT NOT NULL,
                            file_path TEXT NOT NULL,
                            start_line INTEGER NOT NULL,
                            end_line INTEGER NOT NULL,
                            symbol_name TEXT,
                            language TEXT NOT NULL,
                            content TEXT NOT NULL,
                            token_count INTEGER DEFAULT 0,
                            embedding vector({self._dimension}),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_code_chunks_repo ON code_chunks(repo_id);"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_code_chunks_path ON code_chunks(file_path);"
                    )

        await asyncio.to_thread(_init_db)

    async def store_chunks(
        self,
        chunks: Sequence[CodeChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings length mismatch.")
        if not chunks:
            return

        def _insert() -> None:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk, emb in zip(chunks, embeddings):
                        # Convert float list to pgvector string format '[0.1, 0.2, ...]'
                        emb_str = "[" + ",".join(str(f) for f in emb) + "]"
                        cur.execute(
                            """
                            INSERT INTO code_chunks (
                                id, repo_id, file_path, start_line, end_line,
                                symbol_name, language, content, token_count, embedding
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                symbol_name = EXCLUDED.symbol_name,
                                token_count = EXCLUDED.token_count,
                                embedding = EXCLUDED.embedding;
                            """,
                            (
                                chunk.chunk_id,
                                chunk.repo_id,
                                chunk.file_path,
                                chunk.start_line,
                                chunk.end_line,
                                chunk.symbol_name,
                                chunk.language,
                                chunk.content,
                                chunk.token_count,
                                emb_str,
                            ),
                        )

        await asyncio.to_thread(_insert)

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        repo_id: str | None = None,
    ) -> list[RetrievalResult]:
        emb_str = "[" + ",".join(str(f) for f in query_embedding) + "]"

        def _query() -> list[RetrievalResult]:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if repo_id:
                        cur.execute(
                            """
                            SELECT id, repo_id, file_path, start_line, end_line,
                                   symbol_name, language, content, token_count,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM code_chunks
                            WHERE repo_id = %s
                            ORDER BY embedding <=> %s::vector ASC
                            LIMIT %s;
                            """,
                            (emb_str, repo_id, emb_str, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, repo_id, file_path, start_line, end_line,
                                   symbol_name, language, content, token_count,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM code_chunks
                            ORDER BY embedding <=> %s::vector ASC
                            LIMIT %s;
                            """,
                            (emb_str, emb_str, limit),
                        )
                    rows = cur.fetchall()
                    results: list[RetrievalResult] = []
                    for r in rows:
                        chunk = CodeChunk(
                            chunk_id=r[0],
                            repo_id=r[1],
                            file_path=r[2],
                            start_line=r[3],
                            end_line=r[4],
                            symbol_name=r[5],
                            language=r[6],
                            content=r[7],
                            token_count=r[8],
                        )
                        results.append(RetrievalResult(chunk=chunk, score=float(r[9])))
                    return results

        return await asyncio.to_thread(_query)

    async def delete_repository(self, repo_id: str) -> None:
        def _delete() -> None:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM code_chunks WHERE repo_id = %s;", (repo_id,))
                    cur.execute("DELETE FROM repositories WHERE id = %s;", (repo_id,))

        await asyncio.to_thread(_delete)

    async def count_chunks(self, repo_id: str | None = None) -> int:
        def _count() -> int:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if repo_id:
                        cur.execute("SELECT COUNT(*) FROM code_chunks WHERE repo_id = %s;", (repo_id,))
                    else:
                        cur.execute("SELECT COUNT(*) FROM code_chunks;")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0

        return await asyncio.to_thread(_count)


def create_vector_store(settings: Settings) -> VectorStore:
    """Factory creating the configured vector store."""
    if settings.vector_store_type == "memory":
        return InMemoryVectorStore()

    if settings.vector_store_type == "pgvector":
        if not settings.database_url:
            raise VectorStoreConfigurationError(
                "DATABASE_URL must be set when VECTOR_STORE_TYPE=pgvector."
            )
        return PgVectorStore(
            database_url=settings.database_url,
            dimension=settings.embedding_dimension,
        )

    raise VectorStoreConfigurationError(
        f"Unsupported vector store type: {settings.vector_store_type}."
    )
