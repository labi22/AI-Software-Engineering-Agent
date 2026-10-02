"""Durable Postgres and In-Memory ingestion-job queue primitives."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from threading import Lock
from typing import Any, Protocol, runtime_checkable
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionJob:
    """Represents an asynchronous repository ingestion task."""

    id: str
    organization_id: str
    payload: dict[str, Any]
    status: str
    attempts: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@runtime_checkable
class IngestionJobStore(Protocol):
    """Protocol for durable or ephemeral ingestion job queue stores."""

    def enqueue(self, organization_id: str, payload: dict[str, Any]) -> IngestionJob:
        ...

    def get(self, job_id: str, organization_id: str) -> IngestionJob | None:
        ...

    def claim_next(self) -> IngestionJob | None:
        ...

    def finish(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        ...


class InMemoryIngestionJobStore:
    """Thread-safe in-memory queue store for local development and test suites."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def enqueue(self, organization_id: str, payload: dict[str, Any]) -> IngestionJob:
        with self._lock:
            job_id = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()
            job_data = {
                "id": job_id,
                "organization_id": organization_id,
                "payload": payload,
                "status": "queued",
                "attempts": 0,
                "error": None,
                "result": None,
                "created_at": now_iso,
                "started_at": None,
                "completed_at": None,
            }
            self._jobs[job_id] = job_data
            return IngestionJob(**job_data)

    def get(self, job_id: str, organization_id: str) -> IngestionJob | None:
        with self._lock:
            job_data = self._jobs.get(job_id)
            if job_data is None or job_data["organization_id"] != organization_id:
                return None
            return IngestionJob(**job_data)

    def claim_next(self) -> IngestionJob | None:
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            for job_data in self._jobs.values():
                if job_data["status"] == "queued":
                    job_data["status"] = "running"
                    job_data["attempts"] += 1
                    job_data["started_at"] = now_iso
                    return IngestionJob(**job_data)
            return None

    def finish(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job_data = self._jobs.get(job_id)
            if job_data is None or job_data["status"] != "running":
                return
            now_iso = datetime.now(timezone.utc).isoformat()
            job_data["status"] = "failed" if error else "succeeded"
            job_data["error"] = error
            job_data["result"] = result
            job_data["completed_at"] = now_iso


class PostgresIngestionJobStore:
    """Queue backed by PostgreSQL with atomic ``FOR UPDATE SKIP LOCKED`` claims across workers."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._initialized = False

    def _conn(self):
        import psycopg
        return psycopg.connect(self.database_url, autocommit=True)

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id UUID PRIMARY KEY,
                    organization_id TEXT NOT NULL REFERENCES tenant_organizations(id),
                    payload JSONB NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    result JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_claim ON ingestion_jobs(status, created_at) WHERE status = 'queued';"
            )
            # Add result column if upgrading an existing table without it
            cur.execute(
                "ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS result JSONB;"
            )
        self._initialized = True

    def enqueue(self, organization_id: str, payload: dict[str, Any]) -> IngestionJob:
        self.initialize()
        job_id = str(uuid.uuid4())
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_jobs (id, organization_id, payload, status)
                VALUES (%s, %s, %s::jsonb, 'queued')
                RETURNING created_at::text;
                """,
                (job_id, organization_id, json.dumps(payload)),
            )
            row = cur.fetchone()
            created_at = row[0] if row else None
        return IngestionJob(
            id=job_id,
            organization_id=organization_id,
            payload=payload,
            status="queued",
            created_at=created_at,
        )

    def get(self, job_id: str, organization_id: str) -> IngestionJob | None:
        self.initialize()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, organization_id, payload, status, attempts, error, result,
                       created_at::text, started_at::text, completed_at::text
                FROM ingestion_jobs
                WHERE id = %s AND organization_id = %s;
                """,
                (job_id, organization_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        return IngestionJob(
            id=row[0],
            organization_id=row[1],
            payload=row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            status=row[3],
            attempts=row[4],
            error=row[5],
            result=row[6] if isinstance(row[6], dict) or row[6] is None else json.loads(row[6]),
            created_at=row[7],
            started_at=row[8],
            completed_at=row[9],
        )

    def claim_next(self) -> IngestionJob | None:
        self.initialize()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                WITH next AS (
                    SELECT id FROM ingestion_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE ingestion_jobs
                SET status = 'running',
                    attempts = attempts + 1,
                    started_at = CURRENT_TIMESTAMP
                FROM next
                WHERE ingestion_jobs.id = next.id
                RETURNING ingestion_jobs.id::text, organization_id, payload, status,
                          attempts, error, result, created_at::text, started_at::text, completed_at::text;
                """
            )
            row = cur.fetchone()
        if not row:
            return None
        return IngestionJob(
            id=row[0],
            organization_id=row[1],
            payload=row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            status=row[3],
            attempts=row[4],
            error=row[5],
            result=row[6] if isinstance(row[6], dict) or row[6] is None else json.loads(row[6]),
            created_at=row[7],
            started_at=row[8],
            completed_at=row[9],
        )

    def finish(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.initialize()
        result_json = json.dumps(result) if result is not None else None
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_jobs
                SET status = %s,
                    error = %s,
                    result = %s::jsonb,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = 'running';
                """,
                ("failed" if error else "succeeded", error, result_json, job_id),
            )
