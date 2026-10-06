"""Durable persistence for agent execution runs and step audit traces."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from threading import Lock
from typing import Any, Protocol, runtime_checkable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AgentRunRecord:
    """Historical audit record for an executed agent run."""

    id: str
    organization_id: str
    task: str
    status: str
    total_steps: int
    duration_ms: float
    answer: str
    repo_id: str | None = None
    citations: list[dict[str, Any]] | None = None
    steps: list[dict[str, Any]] | None = None
    created_at: str | None = None
    completed_at: str | None = None


@runtime_checkable
class AgentRunStore(Protocol):
    """Protocol for recording and querying historical agent execution runs."""

    def save_run(self, record: AgentRunRecord) -> None:
        ...

    def get_run(self, run_id: str, organization_id: str) -> AgentRunRecord | None:
        ...

    def list_runs(self, organization_id: str, limit: int = 50) -> list[AgentRunRecord]:
        ...


class InMemoryAgentRunStore:
    """Thread-safe in-memory store of agent execution histories."""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRunRecord] = {}
        self._lock = Lock()

    def save_run(self, record: AgentRunRecord) -> None:
        with self._lock:
            self._runs[record.id] = record

    def get_run(self, run_id: str, organization_id: str) -> AgentRunRecord | None:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None or record.organization_id != organization_id:
                return None
            return record

    def list_runs(self, organization_id: str, limit: int = 50) -> list[AgentRunRecord]:
        with self._lock:
            matched = [
                r for r in self._runs.values()
                if r.organization_id == organization_id
            ]
            matched.sort(key=lambda r: r.created_at or "", reverse=True)
            return matched[:limit]


class PostgresAgentRunStore:
    """PostgreSQL-backed durable store for agent runs and audit trails."""

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
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id UUID PRIMARY KEY,
                    organization_id TEXT NOT NULL,
                    repo_id TEXT,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_steps INTEGER NOT NULL,
                    duration_ms DOUBLE PRECISION NOT NULL,
                    answer TEXT NOT NULL,
                    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    steps JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMPTZ
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_runs_org_created ON agent_runs(organization_id, created_at DESC);"
            )
        self._initialized = True

    def save_run(self, record: AgentRunRecord) -> None:
        self.initialize()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_runs (
                    id, organization_id, repo_id, task, status, total_steps,
                    duration_ms, answer, citations, steps, completed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, CURRENT_TIMESTAMP
                )
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    total_steps = EXCLUDED.total_steps,
                    duration_ms = EXCLUDED.duration_ms,
                    answer = EXCLUDED.answer,
                    citations = EXCLUDED.citations,
                    steps = EXCLUDED.steps,
                    completed_at = CURRENT_TIMESTAMP;
                """,
                (
                    record.id,
                    record.organization_id,
                    record.repo_id,
                    record.task,
                    record.status,
                    record.total_steps,
                    record.duration_ms,
                    record.answer,
                    json.dumps(record.citations or []),
                    json.dumps(record.steps or []),
                ),
            )

    def get_run(self, run_id: str, organization_id: str) -> AgentRunRecord | None:
        self.initialize()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, organization_id, repo_id, task, status, total_steps,
                       duration_ms, answer, citations, steps, created_at::text, completed_at::text
                FROM agent_runs
                WHERE id = %s AND organization_id = %s;
                """,
                (run_id, organization_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        return AgentRunRecord(
            id=row[0],
            organization_id=row[1],
            repo_id=row[2],
            task=row[3],
            status=row[4],
            total_steps=row[5],
            duration_ms=row[6],
            answer=row[7],
            citations=row[8] if isinstance(row[8], list) else json.loads(row[8]),
            steps=row[9] if isinstance(row[9], list) else json.loads(row[9]),
            created_at=row[10],
            completed_at=row[11],
        )

    def list_runs(self, organization_id: str, limit: int = 50) -> list[AgentRunRecord]:
        self.initialize()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, organization_id, repo_id, task, status, total_steps,
                       duration_ms, answer, citations, steps, created_at::text, completed_at::text
                FROM agent_runs
                WHERE organization_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (organization_id, limit),
            )
            rows = cur.fetchall()
        return [
            AgentRunRecord(
                id=r[0],
                organization_id=r[1],
                repo_id=r[2],
                task=r[3],
                status=r[4],
                total_steps=r[5],
                duration_ms=r[6],
                answer=r[7],
                citations=r[8] if isinstance(r[8], list) else json.loads(r[8]),
                steps=r[9] if isinstance(r[9], list) else json.loads(r[9]),
                created_at=r[10],
                completed_at=r[11],
            )
            for r in rows
        ]
