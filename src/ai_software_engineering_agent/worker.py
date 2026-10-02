"""Background worker for asynchronous repository ingestion jobs."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .ingestion import cloned_github_repository, validate_repository_path
from .ingestion_jobs import IngestionJob, IngestionJobStore
from .models import RepositorySpec
from .rag import RAGService

logger = logging.getLogger(__name__)


async def process_ingestion_job(
    job: IngestionJob,
    rag_service: RAGService,
    settings: Settings,
) -> dict[str, Any]:
    """Execute ingestion for a single claimed job payload."""
    payload = job.payload
    github_url = payload.get("github_url")
    repository_path = payload.get("repository_path")
    ref = payload.get("ref", "HEAD")
    repo_id = payload.get("repo_id") or "repo-1"
    name = payload.get("name") or repo_id

    if github_url:
        async with cloned_github_repository(github_url, ref, settings) as clone_path:
            summary = await rag_service.ingest_repository(
                RepositorySpec(repo_id=repo_id, root_path=clone_path, name=name),
                allowed_roots=(clone_path,),
                max_files=settings.github_max_files,
                max_total_bytes=settings.github_max_repository_bytes,
            )
    elif repository_path:
        repo_root = validate_repository_path(
            repository_path,
            allowed_roots=settings.allowed_repository_roots,
        )
        summary = await rag_service.ingest_repository(
            RepositorySpec(repo_id=repo_id, root_path=repo_root, name=name),
        )
    else:
        raise ValueError("Job payload missing repository_path or github_url.")

    return {
        "repo_id": summary.repo_id,
        "files_scanned": summary.files_scanned,
        "files_parsed": summary.files_parsed,
        "chunks_created": summary.chunks_created,
        "total_tokens": summary.total_tokens,
        "status": "success",
    }


async def process_next_job(
    store: IngestionJobStore,
    rag_service: RAGService,
    settings: Settings,
) -> IngestionJob | None:
    """Claim and process the next queued ingestion job atomically.

    Returns the claimed job (after updating its state), or None if queue is empty.
    """
    job = store.claim_next()
    if job is None:
        return None

    logger.info("Claimed ingestion job %s (org=%s)", job.id, job.organization_id)
    try:
        result = await process_ingestion_job(job, rag_service, settings)
        store.finish(job.id, result=result)
        logger.info("Ingestion job %s completed successfully", job.id)
    except Exception as exc:
        logger.exception("Ingestion job %s failed: %s", job.id, exc)
        store.finish(job.id, error=str(exc))

    return job


async def run_worker_loop(
    store: IngestionJobStore,
    rag_service_factory: Callable[[], RAGService],
    settings: Settings,
    stop_event: asyncio.Event | None = None,
    poll_interval_seconds: float = 1.0,
) -> None:
    """Continuous polling loop for ingestion workers."""
    logger.info("Starting ingestion worker loop...")
    while stop_event is None or not stop_event.is_set():
        try:
            rag_service = rag_service_factory()
            claimed = await process_next_job(store, rag_service, settings)
            if claimed is None:
                await asyncio.sleep(poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Worker loop cancelled.")
            break
        except Exception as exc:
            logger.exception("Unexpected error in worker loop: %s", exc)
            await asyncio.sleep(poll_interval_seconds)


def main() -> None:
    """CLI entrypoint for running the standalone ingestion worker process."""
    from .embeddings import create_embedding_client
    from .vector_store import create_vector_store
    from .ingestion_jobs import PostgresIngestionJobStore, InMemoryIngestionJobStore

    settings = Settings.from_environment()
    store = (
        PostgresIngestionJobStore(settings.database_url)
        if settings.database_url
        else InMemoryIngestionJobStore()
    )

    def rag_factory() -> RAGService:
        vstore = create_vector_store(settings)
        eclient = create_embedding_client(settings)
        return RAGService(vector_store=vstore, embedding_client=eclient, settings=settings)

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker_loop(store, rag_factory, settings))


if __name__ == "__main__":
    main()
