"""Coordination layer providing Redis-backed or in-memory idempotency, rate limiting, concurrency leasing, and caching."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from threading import Lock
import time
from typing import Any, Protocol, runtime_checkable
import uuid

logger = logging.getLogger(__name__)


@runtime_checkable
class CoordinationService(Protocol):
    """Protocol for distributed or in-memory coordination across replicas."""

    async def check_or_reserve_idempotency(
        self,
        idempotency_key: str,
        organization_id: str,
        ttl_seconds: int = 300,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Check idempotency key.

        Returns:
            (allowed, cached_response)
            - If (False, None): Key is currently in progress (409 Conflict).
            - If (True, cached_response): Key completed; return cached payload.
            - If (True, None): Key successfully claimed; proceed with execution.
        """
        ...

    async def save_idempotent_response(
        self,
        idempotency_key: str,
        organization_id: str,
        response: dict[str, Any],
        status_code: int = 200,
        ttl_seconds: int = 86400,
    ) -> None:
        """Store completed response under the idempotency key."""
        ...

    async def release_idempotency(
        self,
        idempotency_key: str,
        organization_id: str,
    ) -> None:
        """Release reservation if execution failed before completion."""
        ...

    async def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> tuple[bool, int, float]:
        """Sliding-window rate limiter.

        Returns:
            (is_allowed, remaining_requests, retry_after_seconds)
        """
        ...

    async def acquire_concurrency_lease(
        self,
        identifier: str,
        max_concurrent: int = 2,
        lease_timeout_seconds: int = 60,
    ) -> str | None:
        """Acquire a bounded concurrency slot. Returns lease_id on success, or None if full."""
        ...

    async def release_concurrency_lease(
        self,
        identifier: str,
        lease_id: str,
    ) -> None:
        """Release an acquired concurrency lease."""
        ...

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        """Retrieve cached embedding vector by query SHA-256 hash."""
        ...

    async def set_cached_embedding(
        self,
        text_hash: str,
        embedding: list[float],
        ttl_seconds: int = 86400 * 7,
    ) -> None:
        """Store embedding vector under query SHA-256 hash."""
        ...


class InMemoryCoordinationService:
    """Thread-safe and async-safe in-memory coordination service for testing and single-node setups."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._idempotency: dict[str, dict[str, Any]] = {}
        self._rate_limits: dict[str, list[float]] = {}
        self._concurrency: dict[str, dict[str, float]] = {}
        self._embedding_cache: dict[str, tuple[list[float], float]] = {}

    async def check_or_reserve_idempotency(
        self,
        idempotency_key: str,
        organization_id: str,
        ttl_seconds: int = 300,
    ) -> tuple[bool, dict[str, Any] | None]:
        composite_key = f"{organization_id}:{idempotency_key}"
        now = time.time()
        with self._lock:
            entry = self._idempotency.get(composite_key)
            if entry is not None:
                if entry["expires_at"] < now:
                    del self._idempotency[composite_key]
                elif entry["status"] == "in_progress":
                    return False, None
                elif entry["status"] == "completed":
                    return True, entry["response"]

            self._idempotency[composite_key] = {
                "status": "in_progress",
                "expires_at": now + ttl_seconds,
                "response": None,
            }
            return True, None

    async def save_idempotent_response(
        self,
        idempotency_key: str,
        organization_id: str,
        response: dict[str, Any],
        status_code: int = 200,
        ttl_seconds: int = 86400,
    ) -> None:
        composite_key = f"{organization_id}:{idempotency_key}"
        now = time.time()
        with self._lock:
            self._idempotency[composite_key] = {
                "status": "completed",
                "expires_at": now + ttl_seconds,
                "response": {
                    "status_code": status_code,
                    "body": response,
                },
            }

    async def release_idempotency(
        self,
        idempotency_key: str,
        organization_id: str,
    ) -> None:
        composite_key = f"{organization_id}:{idempotency_key}"
        with self._lock:
            self._idempotency.pop(composite_key, None)

    async def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> tuple[bool, int, float]:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            history = self._rate_limits.setdefault(identifier, [])
            # Purge timestamps outside the window
            history[:] = [t for t in history if t > cutoff]
            if len(history) >= max_requests:
                oldest = history[0]
                retry_after = max(1.0, round(oldest + window_seconds - now, 2))
                return False, 0, retry_after
            history.append(now)
            remaining = max_requests - len(history)
            return True, remaining, 0.0

    async def acquire_concurrency_lease(
        self,
        identifier: str,
        max_concurrent: int = 2,
        lease_timeout_seconds: int = 60,
    ) -> str | None:
        now = time.time()
        with self._lock:
            leases = self._concurrency.setdefault(identifier, {})
            # Prune expired leases
            expired = [lid for lid, exp in leases.items() if exp < now]
            for lid in expired:
                del leases[lid]

            if len(leases) >= max_concurrent:
                return None
            lease_id = str(uuid.uuid4())
            leases[lease_id] = now + lease_timeout_seconds
            return lease_id

    async def release_concurrency_lease(
        self,
        identifier: str,
        lease_id: str,
    ) -> None:
        with self._lock:
            leases = self._concurrency.get(identifier)
            if leases:
                leases.pop(lease_id, None)

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        now = time.time()
        with self._lock:
            entry = self._embedding_cache.get(text_hash)
            if entry:
                vec, exp = entry
                if exp > now:
                    return vec
                del self._embedding_cache[text_hash]
        return None

    async def set_cached_embedding(
        self,
        text_hash: str,
        embedding: list[float],
        ttl_seconds: int = 86400 * 7,
    ) -> None:
        now = time.time()
        with self._lock:
            self._embedding_cache[text_hash] = (embedding, now + ttl_seconds)


class RedisCoordinationService:
    """Production Redis-backed coordination service for distributed deployments."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    async def check_or_reserve_idempotency(
        self,
        idempotency_key: str,
        organization_id: str,
        ttl_seconds: int = 300,
    ) -> tuple[bool, dict[str, Any] | None]:
        client = await self._get_client()
        redis_key = f"idemp:{organization_id}:{idempotency_key}"
        # Set NX for reservation
        reserved = await client.set(redis_key, json.dumps({"status": "in_progress"}), nx=True, ex=ttl_seconds)
        if reserved:
            return True, None
        # Key already existed
        val = await client.get(redis_key)
        if not val:
            return True, None
        data = json.loads(val)
        if data.get("status") == "in_progress":
            return False, None
        return True, data.get("response")

    async def save_idempotent_response(
        self,
        idempotency_key: str,
        organization_id: str,
        response: dict[str, Any],
        status_code: int = 200,
        ttl_seconds: int = 86400,
    ) -> None:
        client = await self._get_client()
        redis_key = f"idemp:{organization_id}:{idempotency_key}"
        payload = {
            "status": "completed",
            "response": {
                "status_code": status_code,
                "body": response,
            },
        }
        await client.set(redis_key, json.dumps(payload), ex=ttl_seconds)

    async def release_idempotency(
        self,
        idempotency_key: str,
        organization_id: str,
    ) -> None:
        client = await self._get_client()
        redis_key = f"idemp:{organization_id}:{idempotency_key}"
        await client.delete(redis_key)

    async def check_rate_limit(
        self,
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> tuple[bool, int, float]:
        client = await self._get_client()
        now = time.time()
        cutoff = now - window_seconds
        redis_key = f"ratelimit:{identifier}"

        pipe = client.pipeline()
        pipe.zremrangebyscore(redis_key, 0, cutoff)
        pipe.zcard(redis_key)
        pipe.zrange(redis_key, 0, 0, withscores=True)
        results = await pipe.execute()
        current_count = results[1]
        oldest_items = results[2]

        if current_count >= max_requests:
            retry_after = 1.0
            if oldest_items:
                oldest_ts = oldest_items[0][1]
                retry_after = max(1.0, round(oldest_ts + window_seconds - now, 2))
            return False, 0, retry_after

        pipe = client.pipeline()
        pipe.zadd(redis_key, {str(now): now})
        pipe.expire(redis_key, window_seconds * 2)
        await pipe.execute()
        return True, max_requests - (current_count + 1), 0.0

    async def acquire_concurrency_lease(
        self,
        identifier: str,
        max_concurrent: int = 2,
        lease_timeout_seconds: int = 60,
    ) -> str | None:
        client = await self._get_client()
        now = time.time()
        redis_key = f"concurrency:{identifier}"

        # Clean expired leases
        await client.zremrangebyscore(redis_key, 0, now)
        count = await client.zcard(redis_key)
        if count >= max_concurrent:
            return None

        lease_id = str(uuid.uuid4())
        await client.zadd(redis_key, {lease_id: now + lease_timeout_seconds})
        await client.expire(redis_key, lease_timeout_seconds * 2)
        return lease_id

    async def release_concurrency_lease(
        self,
        identifier: str,
        lease_id: str,
    ) -> None:
        client = await self._get_client()
        redis_key = f"concurrency:{identifier}"
        await client.zrem(redis_key, lease_id)

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        client = await self._get_client()
        val = await client.get(f"emb:{text_hash}")
        if val:
            return json.loads(val)
        return None

    async def set_cached_embedding(
        self,
        text_hash: str,
        embedding: list[float],
        ttl_seconds: int = 86400 * 7,
    ) -> None:
        client = await self._get_client()
        await client.set(f"emb:{text_hash}", json.dumps(embedding), ex=ttl_seconds)


def create_coordination_service(redis_url: str | None = None) -> CoordinationService:
    """Factory creating RedisCoordinationService if redis_url configured, else InMemoryCoordinationService."""
    if redis_url:
        return RedisCoordinationService(redis_url)
    return InMemoryCoordinationService()
