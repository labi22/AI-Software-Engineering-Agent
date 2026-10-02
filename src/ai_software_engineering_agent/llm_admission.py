"""Admission controller, bounded retries, and quota handling for LLM providers."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from .config import Settings
from .llm import (
    LLMClient,
    LLMProviderError,
    LLMQuotaExhaustedError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class AdmissionControlledLLMClient:
    """Wraps an LLMClient with rate pacing, timeout enforcement, and jittered exponential retry on 429/5xx."""

    def __init__(
        self,
        inner_client: LLMClient,
        *,
        rpm_limit: int = 30,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.inner_client = inner_client
        self.rpm_limit = max(1, rpm_limit)
        self.max_retries = max(0, max_retries)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout_seconds = timeout_seconds

        self._min_interval = 60.0 / self.rpm_limit
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def _pace(self) -> None:
        """Throttle requests to respect rpm_limit."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        err_msg = str(exc).lower()
        return (
            "429" in err_msg
            or "rate limit" in err_msg
            or "quota" in err_msg
            or "tpm" in err_msg
            or "rpm" in err_msg
            or "too many requests" in err_msg
        )

    def _is_transient_error(self, exc: Exception) -> bool:
        err_msg = str(exc).lower()
        return (
            self._is_rate_limit_error(exc)
            or "500" in err_msg
            or "502" in err_msg
            or "503" in err_msg
            or "504" in err_msg
            or "service unavailable" in err_msg
            or "bad gateway" in err_msg
            or "connection reset" in err_msg
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute generate request with admission pacing and bounded exponential retries."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            await self._pace()
            try:
                return await asyncio.wait_for(
                    self.inner_client.generate(request),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "LLM request attempt %d timed out after %.1fs",
                    attempt + 1,
                    self.timeout_seconds,
                )
                if attempt >= self.max_retries:
                    raise LLMTimeoutError(
                        f"LLM request timed out after {self.timeout_seconds}s ({self.max_retries + 1} attempts)."
                    ) from exc

            except LLMProviderError as exc:
                last_error = exc
                if not self._is_transient_error(exc) or attempt >= self.max_retries:
                    if self._is_rate_limit_error(exc):
                        raise LLMQuotaExhaustedError(
                            f"LLM provider rate limit or quota exhausted: {exc}"
                        ) from exc
                    raise

            except Exception as exc:
                last_error = exc
                if not self._is_transient_error(exc) or attempt >= self.max_retries:
                    if self._is_rate_limit_error(exc):
                        raise LLMQuotaExhaustedError(
                            f"LLM provider quota exhausted: {exc}"
                        ) from exc
                    raise LLMProviderError(f"LLM generation failed: {exc}") from exc

            # Calculate jittered exponential backoff
            delay = min(self.max_delay, self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.5)
            logger.info(
                "Retrying LLM request in %.2fs (attempt %d/%d) due to: %s",
                delay,
                attempt + 1,
                self.max_retries,
                last_error,
            )
            await asyncio.sleep(delay)

        raise LLMProviderError(f"Exhausted all {self.max_retries + 1} attempts: {last_error}")


def create_admission_controlled_client(
    inner_client: LLMClient,
    settings: Settings,
) -> LLMClient:
    """Wrap an existing LLMClient with configured admission controls."""
    return AdmissionControlledLLMClient(
        inner_client=inner_client,
        rpm_limit=settings.llm_rpm_limit,
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_retry_base_delay,
        max_delay=settings.llm_retry_max_delay,
        timeout_seconds=settings.request_timeout_seconds,
    )
