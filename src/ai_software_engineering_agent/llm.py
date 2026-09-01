"""Provider-neutral language model interface and OpenAI implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings


class LLMError(RuntimeError):
    """Base error for language model operations."""


class LLMConfigurationError(LLMError):
    """Raised when the selected provider cannot be configured."""


class LLMProviderError(LLMError):
    """Raised when a configured provider cannot generate a response."""


@dataclass(frozen=True)
class LLMRequest:
    """Application-owned input for a text generation request."""

    prompt: str
    system_instruction: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Application-owned normalized text generation response."""

    text: str
    provider: str
    model: str
    provider_request_id: str | None


class LLMClient(Protocol):
    """Minimal provider boundary used by application services."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for one prompt."""


class OpenAIResponsesClient:
    """OpenAI Responses API adapter kept outside the FastAPI layer."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise LLMConfigurationError(
                    "The OpenAI SDK is not installed. Install the project dependencies first."
                ) from error
            client = OpenAI(api_key=api_key, timeout=timeout_seconds)

        self._client = client
        self._model = model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        request_args: dict[str, Any] = {"model": self._model, "input": request.prompt}
        if request.system_instruction:
            request_args["instructions"] = request.system_instruction

        try:
            response = await asyncio.to_thread(self._client.responses.create, **request_args)
        except Exception as error:
            raise LLMProviderError("OpenAI could not generate a response.") from error

        text = getattr(response, "output_text", "")
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("OpenAI returned no text output.")

        return LLMResponse(
            text=text,
            provider="openai",
            model=getattr(response, "model", self._model),
            provider_request_id=getattr(response, "id", None),
        )


class FakeLLMClient:
    """Fake LLM client for offline development and testing."""

    def __init__(self, response_text: str = "A mock generated answer grounded in context.") -> None:
        self.response_text = response_text

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=self.response_text,
            provider="fake",
            model="fake-model",
            provider_request_id="fake-req-1",
        )


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the configured LLM client after validating provider-specific settings."""
    if settings.llm_provider == "fake":
        return FakeLLMClient()
    if settings.llm_provider != "openai":
        raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}.")
    if not settings.openai_api_key:
        raise LLMConfigurationError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai.")

    return OpenAIResponsesClient(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.request_timeout_seconds,
    )
