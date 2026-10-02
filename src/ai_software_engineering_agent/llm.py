"""Provider-neutral language model interface and OpenAI implementation."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base error for language model operations."""


class LLMConfigurationError(LLMError):
    """Raised when the selected provider cannot be configured."""


class LLMProviderError(LLMError):
    """Raised when a configured provider cannot generate a response."""


class LLMQuotaExhaustedError(LLMProviderError):
    """Raised when a provider rejects requests due to quota/rate limit exhaustion (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM operation exceeds the configured timeout deadline."""


@dataclass(frozen=True)
class LLMRequest:
    """Application-owned input for a text generation request."""

    prompt: str
    system_instruction: str | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LLMToolCall:
    """A provider-native function call requested by an LLM."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Application-owned normalized text generation response."""

    text: str
    provider: str
    model: str
    provider_request_id: str | None
    tool_calls: tuple[LLMToolCall, ...] = ()


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
        base_url: str | None = None,
        provider: str = "openai",
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise LLMConfigurationError(
                    "The OpenAI SDK is not installed. Install the project dependencies first."
                ) from error
            client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)

        self._client = client
        self._model = model
        self._provider = provider

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.messages is not None:
            return await self._generate_chat_completion(request)

        request_args: dict[str, Any] = {"model": self._model, "input": request.prompt}
        if request.system_instruction:
            request_args["instructions"] = request.system_instruction

        try:
            response = await asyncio.to_thread(self._client.responses.create, **request_args)
        except Exception as error:
            # Keep provider details out of the HTTP response, but preserve the original
            # exception in the server log so provider-specific incompatibilities can be
            # diagnosed (for example, a Groq 400 response).
            logger.exception("%s Responses API request failed", self._provider)
            raise LLMProviderError(f"{self._provider.title()} could not generate a response.") from error

        text = getattr(response, "output_text", "")
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError("OpenAI returned no text output.")

        return LLMResponse(
            text=text,
            provider=self._provider,
            model=getattr(response, "model", self._model),
            provider_request_id=getattr(response, "id", None),
        )

    async def _generate_chat_completion(self, request: LLMRequest) -> LLMResponse:
        """Use Chat Completions for provider-native local function calling."""
        request_args: dict[str, Any] = {"model": self._model, "messages": request.messages}
        if request.tools:
            request_args["tools"] = request.tools
            request_args["tool_choice"] = "auto"
        if self._provider == "groq":
            # Groq reasoning models spend tokens on hidden chain-of-thought before
            # emitting any visible output.  A budget that is too small results in
            # the model returning an empty message (no text, no tool calls) after
            # reasoning is complete.  8 192 gives enough headroom for two rounds of
            # tool use where observations can contain full file contents.
            request_args["max_completion_tokens"] = 8_192
            # These are Groq extensions that OpenAI SDK 2.54 does not expose as
            # typed Chat Completions arguments. ``extra_body`` forwards them
            # unchanged to Groq's OpenAI-compatible endpoint.
            request_args["extra_body"] = {
                "reasoning_effort": "low",
                "include_reasoning": False,
            }

        try:
            response = await asyncio.to_thread(self._client.chat.completions.create, **request_args)
        except Exception as error:
            logger.exception("%s Chat Completions API request failed", self._provider)
            raise LLMProviderError(f"{self._provider.title()} could not generate a response.") from error

        message = response.choices[0].message
        tool_calls: list[LLMToolCall] = []
        for tool_call in message.tool_calls or []:
            raw_args = tool_call.function.arguments
            try:
                arguments = json.loads(raw_args)
            except (TypeError, json.JSONDecodeError):
                # Some Groq models emit the answer text verbatim as the arguments
                # value for final_answer instead of wrapping it in {"answer": "..."}.
                # Recover gracefully by treating the raw string as the "answer" field.
                if tool_call.function.name == "final_answer" and isinstance(raw_args, str):
                    logger.warning(
                        "Provider returned non-JSON arguments for final_answer; recovering as plain text."
                    )
                    arguments = {"answer": raw_args}
                else:
                    logger.error(
                        "Provider returned invalid tool-call arguments for %s: %r",
                        tool_call.function.name,
                        raw_args,
                    )
                    raise LLMProviderError(
                        f"The provider returned invalid tool-call arguments for '{tool_call.function.name}'."
                    )
            tool_calls.append(
                LLMToolCall(call_id=tool_call.id, name=tool_call.function.name, arguments=arguments)
            )

        return LLMResponse(
            text=message.content or "",
            provider=self._provider,
            model=getattr(response, "model", self._model),
            provider_request_id=getattr(response, "id", None),
            tool_calls=tuple(tool_calls),
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
    if settings.llm_provider == "openai":
        api_key = settings.openai_api_key
        base_url = settings.llm_base_url
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai.")
    elif settings.llm_provider == "groq":
        api_key = settings.groq_api_key
        base_url = settings.llm_base_url or "https://api.groq.com/openai/v1"
        if not api_key:
            raise LLMConfigurationError("GROQ_API_KEY must be set when LLM_PROVIDER=groq.")
    elif settings.llm_provider == "ollama":
        api_key = settings.openai_api_key or "ollama"
        base_url = settings.llm_base_url or "http://localhost:11434/v1"
    else:
        raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}.")

    return OpenAIResponsesClient(
        api_key=api_key,
        model=settings.llm_model,
        timeout_seconds=settings.request_timeout_seconds,
        base_url=base_url,
        provider=settings.llm_provider,
    )
