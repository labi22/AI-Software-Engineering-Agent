"""Unit and integration tests for the LangGraph orchestration engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_software_engineering_agent.agent import AgentError
from ai_software_engineering_agent.agent_state import AgentStatus, ToolCall
from ai_software_engineering_agent.app import create_app
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.langgraph_agent import LangGraphAgent
from ai_software_engineering_agent.llm import LLMRequest, LLMResponse, LLMToolCall
from ai_software_engineering_agent.tools import (
    ToolRegistry,
    ToolSchema,
    create_final_answer_handler,
)
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


def fake_settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_model="test-model",
        openai_api_key="mock-key",
        request_timeout_seconds=30,
        allowed_repository_roots=(),
        embedding_provider="fake",
        vector_store_type="memory",
        agent_max_steps=5,
    )


@dataclass
class ScriptedLLMClient:
    """Scripted LLM client for deterministic testing of LangGraph workflows."""

    responses: list[LLMResponse | str]
    _call_index: int = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        item = self.responses[min(self._call_index, len(self.responses) - 1)]
        self._call_index += 1
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=item, provider="fake", model="fake-model", provider_request_id=None)


def create_test_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "final_answer",
        create_final_answer_handler(),
        ToolSchema(
            name="final_answer",
            description="Call when done",
            parameters={"type": "object", "properties": {"answer": {"type": "string"}}},
        ),
    )

    async def echo_handler(args: dict[str, Any], state: Any) -> str:
        msg = args.get("msg", "")
        return f"Echo: {msg}"

    registry.register(
        "echo_tool",
        echo_handler,
        ToolSchema(
            name="echo_tool",
            description="Echo a message",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        ),
    )
    return registry


# ---------------------------------------------------------------------------
# LangGraphAgent Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_langgraph_direct_answer():
    """Agent answers directly without invoking any tools."""
    llm = ScriptedLLMClient(responses=["FastAPI uses Starlette under the hood."])
    registry = create_test_registry()
    agent = LangGraphAgent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    result = await agent.run("What framework does FastAPI build on?")

    assert result.status == "finished"
    assert "Starlette" in result.answer
    assert result.total_steps >= 1
    assert any(s.status == AgentStatus.THINKING for s in result.steps)


@pytest.mark.asyncio
async def test_langgraph_xml_tool_call_and_final_answer():
    """Agent calls a tool via XML tag then calls final_answer."""
    responses = [
        '<tool_call>{"name": "echo_tool", "arguments": {"msg": "hello_graph"}}</tool_call>',
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "Received echo: hello_graph"}}</tool_call>',
    ]
    llm = ScriptedLLMClient(responses=responses)
    registry = create_test_registry()
    agent = LangGraphAgent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    result = await agent.run("Run echo test")

    assert result.status == "finished"
    assert "Received echo: hello_graph" in result.answer
    # Should have thinking, acting, and observing steps
    assert any(s.status == AgentStatus.ACTING and s.tool_call and s.tool_call.tool_name == "echo_tool" for s in result.steps)
    assert any(s.status == AgentStatus.OBSERVING and s.observation and "Echo: hello_graph" in s.observation.content for s in result.steps)


@pytest.mark.asyncio
async def test_langgraph_native_tool_call():
    """Agent receives native function/tool calls from LLM client."""
    responses = [
        LLMResponse(
            text="",
            provider="fake",
            model="fake-model",
            provider_request_id=None,
            tool_calls=(LLMToolCall(name="echo_tool", arguments={"msg": "native_call"}, call_id="c1"),),
        ),
        LLMResponse(
            text="",
            provider="fake",
            model="fake-model",
            provider_request_id=None,
            tool_calls=(LLMToolCall(name="final_answer", arguments={"answer": "Native tool done"}, call_id="c2"),),
        ),
    ]
    llm = ScriptedLLMClient(responses=responses)
    registry = create_test_registry()
    agent = LangGraphAgent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    result = await agent.run("Run native test")

    assert result.status == "finished"
    assert "Native tool done" in result.answer


@pytest.mark.asyncio
async def test_langgraph_step_budget_exhaustion():
    """Agent fails gracefully when exceeding maximum allowed steps."""
    # Keeps calling echo_tool indefinitely
    infinite_tool = '<tool_call>{"name": "echo_tool", "arguments": {"msg": "loop"}}</tool_call>'
    llm = ScriptedLLMClient(responses=[infinite_tool])
    registry = create_test_registry()
    agent = LangGraphAgent(
        llm_client=llm,
        tool_registry=registry,
        settings=fake_settings(),
        max_steps=2,
    )

    result = await agent.run("Loop forever")

    assert result.status == "failed"
    assert "Step limit exceeded" in result.answer


@pytest.mark.asyncio
async def test_langgraph_empty_task_raises_error():
    """LangGraphAgent raises AgentError on empty task string."""
    llm = ScriptedLLMClient(responses=[""])
    registry = create_test_registry()
    agent = LangGraphAgent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    with pytest.raises(AgentError):
        await agent.run("   ")


@pytest.mark.asyncio
async def test_langgraph_checkpoint_persistence():
    """LangGraph checkpointer stores checkpoints during execution."""
    llm = ScriptedLLMClient(responses=["Simple answer."])
    registry = create_test_registry()
    agent = LangGraphAgent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    result = await agent.run("Test checkpointing")
    assert result.status == "finished"
    assert agent.checkpointer is not None


# ---------------------------------------------------------------------------
# API Integration Tests (/v1/agent/run)
# ---------------------------------------------------------------------------


def test_api_agent_run_uses_langgraph():
    """POST /v1/agent/run always executes through LangGraph."""
    settings = fake_settings()
    llm = ScriptedLLMClient(responses=["LangGraph API response."])
    vstore = InMemoryVectorStore()
    eclient = FakeEmbeddingClient()

    app = create_app(
        settings=settings,
        client_factory=lambda s: llm,
        embedding_factory=lambda s: eclient,
        vector_store_factory=lambda s: vstore,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/agent/run",
        json={"task": "Explain LangGraph"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "finished"
    assert data["answer"] == "LangGraph API response."
    assert data["orchestration_engine"] == "langgraph"
    assert data["total_steps"] >= 1


def test_api_agent_run_rejects_legacy_engine_selection():
    """The public API cannot silently accept a retired custom-engine option."""
    app = create_app(
        settings=fake_settings(),
        client_factory=lambda _settings: ScriptedLLMClient(responses=[]),
        embedding_factory=lambda _settings: FakeEmbeddingClient(),
        vector_store_factory=lambda _settings: InMemoryVectorStore(),
    )

    response = TestClient(app).post(
        "/v1/agent/run",
        json={"task": "Explain LangGraph", "engine": "custom"},
    )

    assert response.status_code == 422
