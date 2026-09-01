"""Tests for the Day 7 ReAct agent loop, state machine, tool registry, and API endpoint."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_software_engineering_agent.agent import Agent, AgentError
from ai_software_engineering_agent.agent_state import AgentResult, AgentState, AgentStatus, Observation, ToolCall
from ai_software_engineering_agent.app import create_app
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.embeddings import FakeEmbeddingClient
from ai_software_engineering_agent.llm import LLMRequest, LLMResponse
from ai_software_engineering_agent.tools import (
    ToolRegistry,
    ToolSchema,
    create_default_tool_registry,
    create_final_answer_handler,
)
from ai_software_engineering_agent.vector_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fake_settings() -> Settings:
    return Settings(
        llm_provider="openai",
        llm_model="test-model",
        openai_api_key="mock-key",
        request_timeout_seconds=30,
        allowed_repository_roots=(),
        embedding_provider="fake",
        vector_store_type="memory",
        agent_max_steps=10,
    )


@dataclass
class FakeLLMClient:
    """Fake LLM client that returns scripted responses in sequence."""

    responses: list[str]
    _call_index: int = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        text = self.responses[min(self._call_index, len(self.responses) - 1)]
        self._call_index += 1
        return LLMResponse(
            text=text,
            provider="fake",
            model="fake-model",
            provider_request_id=f"fake-req-{self._call_index}",
        )


@dataclass
class FakeLLMClientSingle:
    """Fake LLM client that always returns the same response."""

    response_text: str = "A generated answer."

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=self.response_text,
            provider="fake",
            model="fake-model",
            provider_request_id="fake-req-1",
        )


def make_tool_call_response(tool_name: str, arguments: dict) -> str:
    """Build a fake LLM response containing a <tool_call> block."""
    payload = json.dumps({"name": tool_name, "arguments": arguments})
    return f"I should use the {tool_name} tool.\n<tool_call>\n{payload}\n</tool_call>"


def make_final_answer_response(answer: str) -> str:
    return make_tool_call_response("final_answer", {"answer": answer})


# ---------------------------------------------------------------------------
# AgentState unit tests
# ---------------------------------------------------------------------------


def test_agent_state_initial_conditions():
    state = AgentState(task="Write a function.", max_steps=5)

    assert state.task == "Write a function."
    assert state.max_steps == 5
    assert state.current_step == 0
    assert state.status == AgentStatus.IDLE
    assert state.final_answer is None
    assert not state.is_done()
    assert state.has_budget()


def test_agent_state_thinking_increments_step():
    state = AgentState(task="Some task.", max_steps=5)

    step = state.add_thinking_step("I need to search for the function.")

    assert state.current_step == 1
    assert state.status == AgentStatus.THINKING
    assert step.reasoning == "I need to search for the function."
    assert step.status == AgentStatus.THINKING


def test_agent_state_action_and_observation_recorded():
    state = AgentState(task="Some task.", max_steps=5)
    state.add_thinking_step("Thinking...")
    tc = ToolCall(tool_name="search_code", arguments={"query": "yield curve"})
    state.add_action_step(tc)
    obs = Observation(call_id=tc.call_id, tool_name="search_code", content="Found 3 chunks.")
    state.add_observation_step(obs)

    assert state.status == AgentStatus.OBSERVING
    assert len(state.steps) == 3
    assert state.steps[1].tool_call is tc
    assert state.steps[2].observation is obs


def test_agent_state_finish_marks_done():
    state = AgentState(task="Some task.", max_steps=5)
    state.finish("The answer is 42.")

    assert state.is_done()
    assert state.status == AgentStatus.FINISHED
    assert state.final_answer == "The answer is 42."


def test_agent_state_fail_marks_done_with_error():
    state = AgentState(task="Some task.", max_steps=5)
    state.fail("LLM timed out.")

    assert state.is_done()
    assert state.status == AgentStatus.ERROR
    assert "LLM timed out." in state.final_answer


def test_agent_state_budget_exhaustion():
    state = AgentState(task="Some task.", max_steps=2)
    state.add_thinking_step("Step 1")
    state.add_thinking_step("Step 2")

    assert not state.has_budget()


def test_agent_state_elapsed_ms_is_positive():
    state = AgentState(task="Some task.")
    elapsed = state.elapsed_ms()
    assert elapsed >= 0


# ---------------------------------------------------------------------------
# ToolRegistry unit tests
# ---------------------------------------------------------------------------


def test_tool_registry_registers_and_retrieves_tool():
    registry = ToolRegistry()
    schema = ToolSchema(name="my_tool", description="Does something.")

    async def handler(args: dict, state: AgentState) -> str:
        return "result"

    registry.register("my_tool", handler, schema)

    assert registry.has("my_tool")
    assert registry.get("my_tool") is handler
    assert "my_tool" in registry.list_names()


def test_tool_registry_execute_unknown_tool_returns_error_observation():
    registry = ToolRegistry()
    tc = ToolCall(tool_name="nonexistent", arguments={})

    obs = asyncio.run(registry.execute(tc, AgentState(task="t")))

    assert obs.is_error
    assert "Unknown tool" in obs.content
    assert obs.tool_name == "nonexistent"


def test_tool_registry_execute_known_tool_returns_result():
    registry = ToolRegistry()
    schema = ToolSchema(name="echo", description="Echoes input.")

    async def echo_handler(args: dict, state: AgentState) -> str:
        return f"echo: {args.get('msg', '')}"

    registry.register("echo", echo_handler, schema)
    tc = ToolCall(tool_name="echo", arguments={"msg": "hello"})

    obs = asyncio.run(registry.execute(tc, AgentState(task="t")))

    assert not obs.is_error
    assert obs.content == "echo: hello"
    assert obs.duration_ms >= 0


def test_tool_registry_execute_captures_handler_exception():
    registry = ToolRegistry()
    schema = ToolSchema(name="bad_tool", description="Always fails.")

    async def bad_handler(args: dict, state: AgentState) -> str:
        raise ValueError("Something went wrong internally")

    registry.register("bad_tool", bad_handler, schema)
    tc = ToolCall(tool_name="bad_tool", arguments={})

    obs = asyncio.run(registry.execute(tc, AgentState(task="t")))

    assert obs.is_error
    assert "Something went wrong internally" in obs.content


def test_tool_registry_format_for_prompt_contains_tool_names():
    registry = ToolRegistry()
    registry.register(
        "search_code",
        create_final_answer_handler(),  # handler doesn't matter for formatting
        ToolSchema(
            name="search_code",
            description="Search the codebase.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
        ),
    )

    prompt_block = registry.format_for_prompt()

    assert "search_code" in prompt_block
    assert "Search the codebase." in prompt_block
    assert "query" in prompt_block
    assert "(required)" in prompt_block


# ---------------------------------------------------------------------------
# Agent._parse_llm_response unit tests
# ---------------------------------------------------------------------------


def test_agent_parses_tool_call_from_response():
    agent = Agent(
        llm_client=FakeLLMClientSingle(),
        tool_registry=ToolRegistry(),
        settings=fake_settings(),
    )

    response_text = make_tool_call_response("search_code", {"query": "bootstrap rate"})
    result = agent._parse_llm_response(response_text)

    assert isinstance(result, ToolCall)
    assert result.tool_name == "search_code"
    assert result.arguments == {"query": "bootstrap rate"}


def test_agent_treats_plain_text_as_direct_answer():
    agent = Agent(
        llm_client=FakeLLMClientSingle(),
        tool_registry=ToolRegistry(),
        settings=fake_settings(),
    )

    plain_text = "The yield curve is constructed by bootstrapping from market instruments."
    result = agent._parse_llm_response(plain_text)

    assert isinstance(result, str)
    assert result == plain_text


def test_agent_falls_back_to_plain_text_on_malformed_json():
    agent = Agent(
        llm_client=FakeLLMClientSingle(),
        tool_registry=ToolRegistry(),
        settings=fake_settings(),
    )

    malformed = "<tool_call>\n{not valid json}\n</tool_call>"
    result = agent._parse_llm_response(malformed)

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Agent.run — full loop tests
# ---------------------------------------------------------------------------


def test_agent_runs_with_direct_answer_no_tool():
    """Agent returns a direct answer when the LLM responds without a tool call."""
    llm = FakeLLMClientSingle(
        response_text="The NPV is calculated by discounting future cash flows."
    )
    registry = ToolRegistry()
    agent = Agent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    result = asyncio.run(agent.run("What is NPV?"))

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.FINISHED.value
    assert "NPV" in result.answer
    assert result.total_steps == 1
    assert result.duration_ms >= 0


def test_agent_calls_final_answer_tool_and_terminates():
    """Agent calls final_answer tool and loop terminates cleanly."""
    expected_answer = "Zero rate is the spot rate for a given maturity."
    llm = FakeLLMClientSingle(
        response_text=make_final_answer_response(expected_answer)
    )
    registry = ToolRegistry()
    registry.register("final_answer", create_final_answer_handler(), ToolSchema(
        name="final_answer",
        description="Submit final answer.",
        parameters={"type": "object", "properties": {"answer": {"type": "string", "description": ""}}, "required": ["answer"]},
    ))
    agent = Agent(llm_client=llm, tool_registry=registry, settings=fake_settings())

    result = asyncio.run(agent.run("What is a zero rate?"))

    assert result.status == AgentStatus.FINISHED.value
    assert result.answer == expected_answer


def test_agent_uses_custom_tool_and_records_observation():
    """Agent selects a custom tool, receives the observation, then answers."""
    tool_response = "Found: zero_rate function at curve.py:10-30"
    expected_answer = "zero_rate is defined at curve.py lines 10-30."

    llm = FakeLLMClient(responses=[
        # Step 1: call search_code
        make_tool_call_response("search_code", {"query": "zero_rate"}),
        # Step 2: after seeing observation, call final_answer
        make_final_answer_response(expected_answer),
    ])

    registry = ToolRegistry()

    async def fake_search(args: dict, state: AgentState) -> str:
        return tool_response

    registry.register("search_code", fake_search, ToolSchema(name="search_code", description="Search."))
    registry.register("final_answer", create_final_answer_handler(), ToolSchema(
        name="final_answer", description="Submit final answer.",
        parameters={"type": "object", "properties": {"answer": {"type": "string", "description": ""}}, "required": ["answer"]},
    ))

    agent = Agent(llm_client=llm, tool_registry=registry, settings=fake_settings(), max_steps=5)

    result = asyncio.run(agent.run("Where is zero_rate defined?"))

    assert result.status == AgentStatus.FINISHED.value
    assert result.answer == expected_answer
    # Trace should contain thinking → acting → observing → thinking → acting → observing
    statuses = [s.status for s in result.steps]
    assert AgentStatus.ACTING in statuses
    assert AgentStatus.OBSERVING in statuses
    # The observation content should be visible in the trace
    obs_steps = [s for s in result.steps if s.status == AgentStatus.OBSERVING]
    assert any(tool_response in s.observation.content for s in obs_steps)


def test_agent_trace_is_fully_inspectable():
    """Every step in the trace has the correct fields populated."""
    expected_answer = "The bond price is the PV of all cash flows."

    llm = FakeLLMClient(responses=[
        make_tool_call_response("lookup", {"key": "bond_price"}),
        make_final_answer_response(expected_answer),
    ])

    registry = ToolRegistry()

    async def lookup_handler(args: dict, state: AgentState) -> str:
        return "bond_price: present value of coupons + principal"

    registry.register("lookup", lookup_handler, ToolSchema(name="lookup", description="Look up a term."))
    registry.register("final_answer", create_final_answer_handler(), ToolSchema(
        name="final_answer", description="Submit final answer.",
        parameters={"type": "object", "properties": {"answer": {"type": "string", "description": ""}}, "required": ["answer"]},
    ))

    agent = Agent(llm_client=llm, tool_registry=registry, settings=fake_settings(), max_steps=5)
    result = asyncio.run(agent.run("How is bond price calculated?"))

    assert result.status == AgentStatus.FINISHED.value

    thinking_steps = [s for s in result.steps if s.status == AgentStatus.THINKING]
    acting_steps = [s for s in result.steps if s.status == AgentStatus.ACTING]
    observing_steps = [s for s in result.steps if s.status == AgentStatus.OBSERVING]

    # Thinking steps have reasoning text
    assert all(s.reasoning is not None for s in thinking_steps)
    # Acting steps have tool_call with name and arguments
    assert all(s.tool_call is not None for s in acting_steps)
    assert all(s.tool_call.tool_name for s in acting_steps)
    # Observing steps have observation content
    assert all(s.observation is not None for s in observing_steps)
    assert all(s.observation.duration_ms >= 0 for s in observing_steps)


def test_agent_enforces_step_limit_and_fails_safely():
    """Agent stops and marks ERROR when max_steps is exhausted without a final answer."""
    # Always returns a tool call to a non-terminating tool — will loop until budget runs out
    llm = FakeLLMClientSingle(
        response_text=make_tool_call_response("search_code", {"query": "something"})
    )

    registry = ToolRegistry()

    async def endless_search(args: dict, state: AgentState) -> str:
        return "Still searching..."

    registry.register("search_code", endless_search, ToolSchema(name="search_code", description="Search."))

    agent = Agent(llm_client=llm, tool_registry=registry, settings=fake_settings(), max_steps=3)

    result = asyncio.run(agent.run("Find everything."))

    assert result.status == AgentStatus.ERROR.value
    assert "Step limit exceeded" in result.answer
    assert result.total_steps == 3


def test_agent_rejects_empty_task():
    """Agent raises AgentError for an empty task string."""
    agent = Agent(
        llm_client=FakeLLMClientSingle(),
        tool_registry=ToolRegistry(),
        settings=fake_settings(),
    )

    with pytest.raises(AgentError, match="Task text cannot be empty"):
        asyncio.run(agent.run("   "))


def test_agent_handles_unknown_tool_gracefully():
    """If the LLM calls a tool that is not registered, the agent gets an error observation and continues."""
    expected_answer = "I could not find the tool, but the answer is 42."

    llm = FakeLLMClient(responses=[
        make_tool_call_response("nonexistent_tool", {"arg": "val"}),
        make_final_answer_response(expected_answer),
    ])

    registry = ToolRegistry()
    registry.register("final_answer", create_final_answer_handler(), ToolSchema(
        name="final_answer", description="Submit final answer.",
        parameters={"type": "object", "properties": {"answer": {"type": "string", "description": ""}}, "required": ["answer"]},
    ))

    agent = Agent(llm_client=llm, tool_registry=registry, settings=fake_settings(), max_steps=5)
    result = asyncio.run(agent.run("Use a nonexistent tool."))

    assert result.status == AgentStatus.FINISHED.value
    # The error observation should be in the trace
    error_obs = [s for s in result.steps if s.status == AgentStatus.OBSERVING and s.observation.is_error]
    assert len(error_obs) == 1
    assert "Unknown tool" in error_obs[0].observation.content


# ---------------------------------------------------------------------------
# POST /v1/agent/run endpoint tests
# ---------------------------------------------------------------------------


def _make_app_with_scripted_llm(responses: list[str]) -> TestClient:
    """Helper to build a TestClient with a scripted fake LLM and in-memory RAG."""
    app = create_app(
        settings=fake_settings(),
        client_factory=lambda _: FakeLLMClient(responses=responses),
        embedding_factory=lambda _: FakeEmbeddingClient(dimension=32),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )
    return TestClient(app)


def test_agent_run_endpoint_returns_direct_answer():
    """POST /v1/agent/run returns 200 with a direct answer when LLM answers without tools."""
    client = _make_app_with_scripted_llm(["The bond price equals the present value of all cash flows."])

    response = client.post("/v1/agent/run", json={"task": "What is bond price?"})

    assert response.status_code == 200
    data = response.json()
    assert "bond price" in data["answer"].lower()
    assert data["status"] == "finished"
    assert data["total_steps"] >= 1
    assert isinstance(data["steps"], list)
    assert data["request_id"]
    assert data["task"] == "What is bond price?"


def test_agent_run_endpoint_trace_contains_step_details():
    """The steps array in the response has the correct structure."""
    client = _make_app_with_scripted_llm(["Duration measures interest rate sensitivity."])

    response = client.post("/v1/agent/run", json={"task": "What is duration?"})

    assert response.status_code == 200
    steps = response.json()["steps"]
    assert len(steps) >= 1
    first_step = steps[0]
    assert "step_number" in first_step
    assert "status" in first_step
    assert first_step["step_number"] == 1


def test_agent_run_endpoint_rejects_empty_task():
    """POST /v1/agent/run returns 422 for an empty task string."""
    client = _make_app_with_scripted_llm(["irrelevant"])

    response = client.post("/v1/agent/run", json={"task": ""})

    assert response.status_code == 422


def test_agent_run_endpoint_respects_max_steps_param():
    """max_steps parameter is passed through and enforced by the agent."""
    # Always returns a looping tool call so the step limit is hit
    responses = [make_tool_call_response("final_answer", {"answer": "done"})] * 5
    # Override with something that won't terminate via final_answer
    looping_responses = [
        make_tool_call_response("search_code", {"query": "loop"})
    ] * 20  # more than any max_steps we'll set

    app = create_app(
        settings=fake_settings(),
        client_factory=lambda _: FakeLLMClient(responses=looping_responses),
        embedding_factory=lambda _: FakeEmbeddingClient(dimension=32),
        vector_store_factory=lambda _: InMemoryVectorStore(),
    )
    client = TestClient(app)

    response = client.post("/v1/agent/run", json={"task": "Keep searching.", "max_steps": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["total_steps"] == 2
    assert "Step limit exceeded" in data["answer"]


def test_agent_run_endpoint_returns_503_without_api_key():
    """POST /v1/agent/run returns 503 when no API key is configured."""
    no_key_settings = Settings(
        llm_provider="openai",
        llm_model="test-model",
        openai_api_key=None,
        request_timeout_seconds=30,
        allowed_repository_roots=(),
    )
    app = create_app(settings=no_key_settings)
    client = TestClient(app)

    response = client.post("/v1/agent/run", json={"task": "Explain zero rates."})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
