"""End-to-end integration tests for the ReAct Agent using safe engineering tools."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from ai_software_engineering_agent.agent import Agent
from ai_software_engineering_agent.app import create_app
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.engineering_tools import (
    EngineeringToolContext,
    create_engineering_tool_registry,
)
from ai_software_engineering_agent.llm import LLMRequest, LLMResponse


class ScriptedLLMClient:
    """Fake LLM client that returns a sequence of pre-scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = '<tool_call>{"name": "final_answer", "arguments": {"answer": "Default finished."}}</tool_call>'
        return LLMResponse(text=text, provider="scripted", model="scripted-model", provider_request_id=f"req-{self.call_count}")


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "integration_repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def subtract(a, b): return a - b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import subtract\ndef test_sub(): assert subtract(5, 2) == 3\n",
        encoding="utf-8",
    )
    return repo


@pytest.mark.asyncio
async def test_agent_navigates_reads_and_tests_codebase(repo_dir: Path):
    context = EngineeringToolContext(repo_root=repo_dir)
    registry = create_engineering_tool_registry(context)

    # Multi-step scripted trajectory:
    # 1. List files
    # 2. Read calc.py
    # 3. Run tests
    # 4. Final answer
    responses = [
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "read_file", "arguments": {"file_path": "calc.py"}}</tool_call>',
        '<tool_call>{"name": "run_tests", "arguments": {"test_target": "test_calc.py"}}</tool_call>',
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "The subtract function is verified and passes tests."}}</tool_call>',
    ]

    client = ScriptedLLMClient(responses)
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key="mock",
        request_timeout_seconds=30.0,
        allowed_repository_roots=(repo_dir,),
    )

    agent = Agent(llm_client=client, tool_registry=registry, settings=settings, max_steps=10)
    result = await agent.run("Verify the subtraction implementation and tests.")

    assert result.status == "finished"
    assert "subtract function is verified" in result.answer
    assert result.total_steps == 4

    # Verify tool calls recorded in trace
    tools_called = [s.tool_call.tool_name for s in result.steps if s.tool_call]
    assert tools_called == ["list_files", "read_file", "run_tests", "final_answer"]

    # Verify observations captured for intermediate tools
    observations = [s.observation.content for s in result.steps if s.observation]
    assert len(observations) == 3
    assert "calc.py" in observations[0]
    assert "def subtract" in observations[1]
    assert "Pytest Run:" in observations[2]


def test_app_agent_run_endpoint_with_repository_path(repo_dir: Path):
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key="mock",
        request_timeout_seconds=30.0,
        allowed_repository_roots=(repo_dir,),
    )

    app = create_app(
        settings=settings,
        client_factory=lambda _: ScriptedLLMClient([
            '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "final_answer", "arguments": {"answer": "Files listed successfully."}}</tool_call>',
        ]),
    )
    test_client = TestClient(app)

    resp = test_client.post(
        "/v1/agent/run",
        json={
            "task": "Explore repository files.",
            "repository_path": str(repo_dir),
            "max_steps": 5,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "finished"
    assert "Files listed" in data["answer"]
    assert data["total_steps"] == 2
    assert len(data["steps"]) >= 4  # (thinking + action + observation) * 2
