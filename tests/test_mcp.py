"""Tests for Model Context Protocol (MCP) server, client, adapter, and JSON-RPC 2.0 engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_software_engineering_agent.agent import Agent
from ai_software_engineering_agent.app import create_app
from ai_software_engineering_agent.config import Settings
from ai_software_engineering_agent.engineering_tools import EngineeringToolContext
from ai_software_engineering_agent.llm import FakeLLMClient
from ai_software_engineering_agent.mcp import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    MCP_PROTOCOL_VERSION,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    MCPClient,
    MCPError,
    MCPServer,
    MCPToolAdapter,
    guess_mime_type,
)
from ai_software_engineering_agent.tools import ToolRegistry, create_final_answer_handler, FINAL_ANSWER_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_fixture(tmp_path: Path) -> Path:
    """Create a temporary repository layout for testing tools and resources."""
    root = tmp_path / "test_repo"
    root.mkdir()

    src_dir = root / "src"
    src_dir.mkdir()
    (src_dir / "calc.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (src_dir / "config.json").write_text(
        '{"env": "test", "debug": true}',
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Test Repo\n", encoding="utf-8")

    # An ignored folder that should not appear in resource list
    git_dir = root / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("dummy git config", encoding="utf-8")

    return root


@pytest.fixture
def tool_context(repo_fixture: Path) -> EngineeringToolContext:
    """Create an EngineeringToolContext scoped to the test repo."""
    return EngineeringToolContext(
        repo_root=repo_fixture,
        allowed_roots=(repo_fixture,),
        default_timeout_seconds=5.0,
        max_output_chars=4000,
    )


@pytest.fixture
def mcp_server(tool_context: EngineeringToolContext) -> MCPServer:
    """Create an MCPServer backed by safe engineering tools and repository resources."""
    return MCPServer(
        tool_context=tool_context,
        repo_id="test-repo",
        server_name="TestMCPServer",
        server_version="1.0.0",
    )


@pytest.fixture
def mcp_client(mcp_server: MCPServer) -> MCPClient:
    """Create an in-memory MCPClient connected to mcp_server."""
    return MCPClient(server=mcp_server)


# ---------------------------------------------------------------------------
# Protocol & Handshake Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_parse_error(mcp_server: MCPServer) -> None:
    """Server returns -32700 Parse error on malformed JSON string."""
    resp = await mcp_server.handle_request("{not-valid-json}")
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] is None
    assert resp["error"]["code"] == PARSE_ERROR


@pytest.mark.asyncio
async def test_mcp_invalid_request(mcp_server: MCPServer) -> None:
    """Server returns -32600 Invalid Request when jsonrpc != '2.0' or method missing."""
    resp = await mcp_server.handle_request({"id": 1, "jsonrpc": "1.0", "method": "ping"})
    assert resp is not None
    assert resp["error"]["code"] == INVALID_REQUEST

    resp2 = await mcp_server.handle_request({"id": 2, "jsonrpc": "2.0"})
    assert resp2 is not None
    assert resp2["error"]["code"] == INVALID_REQUEST


@pytest.mark.asyncio
async def test_mcp_method_not_found(mcp_server: MCPServer) -> None:
    """Server returns -32601 Method not found for unknown methods."""
    req = {"jsonrpc": "2.0", "id": 10, "method": "non_existent_method", "params": {}}
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 10
    assert resp["error"]["code"] == METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_mcp_invalid_params_type(mcp_server: MCPServer) -> None:
    """Server returns -32602 when params is not an object."""
    req = {"jsonrpc": "2.0", "id": 11, "method": "ping", "params": ["invalid_array"]}
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    assert resp["error"]["code"] == INVALID_PARAMS


@pytest.mark.asyncio
async def test_mcp_initialize_and_ping(mcp_server: MCPServer) -> None:
    """Server correctly handles initialize handshake and ping probe."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "TestClient", "version": "1.0.0"},
        },
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert "tools" in result["capabilities"]
    assert "resources" in result["capabilities"]
    assert result["serverInfo"]["name"] == "TestMCPServer"

    # Ping
    ping_resp = await mcp_server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping_resp is not None
    assert ping_resp["result"] == {}


@pytest.mark.asyncio
async def test_mcp_notifications(mcp_server: MCPServer) -> None:
    """Notifications (requests with id=None) produce None response."""
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    resp = await mcp_server.handle_request(notification)
    assert resp is None
    assert mcp_server.is_initialized is True


# ---------------------------------------------------------------------------
# Tool Discovery and Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tools_list(mcp_server: MCPServer) -> None:
    """tools/list enumerates safe engineering tools with JSON schemas."""
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]

    assert "list_files" in tool_names
    assert "read_file" in tool_names
    assert "search_code" in tool_names
    assert "run_python" in tool_names
    assert "run_tests" in tool_names
    assert "git_diff" in tool_names
    # Internal agent termination tool must NOT be exposed as an external MCP tool
    assert "final_answer" not in tool_names

    list_files_desc = next(t for t in tools if t["name"] == "list_files")
    assert "inputSchema" in list_files_desc
    assert list_files_desc["inputSchema"]["type"] == "object"


@pytest.mark.asyncio
async def test_mcp_tools_call_list_files(mcp_server: MCPServer) -> None:
    """tools/call successfully executes list_files within repository boundaries."""
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "list_files", "arguments": {"path": ""}},
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    assert resp["id"] == 4
    result = resp["result"]
    assert result["isError"] is False
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    text = result["content"][0]["text"]
    assert "src/calc.py" in text or "calc.py" in text


@pytest.mark.asyncio
async def test_mcp_tools_call_read_file(mcp_server: MCPServer) -> None:
    """tools/call successfully executes read_file."""
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "src/calc.py"}},
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    result = resp["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "def add(a: int, b: int) -> int:" in text


@pytest.mark.asyncio
async def test_mcp_tools_call_run_python(mcp_server: MCPServer) -> None:
    """tools/call executes run_python safely."""
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "run_python",
            "arguments": {"code": "print(100 + 234)"},
        },
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    result = resp["result"]
    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "334" in text


@pytest.mark.asyncio
async def test_mcp_tools_call_safety_violation(mcp_server: MCPServer) -> None:
    """Path traversal attempt in tools/call returns isError: True content block."""
    req = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {"path": "../../outside.txt"},
        },
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    result = resp["result"]
    assert result["isError"] is True
    assert "Security Error" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_tools_call_unknown_tool(mcp_server: MCPServer) -> None:
    """Calling an unknown tool returns isError: True."""
    req = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "ghost_tool", "arguments": {}},
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    result = resp["result"]
    assert result["isError"] is True
    assert "Tool Not Found" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_server_read_only_mode(tool_context: EngineeringToolContext) -> None:
    """Read-only server mode omits execution tools and disallows running them."""
    ro_server = MCPServer(
        tool_context=tool_context,
        repo_id="test-repo",
        read_only=True,
    )

    # 1. tools/list must not include run_python or run_tests
    list_resp = await ro_server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "run_python" not in tool_names
    assert "run_tests" not in tool_names
    assert "list_files" in tool_names

    # 2. tools/call to run_python must be rejected
    call_resp = await ro_server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "run_python", "arguments": {"code": "print('hi')"}},
        }
    )
    assert call_resp["result"]["isError"] is True
    assert "Permission Denied" in call_resp["result"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Resources Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_resources_list(mcp_server: MCPServer) -> None:
    """resources/list lists repository files with repo:// URIs and MIME types."""
    req = {"jsonrpc": "2.0", "id": 9, "method": "resources/list", "params": {}}
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    resources = resp["result"]["resources"]
    uris = [r["uri"] for r in resources]

    assert "repo://test-repo/src/calc.py" in uris
    assert "repo://test-repo/src/config.json" in uris
    assert "repo://test-repo/README.md" in uris
    # Excluded folders like .git must not be exposed
    assert not any(".git" in u for u in uris)

    calc_res = next(r for r in resources if "calc.py" in r["uri"])
    assert calc_res["mimeType"] == "text/x-python"


@pytest.mark.asyncio
async def test_mcp_resources_read(mcp_server: MCPServer) -> None:
    """resources/read reads file content by URI."""
    req = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "resources/read",
        "params": {"uri": "repo://test-repo/src/calc.py"},
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    contents = resp["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["uri"] == "repo://test-repo/src/calc.py"
    assert contents[0]["mimeType"] == "text/x-python"
    assert "def add(a: int, b: int) -> int:" in contents[0]["text"]


@pytest.mark.asyncio
async def test_mcp_resources_read_path_traversal(mcp_server: MCPServer) -> None:
    """resources/read blocks path traversal via -32602 Invalid params."""
    req = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "resources/read",
        "params": {"uri": "repo://test-repo/../../secret.txt"},
    }
    resp = await mcp_server.handle_request(req)
    assert resp is not None
    assert resp["error"]["code"] == INVALID_PARAMS
    assert "Path Security Error" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# MCP Client Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_client_workflow(mcp_client: MCPClient) -> None:
    """MCPClient successfully initializes, lists tools, calls tools, and reads resources."""
    # 1. Initialize
    init_res = await mcp_client.initialize()
    assert init_res["protocolVersion"] == MCP_PROTOCOL_VERSION

    # 2. Ping
    assert await mcp_client.ping() is True

    # 3. List tools
    tools = await mcp_client.list_tools()
    names = [t.name for t in tools]
    assert "read_file" in names
    assert "list_files" in names

    # 4. Call tool
    output, is_error = await mcp_client.call_tool("read_file", {"path": "src/calc.py"})
    assert is_error is False
    assert "def add(a: int, b: int) -> int:" in output

    # 5. List resources
    resources = await mcp_client.list_resources()
    uris = [r.uri for r in resources]
    assert "repo://test-repo/src/calc.py" in uris

    # 6. Read resource
    content = await mcp_client.read_resource("repo://test-repo/src/calc.py")
    assert "def add(a: int, b: int) -> int:" in content


# ---------------------------------------------------------------------------
# MCP Tool Adapter & ReAct Agent Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_adapter_with_agent(mcp_client: MCPClient) -> None:
    """Agent seamlessly consumes MCP tools adapted into its native ToolRegistry."""
    registry = ToolRegistry()
    registry.register("final_answer", create_final_answer_handler(), FINAL_ANSWER_SCHEMA)

    # Adapt MCP tools from client into registry
    registered = await MCPToolAdapter.adapt_mcp_to_registry(
        client=mcp_client,
        registry=registry,
        include_tools=["read_file"],
    )
    assert "read_file" in registered
    assert registry.has("read_file")

    # Wire Agent with FakeLLMClient to execute read_file via MCP and then answer
    llm_responses = [
        # Step 1: Agent calls read_file (which will be executed via MCP)
        'I need to read the calculator implementation.\n<tool_call>{"name": "read_file", "arguments": {"path": "src/calc.py"}}</tool_call>',
        # Step 2: Agent observes content and finishes
        'I have read calc.py.\n<tool_call>{"name": "final_answer", "arguments": {"answer": "calc.py defines an add function."}}</tool_call>',
    ]

    class SequentialFakeLLM:
        def __init__(self, responses: list[str]) -> None:
            self.responses = list(responses)
            self._idx = 0

        async def generate(self, request: Any) -> Any:
            from ai_software_engineering_agent.llm import LLMResponse
            text = self.responses[min(self._idx, len(self.responses) - 1)]
            self._idx += 1
            return LLMResponse(
                text=text,
                provider="fake",
                model="fake-model",
                provider_request_id="fake-id",
            )

    fake_llm = SequentialFakeLLM(responses=llm_responses)
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key="mock",
        request_timeout_seconds=30.0,
        allowed_repository_roots=(),
    )
    agent = Agent(llm_client=fake_llm, tool_registry=registry, settings=settings)

    result = await agent.run("What does calc.py define?")
    assert result.status == "finished"
    assert "calc.py defines an add function" in result.answer
    assert result.total_steps >= 2

    # Verify that the read_file tool was executed and recorded in the trace
    step_tools = [s.tool_call.tool_name for s in result.steps if s.tool_call is not None]
    assert "read_file" in step_tools
    obs = next(s.observation for s in result.steps if s.observation and s.observation.tool_name == "read_file")
    assert "def add(a: int, b: int) -> int:" in obs.content


# ---------------------------------------------------------------------------
# FastAPI POST /v1/mcp Endpoint Tests
# ---------------------------------------------------------------------------


def test_api_mcp_endpoint_initialize(repo_fixture: Path) -> None:
    """FastAPI POST /v1/mcp handles JSON-RPC initialize handshake."""
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key=None,
        request_timeout_seconds=30.0,
        allowed_repository_roots=(Path(repo_fixture),),
        embedding_provider="fake",
        vector_store_type="memory",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "TestHTTPClient", "version": "1.0"},
        },
    }
    response = client.post("/v1/mcp", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert data["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION


def test_api_mcp_endpoint_tool_call(repo_fixture: Path) -> None:
    """FastAPI POST /v1/mcp executes tool over HTTP with repository context."""
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key=None,
        request_timeout_seconds=30.0,
        allowed_repository_roots=(Path(repo_fixture),),
        embedding_provider="fake",
        vector_store_type="memory",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "read_file",
            "arguments": {"path": "src/calc.py"},
        },
    }
    response = client.post(
        "/v1/mcp",
        json=req,
        headers={"X-Repository-Path": str(repo_fixture), "X-Repo-ID": "test-repo"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["isError"] is False
    text = data["result"]["content"][0]["text"]
    assert "def add(a: int, b: int) -> int:" in text


def test_api_mcp_endpoint_parse_error(repo_fixture: Path) -> None:
    """FastAPI POST /v1/mcp returns parse error when non-JSON body is sent."""
    settings = Settings(
        llm_provider="fake",
        llm_model="fake-model",
        openai_api_key=None,
        request_timeout_seconds=30.0,
        allowed_repository_roots=(Path(repo_fixture),),
        embedding_provider="fake",
        vector_store_type="memory",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.post(
        "/v1/mcp",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["error"]["code"] == PARSE_ERROR


def test_guess_mime_type() -> None:
    """guess_mime_type returns accurate MIME types for common software files."""
    assert guess_mime_type("file.py") == "text/x-python"
    assert guess_mime_type("file.json") == "application/json"
    assert guess_mime_type("file.md") == "text/markdown"
    assert guess_mime_type("file.yaml") == "application/yaml"
    assert guess_mime_type("file.sql") == "text/x-sql"
    assert guess_mime_type("file.unknown") == "text/plain"
