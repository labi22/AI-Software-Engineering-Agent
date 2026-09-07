"""Model Context Protocol (MCP) server, client, tool adapter, and JSON-RPC 2.0 engine.

Standardizes tool and repository resource interfaces according to the Anthropic MCP specification.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from .agent_state import AgentState
from .engineering_tools import (
    EngineeringToolContext,
    create_engineering_tool_registry,
)
from .ingestion import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from .safety import PathSecurityError, validate_safe_path
from .tools import ToolRegistry, ToolSchema

# MCP Protocol Version negotiated during handshake
MCP_PROTOCOL_VERSION = "2024-11-05"

# Standard JSON-RPC 2.0 Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JSONRPCRequest:
    """A standard JSON-RPC 2.0 request or notification."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None
    jsonrpc: str = "2.0"


class JSONRPCError(Exception):
    """A standard JSON-RPC 2.0 error object that can be raised as an exception."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"JSON-RPC Error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass(frozen=True)
class JSONRPCResponse:
    """A standard JSON-RPC 2.0 response."""

    id: str | int | None
    result: Any | None = None
    error: JSONRPCError | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            res["error"] = self.error.to_dict()
        else:
            res["result"] = self.result
        return res


@dataclass(frozen=True)
class MCPTool:
    """An MCP-standard tool descriptor."""

    name: str
    description: str
    inputSchema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPResource:
    """An MCP-standard resource descriptor for repository files."""

    uri: str
    name: str
    description: str | None = None
    mimeType: str | None = None


class MCPError(Exception):
    """Exception raised by MCPClient when a JSON-RPC error is returned."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(f"MCP Error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


# ---------------------------------------------------------------------------
# MIME Helper
# ---------------------------------------------------------------------------


def guess_mime_type(path: str | Path) -> str:
    """Guess MIME type for code files, falling back to clean text types."""
    ext = Path(path).suffix.lower()
    known = {
        ".py": "text/x-python",
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".toml": "application/toml",
        ".html": "text/html",
        ".css": "text/css",
        ".js": "text/javascript",
        ".ts": "text/typescript",
        ".sql": "text/x-sql",
    }
    if ext in known:
        return known[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "text/plain"


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class MCPServer:
    """Model Context Protocol (MCP) server exposing tools and repository resources.

    Supports JSON-RPC 2.0 over standard I/O, direct in-memory calls, or HTTP.
    """

    def __init__(
        self,
        tool_context: EngineeringToolContext | None = None,
        tool_registry: ToolRegistry | None = None,
        repo_id: str = "default",
        read_only: bool = False,
        server_name: str = "AISoftwareEngineeringAgentServer",
        server_version: str = "0.1.0",
    ) -> None:
        self.tool_context = tool_context
        self.repo_id = repo_id
        self.read_only = read_only
        self.server_name = server_name
        self.server_version = server_version
        self.is_initialized = False

        # If a registry is provided directly, use it; otherwise build from tool_context
        if tool_registry is not None:
            self.tool_registry = tool_registry
        elif tool_context is not None:
            self.tool_registry = create_engineering_tool_registry(tool_context)
        else:
            self.tool_registry = ToolRegistry()

    async def handle_request(self, raw_request: dict[str, Any] | str) -> dict[str, Any] | None:
        """Handle a single JSON-RPC 2.0 message and return response dict (or None for notifications)."""
        # 1. Parse payload
        if isinstance(raw_request, str):
            try:
                payload = json.loads(raw_request)
            except (json.JSONDecodeError, ValueError) as exc:
                return JSONRPCResponse(
                    id=None,
                    error=JSONRPCError(code=PARSE_ERROR, message=f"Parse error: {exc}"),
                ).to_dict()
        elif isinstance(raw_request, dict):
            payload = raw_request
        else:
            return JSONRPCResponse(
                id=None,
                error=JSONRPCError(code=INVALID_REQUEST, message="Invalid Request: payload must be a JSON object."),
            ).to_dict()

        # 2. Validate JSON-RPC 2.0 envelope
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "method" not in payload:
            req_id = payload.get("id") if isinstance(payload, dict) else None
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=INVALID_REQUEST, message="Invalid Request: 'jsonrpc': '2.0' and 'method' are required."),
            ).to_dict()

        method = payload.get("method")
        params = payload.get("params") or {}
        req_id = payload.get("id")
        is_notification = req_id is None

        if not isinstance(params, dict):
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=INVALID_PARAMS, message="Invalid params: expected JSON object."),
            ).to_dict()

        # 3. Dispatch method
        try:
            result = await self._dispatch(method, params)
            if is_notification:
                return None
            return JSONRPCResponse(id=req_id, result=result).to_dict()
        except JSONRPCError as rpc_err:
            if is_notification:
                return None
            return JSONRPCResponse(id=req_id, error=rpc_err).to_dict()
        except Exception as exc:
            if is_notification:
                return None
            return JSONRPCResponse(
                id=req_id,
                error=JSONRPCError(code=INTERNAL_ERROR, message=f"Internal server error: {exc}"),
            ).to_dict()

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """Internal router for MCP protocol methods."""
        if method == "initialize":
            return self._handle_initialize(params)
        elif method == "notifications/initialized":
            self.is_initialized = True
            return {}
        elif method == "ping":
            return {}
        elif method == "tools/list":
            return self._handle_tools_list(params)
        elif method == "tools/call":
            return await self._handle_tools_call(params)
        elif method == "resources/list":
            return self._handle_resources_list(params)
        elif method == "resources/read":
            return self._handle_resources_read(params)
        else:
            raise JSONRPCError(code=METHOD_NOT_FOUND, message=f"Method not found: '{method}'.")

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Respond to client handshake with protocol capabilities and version."""
        self.is_initialized = True
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version,
            },
        }

    def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """List registered tools formatted with MCP JSON Schema inputSchema."""
        tools_list: list[dict[str, Any]] = []
        # In read-only mode, exclude execution tools that run code or tests
        prohibited_in_readonly = {"run_python", "run_tests"}

        for schema in self.tool_registry.list_schemas():
            if self.read_only and schema.name in prohibited_in_readonly:
                continue
            # Also exclude final_answer internal agent termination tool from external MCP listing
            if schema.name == "final_answer":
                continue

            tools_list.append(
                {
                    "name": schema.name,
                    "description": schema.description,
                    "inputSchema": schema.parameters or {"type": "object", "properties": {}},
                }
            )

        return {"tools": tools_list}

    async def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool safely and return MCP content block."""
        name = params.get("name")
        if not name or not isinstance(name, str):
            raise JSONRPCError(code=INVALID_PARAMS, message="Invalid params: 'name' string is required for tools/call.")

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise JSONRPCError(code=INVALID_PARAMS, message="Invalid params: 'arguments' must be a dictionary.")

        if self.read_only and name in {"run_python", "run_tests"}:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Permission Denied: '{name}' is disabled because the MCP server is in read-only mode.",
                    }
                ],
                "isError": True,
            }

        handler = self.tool_registry.get(name)
        if handler is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Tool Not Found: Tool '{name}' is not registered on this MCP server.",
                    }
                ],
                "isError": True,
            }

        call_args = dict(arguments)
        if name == "read_file" and "file_path" not in call_args and "path" in call_args:
            call_args["file_path"] = call_args["path"]
        elif name == "read_file" and "path" not in call_args and "file_path" in call_args:
            call_args["path"] = call_args["file_path"]
        if name == "list_files" and "path" not in call_args and "directory" in call_args:
            call_args["path"] = call_args["directory"]

        state = AgentState(task=f"MCP call: {name}")
        try:
            output = await handler(call_args, state)
            # Detect whether output indicates an error/violation
            is_err = any(
                output.startswith(prefix)
                for prefix in (
                    "Path Traversal Error:",
                    "Security Error:",
                    "Path Security Error:",
                    "Command Security Error:",
                    "Subprocess Error:",
                    "Execution Error:",
                    "Pytest Error:",
                    "Git Error:",
                    "Error:",
                )
            )
            return {
                "content": [{"type": "text", "text": output}],
                "isError": is_err,
            }
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": f"Execution Error: {exc}"}],
                "isError": True,
            }

    def _handle_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """List repository files as addressable MCP resources."""
        resources: list[dict[str, Any]] = []
        if self.tool_context is None or not self.tool_context.repo_root.exists():
            return {"resources": resources}

        root = self.tool_context.repo_root
        max_files = 1000

        for current_dir, dirnames, filenames in os.walk(root):
            # Prune excluded directories in-place
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDED_DIRS and not d.startswith(".")]

            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if ext in DEFAULT_EXCLUDED_EXTENSIONS:
                    continue

                abs_path = Path(current_dir) / filename
                try:
                    rel_path = abs_path.relative_to(root)
                except ValueError:
                    continue

                uri = f"repo://{self.repo_id}/{rel_path.as_posix()}"
                resources.append(
                    {
                        "uri": uri,
                        "name": rel_path.as_posix(),
                        "description": f"Repository file at {rel_path.as_posix()}",
                        "mimeType": guess_mime_type(abs_path),
                    }
                )

                if len(resources) >= max_files:
                    break
            if len(resources) >= max_files:
                break

        return {"resources": resources}

    def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read a repository resource by URI with path safety containment checks."""
        uri = params.get("uri")
        if not uri or not isinstance(uri, str):
            raise JSONRPCError(code=INVALID_PARAMS, message="Invalid params: 'uri' string is required for resources/read.")

        prefix = f"repo://{self.repo_id}/"
        alt_prefix = "repo://"

        if uri.startswith(prefix):
            rel_path_str = uri[len(prefix):]
        elif uri.startswith(alt_prefix):
            parts = uri[len(alt_prefix):].split("/", 1)
            rel_path_str = parts[1] if len(parts) > 1 else ""
        else:
            raise JSONRPCError(
                code=INVALID_PARAMS,
                message=f"Unsupported resource URI scheme: '{uri}'. Expected 'repo://{self.repo_id}/<path>'.",
            )

        if self.tool_context is None or not self.tool_context.repo_root.exists():
            raise JSONRPCError(code=INTERNAL_ERROR, message="No repository root context is configured on this MCP server.")

        try:
            safe_file = validate_safe_path(
                target_path=rel_path_str,
                repo_root=self.tool_context.repo_root,
                allowed_roots=self.tool_context.allowed_roots,
            )
        except PathSecurityError as sec_err:
            raise JSONRPCError(code=INVALID_PARAMS, message=f"Path Security Error: {sec_err}")

        if not safe_file.exists():
            raise JSONRPCError(code=INVALID_PARAMS, message=f"Resource not found: file '{rel_path_str}' does not exist.")

        if safe_file.is_dir():
            raise JSONRPCError(code=INVALID_PARAMS, message=f"Resource error: '{rel_path_str}' is a directory, not a file.")

        try:
            content = safe_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise JSONRPCError(code=INTERNAL_ERROR, message=f"Failed to read resource '{rel_path_str}': {exc}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": guess_mime_type(safe_file),
                    "text": content,
                }
            ]
        }


# ---------------------------------------------------------------------------
# MCP Client
# ---------------------------------------------------------------------------


class MCPClient:
    """Client for connecting to and invoking an MCP Server over in-memory, stdio, or HTTP transport."""

    def __init__(
        self,
        server: MCPServer | None = None,
        transport_sender: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]] | None = None,
    ) -> None:
        if server is None and transport_sender is None:
            raise ValueError("Either 'server' or 'transport_sender' must be provided to MCPClient.")
        self.server = server
        self.transport_sender = transport_sender
        self._request_counter = 1

    async def _send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC 2.0 request and return the unwrapped result, or raise MCPError."""
        req_id = self._request_counter
        self._request_counter += 1

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }

        if self.transport_sender is not None:
            response = await self.transport_sender(payload)
        elif self.server is not None:
            response = await self.server.handle_request(payload)
        else:
            raise MCPError(INTERNAL_ERROR, "No transport available for MCPClient.")

        if response is None:
            return None

        if "error" in response and response["error"] is not None:
            err = response["error"]
            raise MCPError(
                code=err.get("code", INTERNAL_ERROR),
                message=err.get("message", "Unknown MCP error"),
                data=err.get("data"),
            )

        return response.get("result")

    async def initialize(
        self,
        client_name: str = "AISoftwareEngineeringAgentClient",
        client_version: str = "0.1.0",
    ) -> dict[str, Any]:
        """Perform the MCP capabilities handshake with the server."""
        init_params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": client_version},
        }
        res = await self._send("initialize", init_params)
        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        if self.transport_sender is not None:
            await self.transport_sender(notification)
        elif self.server is not None:
            await self.server.handle_request(notification)
        return res

    async def ping(self) -> bool:
        """Ping the MCP server."""
        await self._send("ping", {})
        return True

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools exposed by the MCP server."""
        result = await self._send("tools/list", {})
        tools_data = (result or {}).get("tools", [])
        return [
            MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                inputSchema=t.get("inputSchema", {}),
            )
            for t in tools_data
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> tuple[str, bool]:
        """Execute a tool over MCP, returning (output_text, is_error)."""
        params = {"name": name, "arguments": arguments or {}}
        result = await self._send("tools/call", params)
        content_items = (result or {}).get("content", [])
        is_error = bool((result or {}).get("isError", False))

        text_parts: list[str] = []
        for item in content_items:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "\n".join(text_parts), is_error

    async def list_resources(self) -> list[MCPResource]:
        """Discover repository resources exposed by the MCP server."""
        result = await self._send("resources/list", {})
        res_data = (result or {}).get("resources", [])
        return [
            MCPResource(
                uri=r["uri"],
                name=r["name"],
                description=r.get("description"),
                mimeType=r.get("mimeType"),
            )
            for r in res_data
        ]

    async def read_resource(self, uri: str) -> str:
        """Fetch content of an MCP repository resource."""
        result = await self._send("resources/read", {"uri": uri})
        contents = (result or {}).get("contents", [])
        if not contents:
            return ""
        return contents[0].get("text", "")


# ---------------------------------------------------------------------------
# MCP Tool Adapter for ReAct Agent
# ---------------------------------------------------------------------------


class MCPToolAdapter:
    """Adapts tools discovered on an MCP server into an Agent's native ToolRegistry."""

    @staticmethod
    async def adapt_mcp_to_registry(
        client: MCPClient,
        registry: ToolRegistry,
        prefix: str = "",
        include_tools: Sequence[str] | None = None,
    ) -> list[str]:
        """Discover tools from an MCPClient and register them into the given ToolRegistry."""
        tools = await client.list_tools()
        registered_names: list[str] = []

        for tool in tools:
            if include_tools is not None and tool.name not in include_tools:
                continue

            full_name = f"{prefix}{tool.name}"
            schema = ToolSchema(
                name=full_name,
                description=tool.description,
                parameters=tool.inputSchema or {"type": "object", "properties": {}},
            )

            # Define closure capturing the specific tool name
            def make_handler(target_tool_name: str) -> Callable[[dict[str, Any], AgentState], Awaitable[str]]:
                async def _handler(arguments: dict[str, Any], state: AgentState) -> str:
                    try:
                        output, is_error = await client.call_tool(target_tool_name, arguments)
                        return output
                    except MCPError as mcp_err:
                        return f"MCP Tool Error ({target_tool_name}): {mcp_err.message}"
                    except Exception as exc:
                        return f"MCP Client Error ({target_tool_name}): {exc}"

                return _handler

            registry.register(name=full_name, handler=make_handler(tool.name), schema=schema)
            registered_names.append(full_name)

        return registered_names


# ---------------------------------------------------------------------------
# Stdio Server Transport Loop
# ---------------------------------------------------------------------------


async def run_stdio_server(
    server: MCPServer,
    reader: asyncio.StreamReader | None = None,
    writer: asyncio.StreamWriter | None = None,
) -> None:
    """Run an MCP server over stdio for CLI, Cursor, and Claude Desktop integrations."""
    loop = asyncio.get_event_loop()

    # If stream reader/writer are not supplied, wrap sys.stdin/stdout via connect_read_pipe
    if reader is None:
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        raw_text = line.decode("utf-8", errors="replace").strip()
        if not raw_text:
            continue

        response = await server.handle_request(raw_text)
        if response is not None:
            out_bytes = (json.dumps(response) + "\n").encode("utf-8")
            if writer is not None:
                writer.write(out_bytes)
                await writer.drain()
            else:
                sys.stdout.write(out_bytes.decode("utf-8"))
                sys.stdout.flush()
