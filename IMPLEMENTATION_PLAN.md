# Implementation Plan: Agent Loop, State, and Tool Selection

This document records the design and architecture implemented across milestones up to and including the Day 7 milestone (completed 2026-08-27).

---

## Day 5-6 Milestone: Retrieval Quality, Hybrid Search, and Verified Citations

### Goal

Enhance retrieval accuracy, keyword recall, and citation precision across codebases:
1. **BM25 Lexical Baseline**: Native code tokenizer handling `snake_case`, `camelCase`, and symbol identifiers paired with an Okapi BM25 inverted index.
2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Merging dense vector semantic retrieval with sparse lexical retrieval ($k=60$) to capture conceptual descriptions and exact function/class names simultaneously.
3. **Metadata Filtering**: Scoping searches by repository ID, file path glob patterns, programming language, or symbol presence.
4. **Symbol Boost Reranker**: Heuristic reranking layer prioritizing exact symbol matches, method names, and matching file paths.
5. **Query Expansion**: Decomposing domain acronyms (e.g. `YTM`, `NPV`, `bootstrap`) into domain synonyms and code terms.
6. **Citation Source Validation**: Auditing LLM citations to verify line-range overlap against retrieved chunks and flagging unverified/hallucinated source references.
7. **Debug Inspection**: Detailed ranking trace in the API response showing individual dense/BM25 ranks, raw similarity scores, and fusion scores.

### Decisions

| Area | Decision | Why |
| --- | --- | --- |
| Lexical Engine | Native Python `BM25Index` + `CodeTokenizer` | Eliminates external infrastructure (e.g. Elasticsearch) while maintaining sub-millisecond keyword lookups and full offline testability. |
| Fusion Algorithm | Reciprocal Rank Fusion (RRF: $\frac{1}{60 + \text{rank}}$) | Distribution-agnostic score fusion that seamlessly blends bounded cosine similarities $[0, 1]$ and unbounded BM25 scores $[0, \infty)$. |
| Citation Verification | Interval overlap check (`max(start1, start2) <= min(end1, end2)`) against retrieved chunks | Flags hallucinations when the LLM makes assertions or quotes files not in the retrieved context. |
| API Options | `retrieval_strategy` (`dense`, `bm25`, `hybrid`), `path_patterns`, `languages`, `symbol_only`, `include_debug_info` | Provides developers and agents granular control over search boundaries and inspectable retrieval traces. |

### Architecture

```text
User Query: "How is zero_rate calculated in YieldCurve?"
                      |
        +-------------+-------------+
        | Query Analysis & Expansion|
        +-------------+-------------+
                      |
        +-------------+---------------------------+
        |                                         |
        v (Dense Path)                            v (Lexical Path)
+-----------------------+               +-----------------------+
|  Embedding Client     |               |  BM25 Lexical Index   |
|  Dense Vector Search  |               |  Identifier Matching  |
+-----------+-----------+               +-----------+-----------+
            |                                       |
            +-------------------+-------------------+
                                |
                                v
                +-------------------------------+
                |   Hybrid Fusion (RRF Engine)  |
                |  Combines Dense + BM25 Ranks  |
                +---------------+---------------+
                                |
                                v
                +-------------------------------+
                |   Metadata Filter & Reranker  |
                | (Path, Language, Symbol Boost)|
                +---------------+---------------+
                                |
                                v
                +-------------------------------+
                |   Grounded LLM Prompting      |
                |   + Citation Source Validator |
                +-------------------------------+
```

### Implemented Components

1. **`models.py`**:
   - `MetadataFilter`, `RetrievalStrategy`, `RetrievalDebugInfo`, updated `RetrievalResult`, updated `Citation` (`is_verified`).
2. **`lexical.py`**:
   - `CodeTokenizer`: Splits on underscores, camelCase transitions, symbol separators, and filters general stopwords while preserving code identifiers.
   - `BM25Index`: In-memory Okapi BM25 inverted index supporting incremental indexing and metadata filters.
3. **`retrieval.py`**:
   - `reciprocal_rank_fusion`: Rank aggregation formula merging disparate retrieval engines.
   - `SymbolBoostReranker`: Exact and partial symbol definition boosts (+0.25 exact symbol, +0.08 file path).
   - `QueryExpander`: Synonym and acronym expansion.
   - `HybridRetriever`: High-level retriever supporting `dense`, `bm25`, and `hybrid` strategies.
4. **`rag.py`**:
   - Integrated dual-indexing into `ingest_repository` (both VectorStore and BM25).
   - `validate_citations`: Line-range overlap verification.
5. **`app.py`**:
   - Enhanced `POST /v1/rag/query` with strategy selection, path filters, language filters, and debug metrics.

---

## Day 7 Milestone: Agent Loop And State

### Goal

Implement a minimal, inspectable ReAct-style agent loop that:
1. Maintains typed state across reasoning and tool-use steps.
2. Uses the LLM to select tools at each step via `<tool_call>` XML tags in free-form text.
3. Terminates cleanly on a final answer, step budget exhaustion, or LLM error.
4. Exposes a full step trace in every response so every reasoning/action/observation is auditable.
5. Wires the agent into the FastAPI surface as `POST /v1/agent/run`.

### Decisions

| Area | Decision | Why |
| --- | --- | --- |
| Tool invocation protocol | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` XML tags in plain LLM text | Provider-neutral: works with any LLM that produces text, no SDK function-calling feature required. Parsed with a single regex in `Agent._parse_llm_response`. |
| State mutability | `AgentState` is a mutable class; `AgentStep`, `ToolCall`, `Observation`, `AgentResult` are frozen dataclasses | State needs to accumulate steps; individual records should be immutable once written to the trace. |
| Step budget check placement | `has_budget()` checked at the top of each loop iteration before the LLM call | Ensures the agent never starts a new LLM call it cannot finish, and the error message is attributed to the loop controller rather than a tool. |
| `final_answer` tool | Registered tool that calls `state.finish()` and terminates the loop | Uniform mechanism: the LLM decides when it is done by calling a tool, same as any other action. |
| `already_done` guard | Check `state.is_done()` before calling `add_observation_step()` | `add_observation_step()` unconditionally sets `status = OBSERVING`. Without the guard, calling it after `final_answer` (which sets `status = FINISHED`) would mask the terminal state and prevent the loop from breaking. |
| Tool registry injection | `ToolRegistry` passed into `Agent.__init__`; `create_default_tool_registry(rag_service)` factory builds the default set | Keeps the agent testable with minimal registries and allows future tools to be added without touching the loop. |
| Endpoint placement | `POST /v1/agent/run` added to `app.py`; lazily wires LLM client, RAG service, and tool registry on first call | Consistent with existing lazy-initialisation pattern in the app; no new infrastructure needed. |

### Architecture

```text
POST /v1/agent/run  {"task": "...", "max_steps": 10}
           |
           v
      Agent.run(task)
           |
     ┌─────┴──────────────────────────────────┐
     │           ReAct Loop                   │
     │                                        │
     │  ┌─ THINK ──────────────────────────┐  │
     │  │  Format conversation + history   │  │
     │  │  LLM.generate(prompt)            │  │
     │  │  Parse response for <tool_call>  │  │
     │  └──────────────┬───────────────────┘  │
     │                 │                      │
     │         tool call?                     │
     │        /         \                     │
     │      yes          no                   │
     │       │            │                   │
     │  ┌─ ACT ─┐   ┌─ FINISH ─┐             │
     │  │ record│   │ direct   │             │
     │  │ step  │   │ answer   │             │
     │  └───┬───┘   └──────────┘             │
     │      │                                │
     │  ┌─ OBSERVE ──────────────────────┐   │
     │  │  ToolRegistry.execute(call)    │   │
     │  │  record observation            │   │
     │  │  check is_done() → break?      │   │
     │  └────────────────────────────────┘   │
     │                                        │
     │  if steps >= max_steps → ERROR + break │
     └─────────────────────────────────────── ┘
           |
           v
      AgentResult  {answer, status, steps[], citations[], duration_ms}
           |
           v
      AgentRunResponse  (serialised trace → JSON)
```

### Implemented Components

1. **`agent_state.py`**:
   - `AgentStatus` enum: `IDLE`, `THINKING`, `ACTING`, `OBSERVING`, `FINISHED`, `ERROR`.
   - `ToolCall`: frozen dataclass — tool name, arguments dict, unique call ID.
   - `Observation`: frozen dataclass — call ID, tool name, content, `is_error`, `duration_ms`.
   - `AgentStep`: mutable record of a single loop iteration phase.
   - `AgentState`: mutable accumulator — task, step list, status, current step counter, final answer, citations, start time.
   - `AgentResult`: frozen summary returned to the caller after the loop exits.

2. **`tools.py`**:
   - `ToolSchema`: JSON-schema-style descriptor rendered into the LLM system prompt.
   - `ToolRegistry`: maps tool names → async handler functions; formats all schemas as a prompt block; `execute()` wraps handlers with timing and error capture.
   - Built-in handlers: `search_code` (calls `RAGService.retrieve`), `answer_query` (calls `RAGService.answer_query` and stores citations in state), `final_answer` (calls `state.finish()` to terminate the loop).
   - `create_default_tool_registry(rag_service)`: factory wiring all three default tools.

3. **`agent.py`** (ReAct agent additions):
   - `Agent`: takes `llm_client`, `tool_registry`, `settings`, optional `max_steps` override.
   - `Agent.run(task)`: full ReAct loop — Think → Act → Observe → … → Final Answer.
   - `Agent._build_system_prompt()`: embeds tool schemas and ReAct instructions.
   - `Agent._format_conversation(state)`: renders the full step history as a prompt string.
   - `Agent._parse_llm_response(text)`: extracts `ToolCall` from `<tool_call>` tags or returns plain text as a direct answer.
   - Bug fix: `already_done` guard prevents `add_observation_step()` from overwriting `FINISHED` status set by `final_answer`.

4. **`app.py`** (new endpoint):
   - `AgentRunRequest`: `task` (1–10 000 chars), `max_steps` (1–30, default 10).
   - `AgentRunResponse`: `request_id`, `task`, `answer`, `status`, `total_steps`, `duration_ms`, `steps[]`, `citations[]`.
   - `AgentStepModel`: per-step view with `step_number`, `status`, `reasoning`, `tool_name`, `tool_arguments`, `observation`, `is_error`.
   - `POST /v1/agent/run`: validates input, builds agent with injected dependencies, runs loop, serialises full trace.

5. **`tests/test_agent_loop.py`** (27 tests):
   - `AgentState` unit tests: initial conditions, step recording, finish, fail, budget exhaustion.
   - `ToolRegistry` unit tests: register/retrieve, unknown tool error, handler exception capture, prompt formatting.
   - `Agent._parse_llm_response` unit tests: tool call extraction, plain text passthrough, malformed JSON fallback.
   - `Agent.run` integration tests: direct answer, `final_answer` tool, custom tool with trace inspection, full trace field coverage, step limit enforcement, empty task rejection, unknown tool error observation.
   - `POST /v1/agent/run` endpoint tests: happy path, trace structure, empty task 422, `max_steps` enforcement, no API key 503.

---

## Day 8-9 Milestone: Safe Engineering Tools (Completed)

### Goal

Implement a secure, sandboxed suite of software engineering tools that allow the autonomous agent to inspect, navigate, execute, test, and diff code within approved repository boundaries.

### Tools Specification

1. **`list_files`**: Directory exploration scoped to target repository roots with exclusions (`.git`, `.venv`, `__pycache__`, etc.) and entry caps.
2. **`read_file`**: Safe file reading with optional 1-indexed line slicing, path traversal defense, and binary/size protection.
3. **`search_code`**: Fast text and regex grep across repository source files with path pattern filtering.
4. **`run_python`**: Isolated Python subprocess execution using `sys.executable -c <code>`, strict timeouts (default 10s), output truncation (8,000 chars), and clean environment.
5. **`run_tests`**: Test suite execution via `pytest` with argument whitelisting, timeout enforcement, and structured pass/fail summaries.
6. **`git_diff`**: Read-only git diff viewer for working tree and commit changes with git ref sanitization.

### Security & Safety Boundaries

| Safety Feature | Implementation |
| --- | --- |
| Path Traversal Defense | `validate_safe_path`: canonical path resolution, parentage validation within `repo_root`, and symlink escaping checks. |
| Subprocess Isolation | Direct invocation via `sys.executable` with `shell=False`, sanitized args, and working directory pinned to `repo_root`. |
| Execution Timeouts | Process-level timeouts (10s–60s) preventing hung scripts or infinite loops. |
| Output Truncation | `truncate_output` caps stdout/stderr at 8,000 characters to prevent memory blowups and context overflow. |
| Git Ref Sanitization | Strict alphanumeric/ref pattern check avoiding command injection. |
| Tool Result Normalization | Unified `Observation` formatting with error indicators, execution timing, and clean LLM-friendly messages. |

---

## Day 10 Milestone: Model Context Protocol (MCP)

### Goal

Standardize tool and resource interfaces using Anthropic's **Model Context Protocol (MCP)** JSON-RPC 2.0 specification:
1. **`MCPServer`**: Exposes our safe engineering tools as standardized MCP tools (`tools/list`, `tools/call`) and codebase files as MCP resources (`resources/list`, `resources/read` under `repo://` URIs).
2. **`MCPClient` & `MCPToolAdapter`**: Allows our autonomous ReAct `Agent` to connect to internal or external MCP servers, discover tools dynamically, and register them into its `ToolRegistry`.
3. **Multi-Transport Support**:
   - In-memory transport: Zero-overhead direct dispatch for ultrafast execution and offline unit testing.
   - Stdio transport: Standard I/O loop for CLI / IDE integrations (Claude Desktop, Cursor, Antigravity IDE).
   - HTTP transport: FastAPI endpoint (`POST /v1/mcp`) for remote JSON-RPC 2.0 integration.
4. **Standardized Protocol Foundations**: Custom zero-dependency JSON-RPC 2.0 protocol engine enforcing standard error codes (`-32700`, `-32600`, `-32601`, `-32602`, `-32603`) and schema compatibility.

### Architecture

```text
       External Clients (Claude Desktop, Cursor, HTTP)
                           |
                           | JSON-RPC 2.0 (stdio or POST /v1/mcp)
                           v
                     +-----------+
                     | MCPServer |
                     +-----+-----+
                           |
         +-----------------+-----------------+
         |                                   |
         v (tools/list, tools/call)          v (resources/list, resources/read)
+-----------------------+           +-----------------------+
| Safe Engineering Tools|           | Repository Resources  |
| (run_tests, search,   |           | (repo://{repo_id}/    |
|  run_python, etc.)    |           |  {file_path})         |
+-----------+-----------+           +-----------------------+
            ^
            | executes
    +-------+-------+
    | MCPToolAdapter|
    +-------+-------+
            ^
            | adapts
    +-------+-------+
    |   MCPClient   |
    +-------+-------+
            ^
            | invokes
    +-------+-------+
    |  ReAct Agent  |
    +---------------+
```

### Protocol Details

| Method | Direction | Description |
| --- | --- | --- |
| `initialize` | Client -> Server | Protocol capabilities handshake (`tools`, `resources`) and protocol version negotiation (`2024-11-05`). |
| `notifications/initialized` | Client -> Server | Notification confirming client readiness. |
| `ping` | Client -> Server | Connection liveness probe. |
| `tools/list` | Client -> Server | Enumerates available tools with their JSON Schema `inputSchema`. |
| `tools/call` | Client -> Server | Executes a named tool with validated arguments, returning text content blocks and `isError` flag. |
| `resources/list` | Client -> Server | Enumerates readable codebase files with `repo://` URIs and MIME types. |
| `resources/read` | Client -> Server | Reads file content for a given `repo://` URI, subject to path containment checks. |



---

## Day 10 Milestone: Model Context Protocol (MCP)

### Goal

Standardize tool and repository resource interfaces using the Anthropic MCP JSON-RPC 2.0 specification, enabling the agent to both **serve** tools to external clients (Claude Desktop, Cursor, IDEs) and **consume** tools from external MCP servers dynamically.

1. **`MCPServer`**: Exposes safe engineering tools as MCP `tools/list` + `tools/call` and codebase files as MCP `resources/list` + `resources/read` under `repo://` URIs.
2. **`MCPClient` + `MCPToolAdapter`**: Allows the ReAct `Agent` to connect to any MCP server, discover tools, and register them into its `ToolRegistry` — making external tools indistinguishable from internal ones.
3. **Three transports**: In-memory (zero-overhead testing), stdio (CLI/IDE integration), and HTTP (`POST /v1/mcp`).
4. **Self-contained JSON-RPC 2.0 engine**: No external SDK dependency. Custom protocol models enforce standard error codes and response envelope.

### Decisions

| Area | Decision | Why |
| --- | --- | --- |
| Protocol implementation | Self-contained JSON-RPC 2.0 engine (`JSONRPCRequest`, `JSONRPCResponse`, `JSONRPCError`, error codes) without external MCP SDK | Demonstrates deep protocol understanding for interviews; eliminates brittle dependency on an evolving external SDK; enables full offline testing. |
| Dual-mode architecture | Implement both `MCPServer` (expose tools) and `MCPClient` + `MCPToolAdapter` (consume tools) | Covers both sides of the integration: the agent can act as a tool provider for IDEs *and* as a tool consumer for external MCP services. |
| Transport abstraction | `MCPClient` accepts either a direct `MCPServer` reference (in-memory) or a `transport_sender` callable (network/stdio) | Lets tests run at zero latency with real tool execution, while keeping production transports swappable. |
| `read_only` mode | `MCPServer` constructor flag suppresses `run_python` and `run_tests` from `tools/list` and rejects their `tools/call` with `Permission Denied` | Allows safe deployment to untrusted external clients (e.g. a public demo) without refactoring the tool registry. |
| `final_answer` exclusion | `tools/list` never exposes `final_answer` as an external MCP tool | `final_answer` is an internal agent loop termination signal, not a meaningful capability for an external MCP client. |
| `repo://` URI scheme | Resources are addressed as `repo://{repo_id}/{file_path}` | Matches the MCP specification's resource URI convention; `repo_id` scopes resources to a specific ingested repository. |
| `resources/read` path containment | `validate_safe_path` applied to the file path extracted from the URI before reading | Prevents a malicious URI like `repo://id/../../etc/passwd` from escaping the repository root. URI traversal raises `-32602 Invalid params`. |
| HTTP endpoint design | `POST /v1/mcp` accepts repository context via `X-Repository-Path` header or query param; creates a per-request `MCPServer` instance | Stateless per-request design matches FastAPI's concurrency model and allows different repository contexts per call. |

### Architecture

```text
External Clients (Claude Desktop, Cursor, HTTP)
               |
               | JSON-RPC 2.0  (stdio  or  POST /v1/mcp)
               v
         +------------+
         | MCPServer  |
         +-----+------+
               |
     +---------+----------+
     |                    |
     v                    v
tools/list           resources/list
tools/call           resources/read
     |                    |
     v                    v
EngineeringTools      repo:// URIs
(list_files,          (validate_safe_path
 read_file,            → UTF-8 file content
 search_code,          → MIME type)
 run_python,
 run_tests,
 git_diff)
     ^
     | MCPToolAdapter.adapt_mcp_to_registry()
     |
+----------+       +--------------+
| MCPClient| <---> | MCPServer    |   (in-memory or network)
+----------+       +--------------+
     ^
     | tool calls routed through ToolRegistry
     |
+-----------+
| ReAct     |
| Agent     |
+-----------+
```

### Implemented Components

1. **`mcp.py`** (single file, ~640 lines):

   - **Protocol models**: `JSONRPCRequest` (frozen dataclass), `JSONRPCError` (exception with `to_dict()`), `JSONRPCResponse` (frozen dataclass with `to_dict()`). Standard error codes: `PARSE_ERROR=-32700`, `INVALID_REQUEST=-32600`, `METHOD_NOT_FOUND=-32601`, `INVALID_PARAMS=-32602`, `INTERNAL_ERROR=-32603`.
   - **`MCPTool`** / **`MCPResource`** / **`MCPError`**: Typed descriptors and client-side error type.
   - **`guess_mime_type(path)`**: Maps file extensions to accurate MIME types for code files; falls back to `text/plain`.
   - **`MCPServer`**: Accepts `tool_context`, `tool_registry`, `repo_id`, `read_only`, `server_name`, `server_version`. Methods: `handle_request(raw)` → `dict | None`, `_dispatch(method, params)`, `_handle_initialize`, `_handle_tools_list`, `_handle_tools_call`, `_handle_resources_list`, `_handle_resources_read`.
   - **`MCPClient`**: Accepts `server` (in-memory) or `transport_sender` (callable). Methods: `initialize()`, `ping()`, `list_tools() → list[MCPTool]`, `call_tool(name, args) → (text, is_error)`, `list_resources() → list[MCPResource]`, `read_resource(uri) → str`.
   - **`MCPToolAdapter.adapt_mcp_to_registry(client, registry, prefix, include_tools)`**: Discovers tools from client, creates closure-based handlers that delegate back to `client.call_tool()`, and registers them into the agent's `ToolRegistry`. Returns list of registered names.
   - **`run_stdio_server(server, reader, writer)`**: Async line-by-line stdio loop for CLI/IDE integration.

2. **`app.py`** (`POST /v1/mcp` endpoint):
   - Parses JSON-RPC payload from request body (returns `-32700` on parse failure).
   - Reads `X-Repository-Path` header / `repository_path` query param to build `EngineeringToolContext`.
   - Calls `validate_repository_path` against `allowed_repository_roots` (returns `-32602` on violation).
   - Creates a per-request `MCPServer` and calls `server.handle_request(raw_body)`.
   - Returns `HTTP 204` for notifications (no-id requests), `HTTP 200` with JSON-RPC response otherwise.

3. **`tests/test_mcp.py`** (22 tests across 5 layers):
   - **Protocol layer** (5 tests): parse error, invalid request (missing `method`, wrong `jsonrpc`), method not found, invalid params type, initialize + ping.
   - **Notifications** (1 test): confirms `id=None` requests return `None` response and set `is_initialized`.
   - **Tool discovery and execution** (7 tests): `tools/list` schema validation, `list_files` call, `read_file` call (alias mapping `path`→`file_path`), `run_python` execution, path traversal returns `isError: True` with `"Security Error"`, unknown tool returns `isError: True`, read-only mode omits execution tools and blocks their calls.
   - **Resource interface** (3 tests): `resources/list` with MIME types and `.git` exclusion, `resources/read` by URI, path traversal in URI returns `-32602`.
   - **Client workflow** (1 test): full `MCPClient` lifecycle — initialize → ping → list_tools → call_tool → list_resources → read_resource.
   - **Agent integration** (1 test): `MCPToolAdapter` adapts `read_file` from an `MCPClient` into a `ToolRegistry`; a `SequentialFakeLLM`-driven `Agent` calls it and produces the correct answer with the tool visible in the trace.
   - **FastAPI endpoint** (3 tests): HTTP initialize handshake, HTTP tool call with `X-Repository-Path` header, HTTP parse error on non-JSON body.
   - **`guess_mime_type`** (1 test): correct types for `.py`, `.json`, `.md`, `.yaml`, `.sql`, unknown extension.

### Test Bug Fixes Applied

Two bugs in `test_mcp.py` were fixed to align assertions with actual implementation behaviour:

| Test | Bug | Fix |
| --- | --- | --- |
| `test_mcp_tools_call_safety_violation` | Asserted `"Path Traversal Error"` in content text, but `engineering_tools.py` returns `"Security Error: ..."` prefix | Changed assertion to `"Security Error"` |
| `test_mcp_tool_adapter_with_agent` | Used `@dataclass` decorator inside function body without importing `dataclass`; asserted `result.status.value == "FINISHED"` but `AgentResult.status` is a plain string `"finished"`; `Agent()` called without required `settings` kwarg | Replaced with plain class, fixed status assertion to `result.status == "finished"`, added `settings=Settings(...)` |

---

## Next Milestone: Day 11 — Evaluation Pipeline

1. Create 20–30 labelled evaluation questions against Target Repo 1 (bond valuation codebase).
2. Separate retrieval evaluation (Recall@K, MRR, Precision@K) from answer evaluation.
3. Implement answer evaluation rubric: relevance, faithfulness, citation quality, hallucination rate.
4. Run baseline evaluation and document results.
5. Identify and analyse at least 3 failure cases; create improvement tasks.
6. Make evaluation repeatable with a single command.
