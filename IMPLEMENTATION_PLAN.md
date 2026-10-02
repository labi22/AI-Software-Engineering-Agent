# Implementation Plan: Agent Loop, State, and Tool Selection

This document records the design and architecture implemented across milestones up to and including the Day 7 milestone (completed 2026-08-27).

---

## Production-Readiness Backlog (Post-Day 11)

### Goal

Evolve the local, learning-focused agent into a production-shaped, portfolio-ready
service without claiming an unsupported production SLA. The initial deployment will
use Groq as the low-cost inference provider; provider quotas and outages remain an
explicit system constraint rather than something the application hides.

### Delivery Sequence

1. **LangGraph-only orchestration (current step).** Remove the custom ReAct loop
   from the HTTP runtime, configuration, and public request contract. Retain its
   reusable typed tool/result models; retain the original loop only as a documented
   learning reference until the later code-retirement task is complete.
2. **Secure remote GitHub ingestion.** Accept an explicit repository URL and ref,
   shallow-clone into an isolated temporary workspace, apply protocol/size/file-count
   limits, treat repository content as untrusted, and clean up after ingestion.
3. **Multi-language source parsing.** Introduce Tree-sitter-backed parsing for
   Python, TypeScript/JavaScript, Java, Go, and C#, with a safe text-chunking fallback.
4. **Authentication and tenancy.** Add authenticated identities, organisation/user
   ownership to every durable record, and server-side authorisation at every resource
   boundary.
5. **Durable API and worker architecture.** Make long-running ingestion asynchronous
   and persist repository, ingestion-job, document, agent-run, and checkpoint state.
6. **Deployment hardening.** Add Postgres/pgvector, Redis, idempotency records,
   rate and concurrency limits, bounded retries, caching, observability, CI/CD, and
   load/failure testing.

### Target Production Architecture

```text
Client
  -> FastAPI (authentication, validation, rate limit, idempotency)
      -> LangGraph agent run (Groq model + safe tools + durable checkpoint)
      -> PostgreSQL / pgvector (tenant-scoped durable data)
      -> Redis (rate limits, idempotency, cache, coordination)
      -> Worker queue (clone, parse, embed, index)
```

### Key Design Decisions

| Area | Decision | Reason |
| --- | --- | --- |
| Runtime orchestrator | LangGraph only | A single production path avoids duplicated security, tracing, retry, and correctness work. |
| Custom ReAct loop | Learning/reference artifact, not a deployed engine | It demonstrates fundamentals without creating an unsupported public execution mode. |
| Groq | Default low-cost provider behind the existing LLM boundary | Appropriate for demo deployment; quotas and availability require admission control and graceful errors. |
| Parser strategy | Tree-sitter plus a text fallback | Provides consistent multi-language syntax extraction without bespoke parsers per language. |
| Repository trust | Remote repository content is untrusted | Source files can contain prompt injection, oversized artifacts, secrets, or hostile paths. |

### Step 1 Acceptance Checks

- `POST /v1/agent/run` always executes LangGraph.
- The public request no longer accepts an orchestration-engine choice.
- Runtime configuration no longer permits a custom orchestration engine.
- Existing LangGraph direct-answer, tool-call, failure, checkpoint, and API tests pass.
- The custom-loop code remains isolated from the deployed request path pending its
  separately tracked retirement/refactor.

### Step 2: Secure Remote GitHub Ingestion

`POST /v1/repositories/ingest` now accepts exactly one source: a local
`repository_path` or a public `github_url`, plus an optional branch/tag `ref`.
Remote ingestion validates canonical HTTPS `github.com/<owner>/<repository>` URLs,
uses `git clone --depth 1 --filter=blob:none --no-tags --single-branch` with
interactive credentials and Git LFS smudging disabled, and deletes the isolated
temporary checkout after chunks have been indexed. It applies clone timeout,
repository-size, file-count, per-file-size, and aggregate ingestible-source limits.

The URL allowlist and subprocess are deliberately narrow. This is not a general
Git URL fetcher, does not accept credentials or SSH URLs, and does not retain a
working clone for later agent tool execution. Repository source remains untrusted
input and must never be treated as instructions to the system.

### Step 3: Multi-Language Tree-sitter Parsing

Python keeps its standard-library AST parser. JavaScript, TypeScript/TSX, Java,
Go, and C# use individually pinned Tree-sitter grammar bindings supplied through
the optional `parsers` dependency extra. The chunker extracts declarations and
method-qualified symbols, retains uncovered package/import/comment blocks, and
falls back to line-window chunks if a grammar is unavailable or Tree-sitter
reports syntax errors. This preserves successful ingestion of incomplete,
generated, or unsupported source without fabricating symbol metadata.

### Step 4a: Authentication and Tenant Authorization Boundary

The service now supports `AUTH_ENABLED=true` with `X-API-Key` authentication.
Only SHA-256 key digests are configured, and digest comparison uses
constant-time comparison. A principal carries an organization ID; repository
ingestion claims that repository for its organization, and RAG, agent, and MCP
requests are denied unless the caller owns the specified repository. Agent RAG
tools receive a fixed repository filter, so an authorized agent run cannot use
an unfiltered retrieval tool to see other repositories.

With authentication enabled, `DATABASE_URL` is mandatory. The Postgres control
plane idempotently provisions `tenant_organizations`, `tenant_api_keys`, and
`tenant_repositories`, and persists configured key hashes and repository
ownership across restarts and replicas. The in-memory ownership implementation
is retained solely for isolated tests. Ingestion jobs and agent-run records are
not yet durable; they remain explicitly scheduled under the durable API/worker
architecture step.

Remaining identity scope: this release authenticates organization service
principals through API keys. Human-user login, OIDC integration, role-based
permissions, API-key rotation/revocation management endpoints, and database
migration tooling remain future work and must not be represented as implemented.

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

## Day 11 Milestone: Evaluation Pipeline (In Progress)

### Goal

Establish an empirical evaluation harness to measure, benchmark, and improve retrieval and answer generation:
1. **Decoupled Evaluation Pipeline**: Separate **Retrieval Evaluation** (evaluating how well our search engine surfaces relevant code chunks) from **Answer Evaluation** (evaluating answer faithfulness, relevance, and citation precision).
2. **Zero-Cost Retrieval Benchmarking**: Computes standard Information Retrieval (IR) metrics—**Hit@1, Hit@3, Hit@5, Recall@K, Precision@K, and MRR (Mean Reciprocal Rank)**—without requiring any LLM API calls or costs.
3. **Comparative Strategy Evaluation**: Runs side-by-side benchmarks across:
   - Dense Vector Retrieval (semantic search)
   - BM25 Lexical Retrieval (keyword/identifier search)
   - Hybrid Reciprocal Rank Fusion (RRF $k=60$)
   - Hybrid RRF + Symbol Boost Reranker
4. **Hardware-Friendly LLM Configuration**: Adds `llm_base_url` support to allow seamless integration with **Groq Free Tier** (fast cloud inference requiring 0 MB of local RAM) or **Ollama** (offline local inference) for generation evaluation.
5. **Resume-Ready Metrics Table & Failure Analysis**: Generates formatted benchmark reports for portfolio presentation and analyzes 3 edge-case failures.

### Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Golden Benchmark Dataset                    │
│                 (data/eval_questions.json)                  │
│  25 Questions + Ground-Truth Files + Expected Symbols       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────┐
│                 Evaluation Engine (evaluation.py)           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. Retrieval Evaluator (0 LLM calls, $0.00 cost)      │  │
│  │    - Dense Baseline                                   │  │
│  │    - BM25 Baseline                                    │  │
│  │    - Hybrid RRF Baseline                              │  │
│  │    - Hybrid + Symbol Boost Baseline                   │  │
│  │    Computes: Hit@1, Hit@3, Hit@5, Recall@5, MRR       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 2. Answer Evaluator (Optional LLM: Groq / Ollama)     │  │
│  │    - Faithfulness (Citation line-overlap verification)│  │
│  │    - Citation Precision                               │  │
│  │    - Answer Relevance Score                           │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               v
┌─────────────────────────────────────────────────────────────┐
│             Output: Resume-Ready Metrics Table              │
│                 & Failure Analysis Report                   │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval Metrics Formulated

| Metric | Formula | What It Proves in an Interview |
| --- | --- | --- |
| **Hit@K** | $\frac{1}{\|Q\|} \sum_{i=1}^{\|Q\|} \mathbb{I}(\text{relevant chunk in top } K)$ | Baseline check: Did the user find what they were looking for anywhere in the top $K$ results? |
| **Recall@K** | $\frac{\|\text{Retrieved}_K \cap \text{Relevant}\|}{\|\text{Relevant}\|}$ | Completeness: Did the retriever pull in all necessary context needed to answer the question? |
| **Precision@K** | $\frac{\|\text{Retrieved}_K \cap \text{Relevant}\|}{K}$ | Cleanliness: Did the retriever avoid polluting the LLM context window with noisy irrelevant code? |
| **MRR** | $\frac{1}{\|Q\|} \sum_{i=1}^{\|Q\|} \frac{1}{\text{rank}_i}$ | Ranking Quality: How close to the top (rank 1) was the primary ground-truth file? |


---

## Production Readiness Architecture (HLD & LLD)

This section provides the High-Level Design (HLD) and Low-Level Design (LLD) for transforming the codebase into an enterprise-grade, portfolio-ready Agentic AI platform built to withstand production concurrency, provider quota ceilings, and distributed failures.

### High-Level Architecture (HLD)

```text
                                     +-----------------------------------------+
                                     |           API Clients & IDEs            |
                                     +--------------------+--------------------+
                                                          |
                                                          | HTTPS (X-API-Key, Idempotency-Key)
                                                          v
+-----------------------------------------------------------------------------------------------------------------------+
| FastAPI Application Gateway                                                                                            |
|                                                                                                                       |
|  +---------------------------+   +----------------------------+   +----------------------------+                      |
|  | Tenant Auth & Repository  |-->| Rate Limiting & Concurrency|-->| Distributed Idempotency    |                      |
|  | Ownership Verification    |   | (Sliding Window / Leaky)   |   | (In-Progress Lock / Cache) |                      |
|  +---------------------------+   +----------------------------+   +----------------------------+                      |
|                                                |                                                                      |
|                                                v                                                                      |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  | Endpoints Router                                                                                                |  |
|  |  * POST /v1/repositories/ingest/async -> Enqueue background job (returns 202 + job_id)                         |  |
|  |  * GET  /v1/ingestion-jobs/{job_id}   -> Poll job status, worker progress, and chunk metadata                    |  |
|  |  * POST /v1/agent/run                 -> Run LangGraph agent with checkpointing & run persistence               |  |
|  |  * GET  /v1/agent/runs/{run_id}       -> Audit trail & step-by-step trace retrieval                             |  |
|  |  * POST /v1/rag/query                 -> Hybrid RAG with semantic embedding caching                             |  |
|  |  * GET  /health                       -> Multi-tier liveness & readiness check (DB, Redis, LLM provider)        |  |
|  |  * GET  /metrics                      -> Prometheus metrics (latencies, token counters, active runs, queue depth)| |
|  +-----------------------------------------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------------------------------------+
           |                                             |                                           |
           | DB Queries / Enqueue                        | Coordination / Caching                    | Inference
           v                                             v                                           v
+------------------------------------+         +-----------------------+         +-------------------------------------+
| PostgreSQL 16 + pgvector           |         | Redis 7               |         | Groq / LLM Admission Controller     |
|                                    |         |                       |         |                                     |
| * tenant_organizations             |         | * Rate limiter buckets|         | * Token bucket rate pacer           |
| * tenant_api_keys                  |         | * Concurrency leases  |         | * Jittered exponential retry on 429 |
| * tenant_repositories              |         | * Idempotency cache   |         | * Step budget & timeout enforcer    |
| * ingestion_jobs (SKIP LOCKED)     |         | * Query/embedding     |         | * Quota exhaustion graceful fallback|
| * agent_runs (Audit traces)        |         |   cache               |         +-------------------------------------+
| * langgraph_checkpoints (State)    |         +-----------------------+
| * document_chunks (Vectors + Meta) |
+------------------------------------+
           ^
           | Atomic Claims (FOR UPDATE SKIP LOCKED)
           |
+---------------------------------------------------------------------------------------+
| Ingestion Worker Pool (Scale: N replicas)                                             |
|                                                                                       |
|  Loop: claim_next() -> clone repo -> tree-sitter parse -> embed -> store -> finish()  |
+---------------------------------------------------------------------------------------+
```

---

### Phase 5A: Durable Ingestion Jobs & Worker Queue

#### 1. HLD Rationale
Repository cloning (especially remote GitHub repositories), multi-language Tree-sitter parsing, and vector embedding creation are blocking, compute- and network-heavy operations. Running them synchronously inside an HTTP request handler causes:
- HTTP 504 Gateway Timeouts at reverse proxies (Nginx/Cloudflare 30s limits).
- Thread/event loop starvation for incoming API requests.
- Complete data loss and inconsistent state if the web container restarts mid-ingestion.

#### 2. LLD Implementation
- **Queue Engine (`ingestion_jobs.py`)**:
  - `PostgresIngestionJobStore`: Leverages `FOR UPDATE SKIP LOCKED` for atomic, race-free single-item claims across any number of concurrent worker processes.
  - `InMemoryIngestionJobStore`: Lightweight thread-safe in-memory fallback for unit testing and local developer workflows without PostgreSQL.
  - States: `queued` -> `running` -> `succeeded` | `failed`.
- **Worker Runner (`worker.py`)**:
  - Standalone daemon process or scheduled async runner: `run_worker(store, rag_service, settings, stop_event)`.
  - Claims job, sets job to `running`, shallow clones via `cloned_github_repository`, chunks via multi-language chunker, generates embeddings, stores chunks in `VectorStore` + `BM25Index`, and updates job to `succeeded` with summary payload or `failed` with captured error traceback.
- **REST Contract**:
  - `POST /v1/repositories/ingest/async`: Validates request, checks tenant repository ownership, enqueues job into store, returns HTTP 202 Accepted with `{ "job_id": "<uuid>", "status": "queued" }`.
  - `GET /v1/ingestion-jobs/{job_id}`: Checks organization ownership and returns current status, attempts, error message (if any), and ingestion metrics when completed.

---

### Phase 5B: Postgres-Backed LangGraph Checkpoints & Agent-Run Persistence

#### 1. HLD Rationale
Production agent workflows can execute for dozens of seconds or interact over multi-turn conversations. Without durable checkpointing and run persistence:
- If a container restarts, mid-flight reasoning state is completely lost.
- No historical audit trail exists for debugging hallucinations, monitoring cost, or auditing tool execution.
- Multi-turn conversation threads cannot be re-hydrated on different replicas.

#### 2. LLD Implementation
- **Database Schema**:
  - `agent_runs` table: `id (UUID PK)`, `organization_id (FK)`, `repo_id`, `task`, `status`, `total_steps`, `duration_ms`, `answer`, `citations (JSONB)`, `steps (JSONB)`, `created_at`, `completed_at`.
- **LangGraph Checkpointing (`checkpointer.py` / `langgraph_agent.py`)**:
  - Durable Postgres checkpointer storing state snapshots (serialized graph state, messages, node positions) under `thread_id = run_id`.
  - `PostgresCheckpointSaver` implementing LangGraph's checkpoint saver interface or integrated durable snapshot serializer.
  - In-memory fallback (`MemorySaver`) for ephemeral testing.
- **REST Contract**:
  - `POST /v1/agent/run`: Automatically persists the run record and writes final step trace and state to PostgreSQL upon completion.
  - `GET /v1/agent/runs/{run_id}`: Allows retrieving historical run output, tool reasoning trace, citation verification, and timing.

---

### Phase 5C: Redis Coordination - Rate Limits, Concurrency Controls, Idempotency & Caching

#### 1. HLD Rationale
- **Idempotency**: Network retries from clients can trigger duplicate agent runs or duplicate ingestions, costing money and corrupting data.
- **Rate Limiting**: Malicious or runaway clients can exhaust LLM budgets. Rate limits enforce fair multi-tenant quotas.
- **Concurrency Control**: LLM inference and agent tool execution consume heavy memory and concurrent connections. Bounding concurrent runs per tenant prevents resource starvation.
- **Multi-Tier Caching**: Repeated embedding calls for identical text and repeated queries waste provider latency and cost.

#### 2. LLD Implementation
- **Coordination Layer (`coordination.py`)**:
  - `RedisCoordinationService` (using `redis.asyncio` with pooling) + `InMemoryCoordinationService` fallback.
- **Idempotency Primitive**:
  - Evaluated on `Idempotency-Key` header.
  - Atomic reservation: `SET key "in_progress" NX EX 300`. If key exists and is "in_progress", return `409 Conflict` ("Request with this Idempotency-Key is currently being processed").
  - On request completion: save JSON response payload with 24-hour TTL (`EX 86400`). Subsequent identical requests return the cached response with `X-Cache: HIT-IDEMPOTENT`.
- **Sliding-Window Rate Limiter**:
  - Per organization / API key sliding window: max $N$ requests per minute. Returns `429 Too Many Requests` with `Retry-After: <seconds>` and `X-RateLimit-*` headers.
- **Tenant Concurrency Limiter**:
  - Distributed semaphore per tenant: max $K$ concurrent agent runs (e.g. 2 concurrent runs per organization). Returns `429 Too Many Requests` if tenant quota is actively consumed.
- **Embedding & Query Cache**:
  - SHA-256 hash of query text as cache key. Prevents repetitive embedding model calls.

---

### Phase 5D: Groq LLM Admission Control, Bounded Retries & Quota Resilience

#### 1. HLD Rationale
Groq provides fast cloud inference, but free and standard tiers enforce strict RPM (Requests Per Minute) and TPM (Tokens Per Minute) caps. Naive concurrent agent steps will immediately hit `429 Too Many Requests`. The system must:
- Pace LLM calls through an admission controller.
- Automatically retry on transient network errors (HTTP 500, 502, 503) and 429 with jittered exponential backoff.
- Gracefully fail with informative messages and proper HTTP status codes when provider quotas are genuinely exhausted.

#### 2. LLD Implementation
- **Admission Controller (`llm_admission.py`)**:
  - Leaky bucket / token-bucket pacer to throttle dispatch rate below provider RPM limits.
- **Bounded Retry Mechanism**:
  - Wrapped around `LLMClient.generate()`: 3 max attempts.
  - Delay formula: $T_{backoff} = \min(T_{max}, T_{base} 	imes 2^{	ext{attempt}}) \pm 	ext{jitter}$.
  - Respects provider `Retry-After` headers if present in 429 response.
- **Error Mapping**:
  - `LLMQuotaExhaustedError` -> Maps to HTTP 429 with descriptive tenant error.
  - `LLMTimeoutError` -> Maps to HTTP 504 Gateway Timeout.

---

### Phase 5E / Day 12: Production Observability, Health Checks, Docker & CI/CD

#### 1. HLD Rationale
To demonstrate senior-level deployment readiness, the service must be observable, containerized, orchestrated, and validated by automated CI/CD pipelines.

#### 2. LLD Implementation
- **Structured JSON Logging (`logging_config.py`)**:
  - JSON-formatted logs containing timestamp, log level, message, `request_id`, `tenant_id`, and `duration_ms`.
  - Request middleware generating or propagating `X-Request-ID`.
- **Observability & Metrics (`metrics.py`)**:
  - Prometheus `/metrics` endpoint exporting:
    - `http_requests_total{method, path, status}`
    - `http_request_duration_seconds{method, path}`
    - `llm_requests_total{provider, model, status}`
    - `llm_request_duration_seconds{provider, model}`
    - `agent_runs_active{organization_id}`
    - `ingestion_queue_depth{status}`
- **Comprehensive Health Probes (`/health`)**:
  - Deep readiness checks: validates database connectivity (`SELECT 1`), Redis ping, and vector store readiness.
- **Deployment Artifacts**:
  - Multi-stage `Dockerfile`: unprivileged user, lean dependencies, secure non-root runtime.
  - `docker-compose.yml`: API service, Ingestion Worker, PostgreSQL 16 with pgvector extension, and Redis 7 with persistent volume.
  - `.github/workflows/ci.yml`: Automated linting, type checks, and full regression test execution.
