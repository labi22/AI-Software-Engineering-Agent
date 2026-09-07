# AI Software Engineering Agent

A repository-agnostic AI software engineering assistant that scans, ingests, indexes, and retrieves code to answer technical engineering questions with precise source and line-number citations — and an agentic ReAct loop that reasons across multiple retrieval steps to solve complex tasks.

## Features

- **FastAPI REST API**: Endpoints for text generation, repository ingestion, grounded hybrid RAG querying, and agentic task execution.
- **Repository Ingestion & Safe Discovery**: Scans target codebases with security guardrails (`ALLOWED_REPOSITORY_ROOTS`), ignoring binary, build, cache, and lock files.
- **Syntax & Line-Aware Code Chunking**: AST-based chunking for Python (classes, functions, methods) and sliding-window chunking for documentation and other files, tracking exact start and end line ranges.
- **BM25 Lexical Search Engine**: In-memory Okapi BM25 index with a code tokenizer handling `snake_case`, `camelCase`, and code identifiers.
- **Hybrid Retrieval (Reciprocal Rank Fusion)**: Blends dense semantic embeddings and sparse lexical BM25 rankings ($k=60$) with an exact symbol definition reranker.
- **Metadata Filtering**: Scopes queries by repository ID, file path glob patterns (e.g. `models/*.py`), programming language, or named code symbols only.
- **Citation Verification & Grounding**: Automatically verifies line-range overlap for LLM citations against retrieved context to flag hallucinations.
- **ReAct Agent Loop**: Multi-step reasoning agent that selects tools, observes results, and iterates until it has a complete answer — with full step-by-step trace in every response.
- **Safe Engineering Tools**: Suite of 6 secure tools (`list_files`, `read_file`, `search_code`, `run_python`, `run_tests`, `git_diff`) with canonical path validation, process sandboxing, timeouts, and output limits.
- **Model Context Protocol (MCP)**: Standards-based JSON-RPC 2.0 interface exposing all engineering tools and repository files to external clients (Claude Desktop, Cursor, IDEs) via `POST /v1/mcp` and a stdio transport loop. The agent can also consume tools from any external MCP server via `MCPClient` + `MCPToolAdapter`.
- **Provider-Neutral Architecture**: Supports OpenAI (`gpt-5.2`, `text-embedding-3-small`), PostgreSQL with `pgvector`, and 100% offline fake mocks for development/testing.

## Architecture Overview

```
POST /v1/agent/run
       │
       ▼
  Agent.run(task)
       │
  ┌────┴─────────────────────────────────────┐
  │            ReAct Loop                    │
  │                                          │
  │  THINK: LLM reasons about the task       │
  │     │                                    │
  │     ├─ <tool_call> found ──► ACT         │
  │     │                          │         │
  │     │                       OBSERVE      │
  │     │                       (tool runs)  │
  │     │                          │         │
  │     └──────────── loop ◄───────┘         │
  │                                          │
  │  or: plain text ──► direct answer        │
  │  or: step budget exhausted ──► ERROR     │
  └──────────────────────────────────────────┘
       │
       ▼
  AgentResult  { answer, status, steps[], citations[], duration_ms }
```

## Prerequisites

- Python 3.11+
- An OpenAI API key (for live inference; offline mock mode available for tests)
- PostgreSQL + pgvector (optional for durable vector persistence)

## Setup & Local Run

Install the project with development dependencies:

```powershell
python -m pip install -e .[dev]
```

Configure your environment:

```powershell
Copy-Item .env.example .env
$env:OPENAI_API_KEY = "your_openai_api_key"
$env:ALLOWED_REPOSITORY_ROOTS = "D:/Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab"
```

Start the service:

```powershell
uvicorn ai_software_engineering_agent.app:app --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## API Usage Examples

### 1. Ingest a Repository

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/repositories/ingest" `
  -ContentType "application/json" `
  -Body '{"repository_path": "D:/Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab", "repo_id": "bond-lab"}'
```

Response:
```json
{
  "repo_id": "bond-lab",
  "files_scanned": 27,
  "files_parsed": 27,
  "chunks_created": 166,
  "total_tokens": 144464,
  "status": "success"
}
```

### 2. Run the Agent on a Task

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/agent/run" `
  -ContentType "application/json" `
  -Body '{
    "task": "How is zero_rate calculated in the yield curve module? Show me the relevant code and run the tests to verify it.",
    "repository_path": "D:/Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab",
    "max_steps": 10
  }'
```

Response:
```json
{
  "request_id": "a3f1e2b4-...",
  "task": "How is zero_rate calculated...",
  "answer": "The zero_rate function in yield_curve/rates.py (lines 15-45) bootstraps spot rates...",
  "status": "finished",
  "total_steps": 3,
  "duration_ms": 1842.5,
  "steps": [
    {
      "step_number": 1,
      "status": "thinking",
      "reasoning": "I should search for zero_rate in the codebase first.",
      "tool_name": null,
      "tool_arguments": null,
      "observation": null,
      "is_error": false
    },
    {
      "step_number": 1,
      "status": "acting",
      "reasoning": null,
      "tool_name": "search_code",
      "tool_arguments": {"query": "zero_rate yield curve"},
      "observation": null,
      "is_error": false
    },
    {
      "step_number": 1,
      "status": "observing",
      "reasoning": null,
      "tool_name": null,
      "tool_arguments": null,
      "observation": "Found 3 relevant code chunks: [1] yield_curve/rates.py:15-45...",
      "is_error": false
    }
  ],
  "citations": [
    {
      "file_path": "yield_curve/rates.py",
      "start_line": 15,
      "end_line": 45,
      "symbol_name": "zero_rate",
      "is_verified": true
    }
  ]
}
```

### 3. Query Codebase with Hybrid Retrieval & Verified Citations

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/rag/query" `
  -ContentType "application/json" `
  -Body '{
    "query": "How is zero-coupon yield curve bootstrapped?",
    "repo_id": "bond-lab",
    "top_k": 5,
    "retrieval_strategy": "hybrid",
    "path_patterns": ["*.py"],
    "include_debug_info": true
  }'
```

Response:
```json
{
  "query": "How is zero-coupon yield curve bootstrapped?",
  "answer": "The yield curve is bootstrapped iteratively across par instruments in [yield_curve/rates.py:15-45]...",
  "citations": [
    {
      "file_path": "yield_curve/rates.py",
      "start_line": 15,
      "end_line": 45,
      "symbol_name": "zero_rate",
      "is_verified": true
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "chk_123",
      "file_path": "yield_curve/rates.py",
      "start_line": 15,
      "end_line": 45,
      "symbol_name": "zero_rate",
      "language": "python",
      "score": 0.1964,
      "debug_info": {
        "dense_rank": 1,
        "bm25_rank": 1,
        "rerank_boost": 0.25,
        "fusion_score": 0.0328
      }
    }
  ],
  "model": "gpt-5.2",
  "provider": "openai",
  "strategy_used": "hybrid"
}
```

### 4. Call Tools and Read Resources via MCP

```powershell
# List available tools
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/mcp" `
  -ContentType "application/json" `
  -Headers @{"X-Repository-Path"="D:/bond-lab"; "X-Repo-ID"="bond-lab"} `
  -Body '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

# Call a tool
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/mcp" `
  -ContentType "application/json" `
  -Headers @{"X-Repository-Path"="D:/bond-lab"; "X-Repo-ID"="bond-lab"} `
  -Body '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"read_file","arguments":{"file_path":"yield_curve/rates.py"}}}'

# List repository resources
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/mcp" `
  -ContentType "application/json" `
  -Headers @{"X-Repository-Path"="D:/bond-lab"; "X-Repo-ID"="bond-lab"} `
  -Body '{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}'
```

Response (tools/list):
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {"name": "list_files", "description": "List files and directories...", "inputSchema": {...}},
      {"name": "read_file",  "description": "Read file content...",          "inputSchema": {...}},
      {"name": "search_code","description": "Perform a fast text or regex grep...", "inputSchema": {...}},
      {"name": "run_python", "description": "Execute a Python script...",    "inputSchema": {...}},
      {"name": "run_tests",  "description": "Run the repository test suite...", "inputSchema": {...}},
      {"name": "git_diff",   "description": "View read-only git diff...",    "inputSchema": {...}}
    ]
  }
}
```

### 5. Direct Generation

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/generate" `
  -ContentType "application/json" `
  -Body '{"prompt": "Explain the purpose of a yield curve in two sentences."}'
```

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Credential for OpenAI LLM and embeddings | Required for live OpenAI calls |
| `LLM_PROVIDER` | Selected LLM provider (`openai` or `fake`) | `openai` |
| `LLM_MODEL` | LLM model identifier | `gpt-5.2` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | OpenAI request timeout | `30` |
| `ALLOWED_REPOSITORY_ROOTS` | Comma-separated approved paths for ingestion/tools | Empty |
| `EMBEDDING_PROVIDER` | Embedding provider (`openai` or `fake`) | `openai` |
| `EMBEDDING_MODEL` | Embedding model identifier | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Embedding vector dimension | `1536` |
| `VECTOR_STORE_TYPE` | Vector store type (`memory` or `pgvector`) | `memory` |
| `DATABASE_URL` | PostgreSQL connection URL for pgvector | `None` |
| `CHUNK_SIZE_LINES` | Maximum line count per code chunk | `60` |
| `CHUNK_OVERLAP_LINES` | Line overlap for sliding-window chunking | `15` |
| `AGENT_MAX_STEPS` | Default step budget for the ReAct agent loop | `10` |

## Tests

Run the full automated test suite offline:

```powershell
pytest
```

All tests use injected fake LLM clients and in-memory vector stores — no OpenAI API key or database required. **107 tests, all passing.**

## Project Structure

```
src/ai_software_engineering_agent/
├── agent.py            # ReAct agent loop, prompt builder, tool call parser
├── agent_state.py      # AgentStatus, ToolCall, Observation, AgentStep, AgentState, AgentResult
├── tools.py            # ToolSchema, ToolRegistry, built-in tool handlers
├── engineering_tools.py# 6 safe engineering tools + EngineeringToolContext
├── mcp.py              # MCPServer, MCPClient, MCPToolAdapter, JSON-RPC 2.0 engine, stdio transport
├── app.py              # FastAPI app — all HTTP endpoints including /v1/agent/run and /v1/mcp
├── config.py           # Settings loaded from environment variables
├── llm.py              # LLM client abstraction (OpenAI adapter + fake)
├── embeddings.py       # Embedding client abstraction (OpenAI adapter + fake)
├── rag.py              # RAGService: ingest, retrieve, answer_query, citation validation
├── retrieval.py        # HybridRetriever, RRF fusion, SymbolBoostReranker, QueryExpander
├── lexical.py          # CodeTokenizer, BM25Index
├── ingestion.py        # Repository scanner, AST chunker, sliding-window chunker
├── models.py           # Shared dataclasses: Citation, CodeChunk, RetrievalResult, etc.
├── safety.py           # Path confinement, git ref sanitization, pytest arg allowlist
└── vector_store.py     # VectorStore abstraction (InMemory + pgvector)

tests/
├── test_agent.py                   # Day 1-2: keyword extraction, plan builder
├── test_agent_loop.py              # Day 7: AgentState, ToolRegistry, Agent loop, /v1/agent/run
├── test_agent_tools_integration.py # Day 8-9: end-to-end agent with engineering tools
├── test_engineering_tools.py       # Day 8-9: list_files, read_file, search_code, run_python, run_tests, git_diff
├── test_safety.py                  # Day 8-9: path validation, git ref sanitization, pytest arg allowlist
├── test_mcp.py                     # Day 10: MCPServer, MCPClient, MCPToolAdapter, /v1/mcp endpoint
├── test_app.py                     # Day 1-6: all other endpoints, settings, LLM adapter
├── test_citations.py               # Day 5-6: citation verification
├── test_hybrid_retrieval.py        # Day 5-6: RRF, symbol boost, hybrid retriever
├── test_ingestion.py               # Day 3-4: chunking, file discovery
├── test_lexical.py                 # Day 5-6: BM25 index, code tokenizer
├── test_rag.py                     # Day 3-4: RAGService, ingest, retrieve
└── test_vector_store.py            # Day 3-4: in-memory and pgvector stores
```
