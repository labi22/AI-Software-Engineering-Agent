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

## Next Milestone: Day 8-9 — Safe Engineering Tools

1. Implement `list_files` — directory listing scoped to approved repository roots.
2. Implement `read_file` — file content retrieval with path safety checks and size limits.
3. Implement `search_code` — text/regex search within a repository (distinct from RAG retrieval).
4. Implement `run_python` — sandboxed Python snippet execution with timeout and output truncation.
5. Implement `run_tests` — invoke the project test runner and capture structured results.
6. Implement `git_diff` — safe read-only git diff for inspecting recent changes.
7. Add input validation and path confinement for every tool (reject paths outside `allowed_repository_roots`).
8. Add timeouts and output length limits to prevent runaway executions.
9. Register new tools in `create_default_tool_registry`.
10. Add tests for safe and unsafe inputs for every new tool.
