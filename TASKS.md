# AI Software Engineering Agent Task Tracker

This file is the working task board for the 14-day AI Software Engineering Agent build. Update it whenever a task is completed, deferred, or re-scoped.

Status markers:
- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[!]` Blocked or needs a decision

## Project Context

We are building a repository-agnostic AI Software Engineering Agent: a small Codex/Cursor-style assistant that can ingest an external software repository, retrieve relevant code and documentation, answer questions with citations, and eventually perform safe engineering tasks through tools.

Primary goal: become credible and interview-ready for GenAI Engineer, AI Engineer, LLM Engineer, Agentic AI Engineer, Applied AI Engineer, and ML Engineer - GenAI roles by learning through implementation.

Target architecture:
- Python project in this repository
- FastAPI-only interface initially
- OpenAI API as the first provider behind a provider-neutral interface
- Minimal custom orchestration before adding heavier frameworks
- Repository ingestion and chunking
- Embeddings and vector search
- RAG answers with citations
- Agent loop with state and tool selection
- Safe tools for code search, file reads, test execution, Python execution, and git diff
- MCP after tool calling is understood
- Evaluation pipeline for retrieval and answer quality
- FastAPI, Docker, logging, and configuration for production-style packaging

Target repositories stay separate from this repository:
- Primary local path: `C:\Users\saksh\Yield Curve Construction and Bond Valuation & Risk Lab`
- Advanced: `labi22/E-Commerce-Intelligence-Platform`
- Optional: another external Python repository to prove repository-agnostic behavior

## Guardrails

- [ ] Keep this agent repository separate from target repositories.
- [ ] Do not hard-code behavior to one target repository.
- [ ] Prefer simple, observable implementation before introducing abstractions.
- [ ] Inspect retrieved chunks and tool calls while debugging.
- [ ] Do not call the system agentic until it has tool selection plus an action/observation loop.
- [ ] Be honest in README/interviews about current scope and limitations.
- [ ] Prefer a small set of well-understood dependencies over broad framework sprawl.

## Day 1-2: Foundation And First LLM Call

- [x] Create initial Python project scaffold.
- [x] Add base package under `src/ai_software_engineering_agent`.
- [x] Add initial tests.
- [x] Add configuration loading for API keys, model names, allowed repository roots, and runtime settings.
- [x] Define an asynchronous LLM client abstraction and provider factory.
- [x] Implement the OpenAI Responses API adapter.
- [x] Add a FastAPI health endpoint and `POST /v1/generate` endpoint.
- [x] Add tests using a fake/mock LLM client.
- [x] Document request/response flow in README.

Acceptance checks:
- [ ] Project installs locally in editable mode.
- [x] Tests pass.
- [ ] A user can call `POST /v1/generate` and receive an LLM response.
- [ ] No secrets are committed.

Interview checkpoints:
- [ ] Explain tokens, context windows, temperature, prompts, and structured output.
- [ ] Explain why we start with an API model instead of local inference.

## Day 3-4: Repository Ingestion, Chunking, Embeddings, Basic RAG

- [x] Define repository input model for local paths and future repository URLs.
- [x] Implement file discovery with include/exclude rules.
- [x] Parse supported text/code files.
- [x] Create code/document chunking strategy.
- [x] Attach metadata to chunks: repo, path, language, symbol if available, line range.
- [x] Define embedding client abstraction.
- [x] Generate embeddings for chunks.
- [x] Set up PostgreSQL + pgvector storage from the first RAG milestone.
- [x] Store chunks and embeddings.
- [x] Implement semantic retrieval.
- [x] Assemble retrieved context into a prompt.
- [x] Return basic grounded answer.

Acceptance checks:
- [x] Ingest Target Repo 1 from an external path.
- [x] Retrieve relevant chunks for at least 5 known questions.
- [x] Answer includes citations to file paths and line ranges.

Interview checkpoints:
- [x] Explain embeddings and semantic similarity.
- [x] Explain chunking trade-offs for code.
- [x] Explain RAG flow and hallucination reduction.

## Day 5-6: Retrieval Quality And Citations

- [x] Add keyword/BM25-style search or another lexical retrieval baseline.
- [x] Add hybrid retrieval strategy.
- [x] Add metadata filters.
- [x] Add reranking or a simple scoring layer.
- [x] Add query rewriting or query expansion if needed.
- [x] Add citation formatting and source validation.
- [x] Add debug output for retrieved chunks.
- [x] Add retrieval quality experiments for Target Repo 1.

Acceptance checks:
- [x] Compare baseline retrieval and improved retrieval.
- [x] Record retrieval examples and failure cases.
- [x] RAG answers cite the chunks they actually used.

Interview checkpoints:
- [x] Explain top-k, recall, precision, MRR, reranking, and hybrid search.
- [x] Explain common RAG failure modes and how to debug them.

## Day 7: Agent Loop And State

- [x] Define agent state model.
- [x] Define action and observation records.
- [x] Implement minimal agent loop.
- [x] Add tool selection placeholder or LLM-driven function selection.
- [x] Add step limits and stopping conditions.
- [x] Add trace logging for each reasoning/action step.
- [x] Wire `POST /v1/agent/run` endpoint with request/response models and full trace serialisation.
- [x] Add tests for agent state, tool registry, loop behaviour, and API endpoint.

Acceptance checks:
- [x] Agent can decide between answering directly and using retrieval.
- [x] Agent trace is inspectable.
- [x] Loop fails safely when it cannot complete a task.

Interview checkpoints:
- [ ] Explain the difference between RAG and an agent.
- [ ] Explain planning, state, tools, observations, and termination.

## Day 8-9: Safe Engineering Tools

- [x] Implement `list_files`.
- [x] Implement `search_code`.
- [x] Implement `read_file`.
- [x] Implement `run_python`.
- [x] Implement `run_tests`.
- [x] Implement `git_diff`.
- [x] Add input schemas and validation for every tool.
- [x] Add path safety checks to keep tools inside approved target repositories.
- [x] Add timeouts and output limits.
- [x] Add error handling and tool-result normalization.
- [x] Add tests for safe and unsafe tool inputs.

Acceptance checks:
- [x] Agent can find relevant code, read it, and explain it.
- [x] Agent can run tests and summarize failures.
- [x] Unsafe paths and commands are rejected.

Interview checkpoints:
- [x] Explain function calling/tool schemas.
- [x] Explain safe execution boundaries and failure handling.

## Day 10: MCP

- [x] Learn MCP concepts through this project context.
- [x] Decide whether to expose repository tools through an MCP server, MCP client, or both (Decision: Implement dual-mode architecture: MCPServer + MCPClient/ToolAdapter).
- [x] Implement standard JSON-RPC 2.0 protocol models (`initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`).
- [x] Implement `MCPServer` exposing safe engineering tools (`list_files`, `read_file`, `search_code`, `run_python`, `run_tests`, `git_diff`) and repository resources (`repo://`).
- [x] Implement `MCPClient` and `MCPToolAdapter` allowing the autonomous `Agent` to consume external/internal MCP tools dynamically into its `ToolRegistry`.
- [x] Add stdio transport CLI runner and FastAPI endpoint (`POST /v1/mcp`) for external client integration (Claude Desktop, Cursor, IDEs).
- [x] Add comprehensive test suite (`tests/test_mcp.py`) verifying handshake, tool calling, resource reading, and agent integration.
- [x] Document why standardized tool/resource interfaces matter and prepare interview questions/answers.

Acceptance checks:
- [x] Demonstrate one MCP-backed capability (agent calling safe tools via MCP; external client querying repository tools/resources).
- [x] Explain MCP without relying on abstract jargon.

Interview checkpoints:
- [x] Explain what problem MCP solves.
- [x] Explain when MCP is useful versus ordinary function calling.

## Day 11: Evaluation

- [ ] Create 20-30 Target Repo 1 evaluation questions.
- [ ] Label expected relevant files/chunks where possible.
- [ ] Separate retrieval evaluation from answer evaluation.
- [ ] Implement retrieval metrics such as Recall@K and MRR.
- [ ] Implement answer evaluation rubric for relevance, faithfulness, and citation quality.
- [ ] Run evaluation and save results.
- [ ] Inspect failures and create improvement tasks.

Acceptance checks:
- [ ] Evaluation can be run repeatedly.
- [ ] Results are documented in a readable format.
- [ ] At least 3 failure cases are analyzed.

Interview checkpoints:
- [ ] Explain how to evaluate a RAG system.
- [ ] Explain grounding, faithfulness, latency, cost, and reliability trade-offs.

## Day 12: API, Docker, Configuration, Logging

- [ ] Create FastAPI app.
- [ ] Add endpoint for repository ingestion.
- [ ] Add endpoint for questions/agent runs.
- [ ] Add request/response models.
- [ ] Add structured logging.
- [ ] Add configuration via environment variables.
- [ ] Add Dockerfile.
- [ ] Add Docker Compose for API plus PostgreSQL/pgvector if needed.
- [ ] Add basic health check.

Acceptance checks:
- [ ] API starts locally.
- [ ] API can answer against an ingested repository.
- [ ] Docker workflow is documented.

Interview checkpoints:
- [ ] Explain deployment shape, config, logging, latency, cost, and reliability.

## Day 13: Portfolio Polish

- [ ] Improve README with project story and setup.
- [ ] Add architecture diagram.
- [ ] Add example questions and outputs.
- [ ] Add evaluation results.
- [ ] Add limitations and future work.
- [ ] Add screenshots or GIF if useful.
- [ ] Clean up dependency list.
- [ ] Confirm all tests pass.

Acceptance checks:
- [ ] A reviewer can understand the project in 5 minutes.
- [ ] A reviewer can run the main demo path.
- [ ] README does not overclaim production readiness.

## Day 14: Interview Mode

- [ ] Prepare 30-45 minute project walkthrough.
- [ ] Prepare concise resume/project framing.
- [ ] Drill LLM fundamentals.
- [ ] Drill embeddings and RAG.
- [ ] Drill agent loop and tool calling.
- [ ] Drill MCP basics.
- [ ] Drill RAG/LLM evaluation.
- [ ] Drill Python coding.
- [ ] Drill medium DSA patterns.
- [ ] Drill SQL basics to medium.
- [ ] Collect interview feedback and convert gaps into follow-up tasks.

Acceptance checks:
- [ ] Can explain every major component built so far.
- [ ] Can defend major design decisions and trade-offs.
- [ ] Can honestly describe limitations and next steps.

## Backlog

- [ ] Add LangGraph after the custom agent loop is understood.
- [ ] Add local Hugging Face embeddings if time permits.
- [ ] Add support for remote GitHub repository ingestion.
- [ ] Add more language parsers beyond Python.
- [ ] Add generated unit test workflow.
- [ ] Add patch proposal workflow with human approval.
- [ ] Add memory beyond per-run state.
- [ ] Add advanced UI if the API/CLI experience becomes too limiting.

## Current Next Tasks
 
 - [x] Add configuration loading.
 - [x] Define the asynchronous LLM client abstraction and factory.
 - [x] Implement the OpenAI adapter.
 - [x] Implement FastAPI `/health` and `/v1/generate` endpoints.
 - [x] Add mock-client and API tests.
 - [x] Implement repository ingestion, safety rules, and AST/line-aware chunking.
 - [x] Implement embedding abstraction and OpenAI embeddings adapter.
 - [x] Implement VectorStore abstraction, in-memory store, and pgvector store.
 - [x] Implement RAGService and FastAPI `/v1/repositories/ingest` and `/v1/rag/query`.
 - [x] Implement BM25 lexical index, Reciprocal Rank Fusion (RRF), and SymbolBoostReranker.
 - [x] Implement citation source verification and retrieval rank debug tracing.
 - [x] Day 7: Implement Agent Loop, State Management, Actions, Observations, Tool Selection, and `POST /v1/agent/run` endpoint.
 - [x] Day 8-9: Implement safe engineering tools: `list_files`, `search_code`, `read_file`, `run_python`, `run_tests`, `git_diff` with path safety checks, timeouts, and output limits.
 - [x] Day 10: Model Context Protocol (MCP) — `MCPServer`, `MCPClient`, `MCPToolAdapter`, `repo://` resources, JSON-RPC 2.0 engine, stdio transport, and `POST /v1/mcp` endpoint. 107/107 tests passing.
 - [ ] Day 11: Evaluation pipeline — retrieval metrics (Recall@K, MRR), answer evaluation rubric (relevance, faithfulness, citation quality), and failure case analysis.

## Decision Log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-08-25 | Build minimal custom orchestration first. | The plan prioritizes understanding the agent loop before adding frameworks. |
| 2026-08-25 | Keep target repositories external. | The agent must be repository-agnostic and portfolio-ready. |
| 2026-08-25 | Use an API-only FastAPI interface for the first milestone. | The finished project needs an API, and keeping the first vertical slice focused reduces duplicated interface work. |
| 2026-08-25 | Use the OpenAI API as the first LLM provider. | It is a direct fit for Python, FastAPI, later tool calling, and the planned agent features. |
| 2026-08-25 | Do not include GitHub Copilot as a provider adapter. | It requires an authenticated Copilot CLI/runtime and subscription. |
| 2026-08-25 | Use the existing local bond valuation project as Target Repo 1. | It is an external local repository suitable for validating repository-agnostic ingestion. |
| 2026-08-25 | Use PostgreSQL + pgvector from the first RAG milestone. | The project should build directly on its chosen durable vector storage layer. |
| 2026-08-27 | Use `<tool_call>` XML tags for tool invocation instead of OpenAI function-calling JSON mode. | Keeps the agent portable across LLM providers and avoids SDK-specific structured output requirements. |
| 2026-08-27 | Fix: check `is_done()` before `add_observation_step()` in the agent loop. | `add_observation_step()` overwrites status to `OBSERVING`, masking the `FINISHED` status set by `final_answer` inside `execute()`. The `already_done` guard preserves terminal state across the observation recording boundary. |
| 2026-09-03 | Implement path boundary containment via canonical path resolution (`resolve().is_relative_to()`). | Blocks directory traversal (`../`), null-byte injection, and symlink breakouts to protect host filesystem. |
| 2026-09-05 | Implement self-contained, zero-dependency JSON-RPC 2.0 MCP protocol engine with dual-mode architecture (Server + Client/Adapter). | Demonstrates deep protocol mastery for interviews, avoids brittle external SDK dependencies, enables native offline testing, and allows our Agent to seamlessly consume both internal and external MCP tools. |

## Open Questions

- [x] Which LLM provider should be used for the first API call? OpenAI API only; GitHub Copilot is excluded because it requires a subscription.
- [x] Should the first interface be CLI-only, API-only, or both? API-only via FastAPI.
- [x] Where will Target Repo 1 live locally during development? `C:\Users\saksh\Yield Curve Construction and Bond Valuation & Risk Lab`.
- [x] Should pgvector be required immediately, or should we use an in-memory vector store for the first RAG milestone and migrate afterward? Use PostgreSQL + pgvector from the first RAG milestone.
