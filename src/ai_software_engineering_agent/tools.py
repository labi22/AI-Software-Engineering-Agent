"""Tool registry, schemas, and built-in tool handlers for the agent loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .agent_state import AgentState, Observation, ToolCall
from .models import MetadataFilter, RetrievalStrategy


@dataclass(frozen=True)
class ToolSchema:
    """JSON-schema-style description of a tool for the LLM system prompt."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# Type alias for tool handler functions.
# Each handler receives (arguments, state) and returns the observation text.
ToolHandler = Callable[[dict[str, Any], AgentState], Awaitable[str]]


class ToolRegistry:
    """Registry mapping tool names to handler functions and their schemas."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._schemas: dict[str, ToolSchema] = {}

    def register(self, name: str, handler: ToolHandler, schema: ToolSchema) -> None:
        """Register a tool handler with its schema."""
        self._handlers[name] = handler
        self._schemas[name] = schema

    def get(self, name: str) -> ToolHandler | None:
        """Look up a tool handler by name."""
        return self._handlers.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._handlers

    def list_schemas(self) -> list[ToolSchema]:
        """Return all registered tool schemas."""
        return list(self._schemas.values())

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._handlers.keys())

    def format_for_prompt(self) -> str:
        """Render tool schemas as a text block for the LLM system prompt."""
        if not self._schemas:
            return "No tools are available."

        lines: list[str] = ["Available tools:"]
        for schema in self._schemas.values():
            lines.append(f"\n### {schema.name}")
            lines.append(f"Description: {schema.description}")
            if schema.parameters:
                params = schema.parameters.get("properties", {})
                required = set(schema.parameters.get("required", []))
                if params:
                    lines.append("Parameters:")
                    for pname, pinfo in params.items():
                        req_tag = " (required)" if pname in required else " (optional)"
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        lines.append(f"  - {pname}{req_tag}: {ptype} — {pdesc}")

        return "\n".join(lines)

    async def execute(
        self,
        tool_call: ToolCall,
        state: AgentState,
    ) -> Observation:
        """Execute a tool call and return an Observation."""
        handler = self.get(tool_call.tool_name)
        if handler is None:
            return Observation(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                content=f"Error: Unknown tool '{tool_call.tool_name}'. Available tools: {', '.join(self.list_names())}",
                is_error=True,
            )

        start = time.time()
        try:
            result_text = await handler(tool_call.arguments, state)
            duration_ms = (time.time() - start) * 1000
            return Observation(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                content=result_text,
                is_error=False,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            return Observation(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                content=f"Error executing {tool_call.tool_name}: {exc}",
                is_error=True,
                duration_ms=duration_ms,
            )


# ---------------------------------------------------------------------------
# Built-in tool schemas
# ---------------------------------------------------------------------------

SEARCH_CODE_SCHEMA = ToolSchema(
    name="search_code",
    description=(
        "Search the ingested codebase for relevant code snippets using hybrid retrieval "
        "(dense embeddings + BM25 keyword matching). Returns the top matching code chunks "
        "with file paths, line numbers, and symbol names."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query describing the code or concept to find.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-10).",
            },
            "strategy": {
                "type": "string",
                "description": "Retrieval strategy: 'dense', 'bm25', or 'hybrid'.",
            },
        },
        "required": ["query"],
    },
)

ANSWER_QUERY_SCHEMA = ToolSchema(
    name="answer_query",
    description=(
        "Retrieve relevant code from the codebase and generate a grounded answer with "
        "source citations. Use this when you have a clear question and want a complete "
        "cited answer in one step."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to answer about the codebase.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of code chunks to retrieve for context.",
            },
        },
        "required": ["query"],
    },
)

FINAL_ANSWER_SCHEMA = ToolSchema(
    name="final_answer",
    description=(
        "Submit your final answer to the user's task. Call this tool when you have "
        "gathered enough information to provide a complete response. This ends the "
        "agent loop."
    ),
    parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The complete final answer to present to the user.",
            },
        },
        "required": ["answer"],
    },
)


# ---------------------------------------------------------------------------
# Built-in tool handler factories
# ---------------------------------------------------------------------------

def create_search_code_handler(rag_service: Any) -> ToolHandler:
    """Create a search_code tool handler bound to a RAGService instance."""

    async def search_code_handler(arguments: dict[str, Any], state: AgentState) -> str:
        query = arguments.get("query", "")
        if not query:
            return "Error: 'query' argument is required."

        top_k = min(int(arguments.get("top_k", 5)), 10)
        strategy_str = arguments.get("strategy", "hybrid").lower()
        strategy = RetrievalStrategy(strategy_str) if strategy_str in ("dense", "bm25", "hybrid") else RetrievalStrategy.HYBRID

        results = await rag_service.retrieve(
            query=query,
            strategy=strategy,
            top_k=top_k,
        )

        if not results:
            return f"No code found for query: '{query}'"

        sections: list[str] = []
        for i, res in enumerate(results, 1):
            chunk = res.chunk
            symbol = f" (Symbol: {chunk.symbol_name})" if chunk.symbol_name else ""
            header = f"[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}{symbol} (score: {res.score:.4f})"
            sections.append(f"{header}\n```\n{chunk.content}\n```")

        return f"Found {len(results)} relevant code chunks:\n\n" + "\n\n".join(sections)

    return search_code_handler


def create_answer_query_handler(rag_service: Any) -> ToolHandler:
    """Create an answer_query tool handler bound to a RAGService instance."""

    async def answer_query_handler(arguments: dict[str, Any], state: AgentState) -> str:
        query = arguments.get("query", "")
        if not query:
            return "Error: 'query' argument is required."

        top_k = min(int(arguments.get("top_k", 5)), 10)

        try:
            response = await rag_service.answer_query(
                query=query,
                strategy=RetrievalStrategy.HYBRID,
                top_k=top_k,
            )
        except Exception as exc:
            return f"Error generating answer: {exc}"

        # Store citations in state for the final result
        state.citations.extend(response.citations)

        citation_strs = [c.formatted() for c in response.citations]
        citations_block = "\n".join(citation_strs) if citation_strs else "No citations found."

        return f"Answer: {response.answer}\n\nCitations:\n{citations_block}"

    return answer_query_handler


def create_final_answer_handler() -> ToolHandler:
    """Create a final_answer tool handler that terminates the agent loop."""

    async def final_answer_handler(arguments: dict[str, Any], state: AgentState) -> str:
        answer = arguments.get("answer", "")
        if not answer:
            return "Error: 'answer' argument is required."

        state.finish(answer, citations=state.citations)
        return answer

    return final_answer_handler


def create_default_tool_registry(rag_service: Any) -> ToolRegistry:
    """Create a ToolRegistry pre-populated with the default built-in tools."""
    registry = ToolRegistry()
    registry.register("search_code", create_search_code_handler(rag_service), SEARCH_CODE_SCHEMA)
    registry.register("answer_query", create_answer_query_handler(rag_service), ANSWER_QUERY_SCHEMA)
    registry.register("final_answer", create_final_answer_handler(), FINAL_ANSWER_SCHEMA)
    return registry


# Lazy imports for safe engineering tools to avoid circular dependencies
def get_engineering_tool_registry(
    repo_root: Any,
    allowed_roots: tuple[Any, ...] = (),
    rag_service: Any | None = None,
) -> ToolRegistry:
    """Convenience factory creating a ToolRegistry with all 6 safe engineering tools."""
    from .engineering_tools import EngineeringToolContext, create_engineering_tool_registry

    context = EngineeringToolContext(
        repo_root=repo_root,
        allowed_roots=allowed_roots,
        rag_service=rag_service,
    )
    return create_engineering_tool_registry(context)

