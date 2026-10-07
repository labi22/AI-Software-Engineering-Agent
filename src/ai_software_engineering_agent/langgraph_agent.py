"""LangGraph-based agent orchestration engine with compiled StateGraph."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .agent import AgentError
from .agent_state import (
    AgentResult,
    AgentState,
    AgentStatus,
    AgentStep,
    Observation,
    ToolCall,
)
from .config import Settings
from .langchain_adapters import CustomChatModel
from .llm import LLMClient
from .models import Citation
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict):
    """Typed state container flowing through the LangGraph StateGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    current_step: int
    max_steps: int
    status: str
    final_answer: str | None
    citations: list[Citation]
    trace_steps: list[AgentStep]
    start_time: float


class LangGraphAgent:
    """Agent orchestrator powered by LangGraph compiled StateGraph.

    Maintains full parity with the custom ReAct agent loop:
    - Preserves path traversal safety, process isolation, and output limits.
    - Bridges the project's ToolRegistry and LLMClient into LangGraph nodes.
    - Emits identical AgentResult / AgentStep traces for API & evaluation compatibility.
    - Includes memory checkpointing for state inspection and auditability.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        settings: Settings,
        max_steps: int | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.settings = settings
        self.max_steps = max_steps or settings.agent_max_steps
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()

        # Build ChatModel adapter with bound native tools
        self._chat_model = CustomChatModel(llm_client=self.llm_client)
        openai_tools = self.tool_registry.to_openai_tools()
        if openai_tools:
            self._chat_model = self._chat_model.bind_tools(openai_tools)

        # Build and compile the graph
        self.graph = self._build_graph()

    def _build_system_prompt(self) -> str:
        """Construct prompt instructions for LangGraph reasoning."""
        tools_block = self.tool_registry.format_for_prompt()
        return (
            "You are an expert AI software engineering agent. Solve the user's task by "
            "inspecting the codebase using the available tools.\n\n"
            "## Instructions\n"
            "1. Reason carefully about what information you need at each step.\n"
            "2. When you need to read or search code, invoke the appropriate tool.\n"
            "3. When you have collected enough information to completely answer the task, "
            "call the `final_answer` tool with your response in JSON: {'answer': '...'}.\n"
            "4. You may also provide a direct, comprehensive final answer when no tool calls are needed.\n"
            "5. Cite specific file paths and line numbers when referencing code.\n\n"
            f"## {tools_block}\n"
        )

    def _build_graph(self):
        """Construct and compile the LangGraph StateGraph."""
        workflow = StateGraph(AgentGraphState)

        # 1. Reasoner Node
        async def reasoner_node(state: AgentGraphState) -> dict[str, Any]:
            current_step = state["current_step"] + 1
            trace_steps = list(state.get("trace_steps", []))

            if current_step > state["max_steps"]:
                return {
                    "current_step": current_step,
                    "status": "failed",
                    "final_answer": f"Step limit exceeded ({state['max_steps']} steps).",
                }

            try:
                response = await self._chat_model.ainvoke(state["messages"])
            except Exception as exc:
                logger.error("LangGraph LLM call failed at step %d: %s", current_step, exc)
                return {
                    "current_step": current_step,
                    "status": "failed",
                    "final_answer": f"LLM generation error: {exc}",
                }

            # Record thinking step in trace
            thinking_content = response.content if isinstance(response.content, str) else str(response.content)
            trace_steps.append(
                AgentStep(
                    step_number=current_step,
                    status=AgentStatus.THINKING,
                    reasoning=thinking_content or "Decided next action.",
                )
            )

            # Check if LLM produced tool calls
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                # Direct answer without tool call — carry citations accumulated so far
                answer = thinking_content.strip() or "Task completed."
                return {
                    "messages": [response],
                    "current_step": current_step,
                    "status": "finished",
                    "final_answer": answer,
                    "trace_steps": trace_steps,
                    "citations": state.get("citations", []),
                }

            return {
                "messages": [response],
                "current_step": current_step,
                "trace_steps": trace_steps,
            }

        # 2. Tool Execution Node
        async def tools_node(state: AgentGraphState) -> dict[str, Any]:
            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", []) or []
            trace_steps = list(state.get("trace_steps", []))
            citations = list(state.get("citations", []))
            tool_messages: list[ToolMessage] = []
            status = state.get("status", "running")
            final_answer = state.get("final_answer")

            # Temporary agent state for citation collection & safety
            mock_agent_state = AgentState(task=state["task"], max_steps=state["max_steps"])
            mock_agent_state.citations = citations

            for call in tool_calls:
                tname = call.get("name", "")
                targs = call.get("args", {})
                cid = call.get("id", f"call_{uuid.uuid4().hex[:8]}")

                # Record action step
                action_call = ToolCall(tool_name=tname, arguments=targs, call_id=cid)
                trace_steps.append(
                    AgentStep(
                        step_number=state["current_step"],
                        status=AgentStatus.ACTING,
                        tool_call=action_call,
                    )
                )

                if tname == "final_answer":
                    # Final answer tool invoked
                    answer = targs.get("answer") or targs.get("content") or json.dumps(targs)
                    final_answer = str(answer)
                    status = "finished"
                    obs = Observation(
                        call_id=cid,
                        tool_name=tname,
                        content=final_answer,
                        is_error=False,
                    )
                    trace_steps.append(
                        AgentStep(
                            step_number=state["current_step"],
                            status=AgentStatus.OBSERVING,
                            observation=obs,
                        )
                    )
                    tool_messages.append(
                        ToolMessage(content=final_answer, tool_call_id=cid, name=tname)
                    )
                    break

                # Execute standard engineering or RAG tool
                obs = await self.tool_registry.execute(action_call, mock_agent_state)
                trace_steps.append(
                    AgentStep(
                        step_number=state["current_step"],
                        status=AgentStatus.OBSERVING,
                        observation=obs,
                    )
                )
                tool_messages.append(
                    ToolMessage(content=obs.content, tool_call_id=cid, name=tname)
                )

                # Collect any newly discovered citations
                citations = list(mock_agent_state.citations)

                if mock_agent_state.is_done():
                    final_answer = mock_agent_state.final_answer
                    status = "finished"
                    break

            return {
                "messages": tool_messages,
                "status": status,
                "final_answer": final_answer,
                "trace_steps": trace_steps,
                "citations": citations,
            }

        # 3. Conditional routing
        def route_after_reasoner(state: AgentGraphState) -> str:
            if state.get("status") in ("finished", "failed"):
                return "end"
            if state["current_step"] >= state["max_steps"]:
                return "end"
            last_message = state["messages"][-1]
            tool_calls = getattr(last_message, "tool_calls", None)
            if not tool_calls:
                return "end"
            return "tools"

        def route_after_tools(state: AgentGraphState) -> str:
            """Stop immediately after a terminal tool result.

            In particular, the ``final_answer`` tool sets ``status`` to
            ``finished``. Routing it back to the reasoner would spend another
            LLM request after the answer had already been committed.
            """
            if state.get("status") in ("finished", "failed"):
                return "end"
            return "reasoner"

        workflow.add_node("reasoner", reasoner_node)
        workflow.add_node("tools", tools_node)

        workflow.add_edge(START, "reasoner")
        workflow.add_conditional_edges(
            "reasoner",
            route_after_reasoner,
            {"tools": "tools", "end": END},
        )
        workflow.add_conditional_edges(
            "tools",
            route_after_tools,
            {"reasoner": "reasoner", "end": END},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    async def run(self, task: str, thread_id: str | None = None) -> AgentResult:
        """Execute the task through the compiled LangGraph workflow."""
        if not task or not task.strip():
            raise AgentError("Task text cannot be empty.")

        t0 = time.perf_counter()
        active_thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": active_thread_id}}

        initial_state: AgentGraphState = {
            "messages": [
                SystemMessage(content=self._build_system_prompt()),
                HumanMessage(content=task),
            ],
            "task": task,
            "current_step": 0,
            "max_steps": self.max_steps,
            "status": "running",
            "final_answer": None,
            "citations": [],
            "trace_steps": [],
            "start_time": t0,
        }

        final_state = await self.graph.ainvoke(initial_state, config=config)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status_val = final_state.get("status")
        answer = final_state.get("final_answer")

        if not answer:
            if final_state.get("current_step", 0) >= self.max_steps:
                status_val = "failed"
                answer = f"Step limit exceeded ({self.max_steps} steps). The agent could not complete the task within the allowed budget."
            else:
                answer = "The agent could not produce an answer."

        if status_val not in ("finished", "failed"):
            status_val = "finished" if final_state.get("final_answer") else "failed"

        return AgentResult(
            task=task,
            answer=answer,
            steps=final_state.get("trace_steps", []),
            total_steps=final_state.get("current_step", 0),
            status=status_val,
            duration_ms=elapsed_ms,
            citations=final_state.get("citations", []),
        )
