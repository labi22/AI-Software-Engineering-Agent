"""Adapters bridging the agent's ToolRegistry and LLMClient to LangChain / LangGraph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Sequence

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from .agent_state import AgentState, Observation, ToolCall
from .llm import LLMClient, LLMRequest
from .tools import ToolRegistry, ToolSchema

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


class RegistryTool(BaseTool):
    """LangChain BaseTool wrapper around our project's native ToolRegistry."""

    name: str
    description: str
    tool_registry: Any = Field(exclude=True)
    agent_state: Any = Field(exclude=True)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Synchronous tool invocation (runs async in event loop if available)."""
        combined_args: dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            combined_args.update(args[0])
        combined_args.update(kwargs)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._arun(**combined_args), loop
                )
                return future.result()
            return loop.run_until_complete(self._arun(**combined_args))
        except RuntimeError:
            return asyncio.run(self._arun(**combined_args))

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Asynchronous tool execution preserving safety checks and state tracking."""
        combined_args: dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            combined_args.update(args[0])
        combined_args.update(kwargs)

        tool_call = ToolCall(tool_name=self.name, arguments=combined_args)
        observation: Observation = await self.tool_registry.execute(
            tool_call, self.agent_state
        )
        return observation.content


def adapt_tool_registry(
    tool_registry: ToolRegistry,
    agent_state: AgentState,
) -> list[BaseTool]:
    """Convert all registered tools in a ToolRegistry to LangChain BaseTool instances."""
    tools: list[BaseTool] = []
    for schema in tool_registry.list_schemas():
        tool = RegistryTool(
            name=schema.name,
            description=schema.description or f"Tool: {schema.name}",
            tool_registry=tool_registry,
            agent_state=agent_state,
        )
        tools.append(tool)
    return tools


class CustomChatModel(BaseChatModel):
    """LangChain BaseChatModel adapter wrapping our project's async LLMClient protocol."""

    llm_client: Any = Field(exclude=True)
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "custom_ai_swe_chat_model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> CustomChatModel:
        """Bind tools to the chat model in standard OpenAI function format."""
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": getattr(tool, "args", {}) or {},
                        },
                    }
                )
            elif isinstance(tool, dict) and "function" in tool:
                formatted_tools.append(tool)
            elif isinstance(tool, dict) and "name" in tool:
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )
        return CustomChatModel(llm_client=self.llm_client, bound_tools=formatted_tools)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the wrapped LLMClient."""
        converted_messages: list[dict[str, object]] = []
        system_instruction: str | None = None
        user_prompt: str = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = str(msg.content)
                converted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                user_prompt = str(msg.content)
                converted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                payload: dict[str, object] = {"role": "assistant", "content": msg.content or None}
                if msg.tool_calls:
                    payload["tool_calls"] = [
                        {
                            "id": tc.get("id") or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ]
                converted_messages.append(payload)
            elif isinstance(msg, ToolMessage):
                converted_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": str(msg.content),
                    }
                )
            else:
                converted_messages.append({"role": "user", "content": str(msg.content)})

        llm_request = LLMRequest(
            prompt=user_prompt or (messages[-1].content if messages else ""),
            system_instruction=system_instruction,
            messages=converted_messages,
            tools=self.bound_tools or None,
        )

        llm_response = await self.llm_client.generate(llm_request)

        # Build AIMessage from LLMResponse
        tool_calls: list[dict[str, Any]] = []

        # 1. Check native tool calls
        if llm_response.tool_calls:
            for tc in llm_response.tool_calls:
                tool_calls.append(
                    {
                        "name": tc.name,
                        "args": tc.arguments,
                        "id": tc.call_id,
                    }
                )
        else:
            # 2. Check XML <tool_call> tags (ReAct fallback)
            text = llm_response.text or ""
            match = _TOOL_CALL_PATTERN.search(text)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    tname = payload.get("name", "")
                    targs = payload.get("arguments", {})
                    if tname:
                        tool_calls.append(
                            {
                                "name": tname,
                                "args": targs,
                                "id": f"call_{tname}",
                            }
                        )
                except json.JSONDecodeError:
                    pass

        ai_message = AIMessage(
            content=llm_response.text or "",
            tool_calls=tool_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])
